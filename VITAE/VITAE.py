import enum
import warnings
from typing import Optional

import VITAE.model as model 
import VITAE.train as train 
from VITAE.inference import Inferer
from VITAE.utils import load_data, clustering, get_igraph, leidenalg_igraph, \
   DE_test, _comp_dist
from VITAE.metric import topology, get_GRI
import tensorflow as tf

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
import scanpy as sc


class VITAE():
    """
    Variational Inference for Trajectory by AutoEncoder.
    """
    def __init__(self):
        self.dict_method_scname = {
            'PCA' : 'X_pca',
            'UMAP' : 'X_umap',
            'TSNE' : 'X_tsne',
            'diffmap' : 'X_diffmap',
            'draw_graph' : 'X_draw_graph_fa'
        }
    
    def initialize(self, adata: sc.AnnData, 
               covariates = None,
               model_type: str = 'Gaussian',
               npc: int = 64,
               adata_layer_counts = None,
               copy_adata: bool = False,
               hidden_layers = [16],
               latent_space_dim: int = 8):
        '''
        Get input data for model. Data need to be first processed using scancy and stored as an AnnData object
         The 'UMI' or 'non-UMI' model need the original count matrix, so the count matrix need to be saved in
         adata.layers in order to use these models.


        Parameters
        ----------
        adata : sc.AnnData
            The scanpy AnnData object. 
        covariates : list, optional
            A list of names of covariate vectors that are stored in adata.obs
        model_type : str, optional
            'UMI', 'non-UMI' and 'Gaussian', default is 'Gaussian'. 
        npc : int, optional
            The number of PCs to use when model_type is 'Gaussian'. The default is 64.
        adata_layer_counts: str, optional
            the key name of adata.layers that stores the count data if model_type is
            'UMI' or 'non-UMI'
        copy_adata: bool, optional
        hidden_layers : list, optional
            The list of dimensions of layers of autoencoder between latent space and original space.
        latent_space_dim : int, optional
            The dimension of latent space.
            

        Returns
        -------
        None.

        '''
        
        if model_type != 'Gaussian':
            if adata_layer_counts is None:
                raise ValueError("need to provide the name in adata.layers that stores the raw count data")
        
        
        if copy_adata:
            self.adata = adata.copy()
        else:
            self.adata = adata
        if covariates is not None:
            self.c_score = adata.obs[covariates].to_numpy()
        else:
            self.c_score = None
        
        self.model_type = model_type
        self._adata = sc.AnnData(X = self.adata.X, var = self.adata.var)
        self._adata.obs = self.adata.obs
        self._adata.uns = self.adata.uns
 
    
        if model_type == 'Gaussian':
            sc.tl.pca(adata, n_comps = npc)
            self.X_input = self.X_output = adata.obsm['X_pca']
            self.scale_factor = np.ones(self.X_output.shape[0])
        else:
            self.X_input = adata.X[:, adata.var.highly_variable]
            self.X_output = adata.layers[adata_layer_counts][ :, adata.var.highly_variable]
            self.scale_factor = np.sum(self.X_output, axis=1, keepdims=True)/1e4
            
        self.dimensions = hidden_layers
        self.dim_latent = latent_space_dim
    
        self.vae = model.VariationalAutoEncoder(
            self.X_output.shape[1], self.dimensions,
            self.dim_latent, self.model_type,
            False if self.c_score is None else True
            )
        
        if hasattr(self, 'inferer'):
            delattr(self, 'inferer')
        
        
## TODO: should we convert everything to dense matrix?
## TODO: Add load_model and save_model back if needed
        

    def pre_train(self, test_size = 0.1, random_state: int = 0,
            learning_rate: float = 1e-2, batch_size: int = 256, L: int = 1, alpha: float = 0.10,
            num_epoch: int = 300, num_step_per_epoch: Optional[int] = None,
            early_stopping_patience: int = 20, early_stopping_tolerance: float = 1.0,
            path_to_weights: Optional[str] = None):
        '''Pretrain the model with specified learning rate.

        Parameters
        ----------
        test_size : float or int, optional
            The proportion or size of the test set.
        random_state : int, optional
            The random state for data splitting.
        learning_rate : float, optional
            The initial learning rate for the Adam optimizer.
        batch_size : int, optional 
            The batch size for pre-training.  Default is 256. Set to 32 if number of cells is small (less than 1000)
        L : int, optional 
            The number of MC samples.
        alpha : float, optional
            The value of alpha in [0,1] to encourage covariate adjustment. Not used if there is no covariates.
        num_epoch : int, optional 
            The maximum number of epochs.
        num_step_per_epoch : int, optional 
            The number of step per epoch, it will be inferred from number of cells and batch size if it is None.            
        early_stopping_patience : int, optional 
            The maximum number of epochs if there is no improvement.
        early_stopping_tolerance : float, optional 
            The minimum change of loss to be considered as an improvement.
        path_to_weights : str, optional 
            The path of weight file to be saved; not saving weight if None.
        '''                    

        id_train, id_test = train_test_split(
                                np.arange(self.X_input.shape[0]), 
                                test_size=test_size, 
                                random_state=random_state)
        if num_step_per_epoch is None:
            num_step_per_epoch = len(id_train)//batch_size+1
        self.train_dataset = train.warp_dataset(self.X_input[id_train].astype(tf.keras.backend.floatx()), 
                                                None if self.c_score is None else self.c_score[id_train].astype(tf.keras.backend.floatx()),
                                                batch_size, 
                                                self.X_output[id_train].astype(tf.keras.backend.floatx()), 
                                                self.scale_factor[id_train].astype(tf.keras.backend.floatx()))
        self.test_dataset = train.warp_dataset(self.X_input[id_test], 
                                                None if self.c_score is None else self.c_score[id_test].astype(tf.keras.backend.floatx()),
                                                batch_size, 
                                                self.X_output[id_test].astype(tf.keras.backend.floatx()), 
                                                self.scale_factor[id_test].astype(tf.keras.backend.floatx()))
        self.vae = train.pre_train(
            self.train_dataset,
            self.test_dataset,
            self.vae,
            learning_rate,                        
            L, alpha,
            num_epoch,
            num_step_per_epoch,
            early_stopping_patience,
            early_stopping_tolerance,
            0)
        
        self.update_z()

        if path_to_weights is not None:
            self.save_model(path_to_weights)
            

    def update_z(self):
        self.z = self.get_latent_z()        
        self._adata_z = sc.AnnData(self.z)
        sc.pp.neighbors(self._adata_z)

            
    def get_latent_z(self):
        ''' get the current latent space z

        Returns
        ----------
        z : np.array
            \([N,d]\) The latent means.
        ''' 
        c = None if self.c_score is None else self.c_score
        return self.vae.get_z(self.X_input, c)
            
    
    def visualize_latent(self, method: str = "UMAP", 
                         color = None, **kwargs):
        '''
        visualize the current latent space z using the scanpy visualization tools

        Parameters
        ----------
        method : str, optional
            Visualization method to use. The default is "draw_graph" (the FA plot). Possible choices include "PCA", "UMAP", 
            "diffmap", "TSNE" and "draw_graph"
        color : TYPE, optional
            Keys for annotations of observations/cells or variables/genes, e.g., 'ann1' or ['ann1', 'ann2'].
            The default is None. Same as scanpy.
        **kwargs :  
            Extra key-value arguments that can be passed to scanpy plotting functions (scanpy.pl.XX).   

        Returns
        -------
        None.

        '''
          
        if method not in ['PCA', 'UMAP', 'TSNE', 'diffmap', 'draw_graph']:
            raise ValueError("visualization method should be one of 'PCA', 'UMAP', 'TSNE', 'diffmap' and 'draw_graph'")
        
        temp = list(self._adata_z.obsm.keys())
        if method == 'PCA' and not 'X_pca' in temp:
            print("Calculate PCs ...")
            sc.tl.pca(self._adata_z)
        elif method == 'UMAP' and not 'X_umap' in temp:  
            print("Calculate UMAP ...")
            sc.tl.umap(self._adata_z)
        elif method == 'TSNE' and not 'X_tsne' in temp:
            print("Calculate TSNE ...")
            sc.tl.tsne(self._adata_z)
        elif method == 'diffmap' and not 'X_diffmap' in temp:
            print("Calculate diffusion map ...")
            sc.tl.diffmap(self._adata_z)
        elif method == 'draw_graph' and not 'X_draw_graph_fa' in temp:
            print("Calculate FA ...")
            sc.tl.draw_graph(self._adata_z)
            
  
        self._adata.obsp = self._adata_z.obsp
#        self._adata.uns = self._adata_z.uns
        self._adata.obsm = self._adata_z.obsm
    
        if method == 'PCA':
            axes = sc.pl.pca(self._adata, color = color, **kwargs)
        elif method == 'UMAP':            
            axes = sc.pl.umap(self._adata, color = color, **kwargs)
        elif method == 'TSNE':
            axes = sc.pl.tsne(self._adata, color = color, **kwargs)
        elif method == 'diffmap':
            axes = sc.pl.diffmap(self._adata, color = color, **kwargs)
        elif method == 'draw_graph':
            axes = sc.pl.draw_graph(self._adata, color = color, **kwargs)
            
        return axes



    def init_latent_space(self, cluster_label = None, log_pi = None, res: float = 1.0, 
                          ratio_prune=0.0):
        '''Initialize the latent space.

        Parameters
        ----------
        cluster_label : str, optional
            the name of vector of labels that can be found in self.adata.obs. 
            Default is None, which will perform leiden clustering on the pretrained z to get clusters
        mu : np.array, optional
            \([d,k]\) The value of initial \(\\mu\).
        log_pi : np.array, optional
            \([1,K]\) The value of initial \(\\log(\\pi)\).
        res: 
            The resolution of leiden clustering, which is a parameter value controlling the coarseness of the clustering. 
            Higher values lead to more clusters. Deafult is 1.
        ratio_prune : float, optional
            The ratio of edges to be removed before estimating.
        '''   
    
        
        if cluster_label is None:
            print("Perform leiden clustering on the latent space z ...")
            g = get_igraph(self.z)
            labels = leidenalg_igraph(g, res = res)
            self.adata.obs['vitae_init_clustering'] = labels
            self.adata.obs['vitae_init_clustering'] = self.adata.obs['vitae_init_clustering'].astype('category')
            print("Clustering labels saved as 'vitae_init_clustering' in self.adata.obs.")
            cluster_label = 'vitae_init_clustering'
        
        n_clusters = np.unique(self.adata.obs[cluster_label]).shape[0]
        cluster_labels = self.adata.obs[cluster_label].to_numpy()        
        uni_cluster_labels = list(self.adata.obs[cluster_label].cat.categories)
        if not hasattr(self, 'z'):
            self.update_z()        
        z = self.z
        mu = np.zeros((z.shape[1], n_clusters))
        for i,l in enumerate(uni_cluster_labels):
            mu[:,i] = np.mean(z[cluster_labels==l], axis=0)
   #         mu[:,i] = z[cluster_labels==l][np.argmin(np.mean((z[cluster_labels==l] - mu[:,i])**2, axis=1)),:]
        if (log_pi is None) and (cluster_labels is not None) and (n_clusters>3):                         
            n_states = int((n_clusters+1)*n_clusters/2)
            d = _comp_dist(z, cluster_labels, mu.T)

            C = np.triu(np.ones(n_clusters))
            C[C>0] = np.arange(n_states)
            C = C.astype(int)

            log_pi = np.zeros((1,n_states))
            log_pi[0, C[np.triu(d)>np.quantile(d[np.triu_indices(n_clusters, 1)], 1-ratio_prune)]] = - np.inf

        self.n_clusters = n_clusters
        self.init_labels = cluster_labels
        # Not sure if storing the this will be useful
        # self.init_labels_name = cluster_label
        self.labels_map = pd.DataFrame.from_dict(
            {i:label for i,label in enumerate(uni_cluster_labels)}, 
            orient='index', columns=['label_names'], dtype=str
            )
        self.vae.init_latent_space(n_clusters, mu, log_pi)
        self.inferer = Inferer(self.n_clusters)


    def update_latent_space(self, dist: float=0.5):
        pi = tf.nn.softmax(self.vae.latent_space.pi).numpy()
        mu = self.vae.latent_space.mu.numpy()    
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=dist,
            linkage='complete'
            ).fit(mu.T/np.sqrt(mu.shape[0]))
        n_clusters = clustering.n_clusters_   

        if n_clusters<self.n_clusters:      
            print("Aggregate clusters ...")
            mu_new = np.empty((self.dim_latent, n_clusters))
            C = np.zeros((self.n_clusters, self.n_clusters))
            C[np.triu_indices(self.n_clusters, 0)] = pi
            C = np.triu(C, 1) + C.T
            C_new = np.zeros((n_clusters, n_clusters))
            
            labels_map_new = {}
            for i in range(n_clusters):                       
                # update label map: int->str
                labels_map_new[i] = self.labels_map.loc[clustering.labels_==i, 'label_names'].str.cat(sep=',')
                if np.sum(clustering.labels_==i)>1:
                    print('Merge %s'%labels_map_new[i])
                # mean of the aggregated cluster means
                mu_new[:, i] = np.mean(mu[:,clustering.labels_==i], axis=-1)
                # sum of the aggregated pi's
                C_new[i, i] = np.sum(np.triu(C[clustering.labels_==i,:][:,clustering.labels_==i]))
                for j in range(i+1, n_clusters):
                    C_new[i, j] = np.sum(C[clustering.labels_== i, :][:, clustering.labels_==j])
            C_new = np.triu(C_new,1) + C_new.T

            pi_new = C_new[np.triu_indices(n_clusters)]
            log_pi_new = np.log(pi_new, out=np.ones_like(pi_new)*(-np.inf), where=(pi_new!=0)).reshape((1,-1))
            self.n_clusters = n_clusters
            self.labels_map = pd.DataFrame.from_dict(
                labels_map_new, orient='index', columns=['label_names'], dtype=str
            )
            self.vae.init_latent_space(self.n_clusters, mu_new, log_pi_new)
            self.inferer = Inferer(self.n_clusters)  



    def train(self, stratify = False, test_size = 0.1, random_state: int = 0,
            learning_rate: float = 1e-3, batch_size: int = 256, 
            L: int = 1, alpha: float = 0.10, beta: float = 2, 
            num_epoch: int = 300, num_step_per_epoch: Optional[int] =  None,
            early_stopping_patience: int = 5, early_stopping_tolerance: float = 1.0, early_stopping_warmup: int = 0,
            path_to_weights: Optional[str] = None, **kwargs):
        '''Train the model.

        Parameters
        ----------
        stratify : np.array, None, or False
            If an array is provided, or `stratify=None` and `self.labels` is available, then they will be used to perform stratified shuffle splitting. Otherwise, general shuffle splitting is used. Set to `False` if `self.labels` is not intended for stratified shuffle splitting.
        test_size : float or int, optional
            The proportion or size of the test set.
        random_state : int, optional
            The random state for data splitting.
        learning_rate : float, optional  
            The initial learning rate for the Adam optimizer.
        batch_size : int, optional  
            The batch size for training. Default is 256. Set to 32 if number of cells is small (less than 1000)
        L : int, optional  
            The number of MC samples.
        alpha : float, optional  
            The value of alpha in [0,1] to encourage covariate adjustment. Not used if there is no covariates.
        beta : float, optional  
            The value of beta in beta-VAE.
        num_epoch : int, optional  
            The number of epoch.
        num_step_per_epoch : int, optional 
            The number of step per epoch, it will be inferred from number of cells and batch size if it is None.
        early_stopping_patience : int, optional 
            The maximum number of epochs if there is no improvement.
        early_stopping_tolerance : float, optional 
            The minimum change of loss to be considered as an improvement.
        early_stopping_warmup : int, optional 
            The number of warmup epochs.            
        path_to_weights : str, optional 
            The path of weight file to be saved; not saving weight if None.
        **kwargs :  
            Extra key-value arguments for dimension reduction algorithms.        
        '''        
        if stratify is None:
            stratify = self.init_labels
        elif stratify is False:
            stratify = None    
        id_train, id_test = train_test_split(
                                np.arange(self.X_input.shape[0]), 
                                test_size=test_size, 
                                stratify=stratify, 
                                random_state=random_state)
        if num_step_per_epoch is None:
            num_step_per_epoch = len(id_train)//batch_size+1
        c = None if self.c_score is None else self.c_score.astype(tf.keras.backend.floatx())
        self.train_dataset = train.warp_dataset(self.X_input[id_train].astype(tf.keras.backend.floatx()),
                                                None if c is None else c[id_train],
                                                batch_size, 
                                                self.X_output[id_train].astype(tf.keras.backend.floatx()), 
                                                self.scale_factor[id_train].astype(tf.keras.backend.floatx()))
        self.test_dataset = train.warp_dataset(self.X_input[id_test].astype(tf.keras.backend.floatx()),
                                                None if c is None else c[id_test],
                                                batch_size, 
                                                self.X_output[id_test].astype(tf.keras.backend.floatx()), 
                                                self.scale_factor[id_test].astype(tf.keras.backend.floatx()))    
                                   
        self.vae = train.train(
            self.train_dataset,
            self.test_dataset,
            self.vae,
            learning_rate,
            L,
            alpha,
            beta,
            num_epoch,
            num_step_per_epoch,
            early_stopping_patience,
            early_stopping_tolerance,
            early_stopping_warmup,            
            **kwargs            
            )
        
        self.update_z()
            
        if path_to_weights is not None:
            self.save_model(path_to_weights)
          

    def init_inference(self, batch_size: int = 32, L: int = 5, **kwargs):
        '''Initialize trajectory inference by computing the posterior estimations.        

        Parameters
        ----------
        batch_size : int, optional
            The batch size when doing inference.
        L : int, optional
            The number of MC samples when doing inference.
        **kwargs :  
            Extra key-value arguments for dimension reduction algorithms.              
        '''
        c = None if self.c_score is None else self.c_score.astype(tf.keras.backend.floatx())
        self.test_dataset = train.warp_dataset(self.X_input.astype(tf.keras.backend.floatx()), 
                                               c,
                                               batch_size)
        self.pi, self.mu, self.pc_x,\
            self.cell_position_posterior,self.cell_position_variance,_ = self.vae.inference(self.test_dataset, L=L)
            
        self.adata.obs['vitae_new_clustering'] = np.argmax(self.cell_position_posterior, 1)
        self.adata.obs['vitae_new_clustering'] = self.adata.obs['vitae_new_clustering'].astype('category')
        print("New clustering labels saved as 'vitae_new_clustering' in self.adata.obs.")
        return None
        

    def select_root(self, days, method: str = 'proportion'):
        '''Select the root vertex based on days information.      

        Parameters
        ----------
        day : np.array, optional
            The day information for selected cells used to determine the root vertex.
            The dtype should be 'int' or 'float'.
        method : str, optional
            'sum' or 'mean'. 
            For 'proportion', the root is the one with maximal proportion of cells from the earliest day.
            For 'mean', the root is the one with earliest mean time among cells associated with it.

        Returns
        ----------
        root : int
            The root vertex in the inferred trajectory based on given day information.
        '''
        if days is not None and len(days)!=self.X_input.shape[0]:
            raise ValueError("The length of day information ({}) is not "
                "consistent with the number of selected cells ({})!".format(
                    len(days), self.X_input.shape[0]))
        if not hasattr(self, 'cell_position_posterior'):
            raise ValueError("Need to call 'init_inference' first!")

        estimated_cell_types = np.argmax(self.cell_position_posterior, axis=-1)
        if method=='proportion':
            root = np.argmax([np.mean(days[estimated_cell_types==i]==np.min(days)) for i in range(self.cell_position_posterior.shape[-1])])
        elif method=='mean':
            root = np.argmin([np.mean(days[estimated_cell_types==i]) for i in range(self.cell_position_posterior.shape[-1])])
        else:
            raise ValueError("Method can be either 'proportion' or 'mean'!")
        return root

        
    def comp_inference_score(self, method: str = 'modified_map', thres = 0.5, 
            no_loop: bool = False, cutoff: Optional[float] = None,
            plot_backbone: bool = True):
        ''' Compute edge scores.

        Parameters
        ----------
        method : string, optional
            'mean', 'modified_mean', 'map', or 'modified_map'.
        thres : float, optional 
            The threshold used for filtering edges \(e_{ij}\) that \((n_{i}+n_{j}+e_{ij})/N<thres\), only applied to mean method.
        no_loop : boolean, optional 
            Whether loops are allowed to exist in the graph. If no_loop is true, will prune the graph to contain only the 
            maximum spanning true
        cutoff : string, optional
            The score threshold for filtering edges with scores less than cutoff.
        plot_backbone: boolean
            whether plot the current trajectory backbone (undirected graph)
        
        Returns
        ----------
        G : nx.Graph 
            The weighted graph with weight on each edge indicating its score of existence.
        '''
        print("Estimate the backbone of the trajectory ...")
        G, _ = self.inferer.init_inference(self.cell_position_posterior, 
                                                self.pc_x, 
                                                thres, method, no_loop)
        if cutoff is not None:
            graph = nx.to_numpy_matrix(G)
            graph[graph<=cutoff] = 0
            G = nx.from_numpy_array(graph)
        if plot_backbone:
            edgewidth = [ d['weight'] for (u,v,d) in G.edges(data=True)]
            nx.draw_spring(G, width = edgewidth/np.mean(edgewidth), with_labels = True)
        return G
        
        
    def infer_trajectory(self, init_node: int, cutoff: Optional[float] = None,
                         plot_backbone: bool = True, method: str = 'UMAP', **kwargs):
        '''Infer the trajectory.

        Parameters
        ----------
        init_node : int
            The initial node for the inferred trajectory.
        cutoff : string, optional
            The threshold for filtering edges with scores less than cutoff.
        is_plot : boolean, optional
            Whether to plot or not.
        path : string, optional  
            The path to save figure, or don't save if it is None.
        plot_backbone: boolean
            whether plot the current trajectory backbone (directed graph)

        Returns
        ----------
        '''
        self.backbone, self.cell_position_projected, self.pseudotime = self.inferer.infer_trajectory(init_node, cutoff)
        self.uncertainty = np.sum((self.cell_position_projected - self.cell_position_posterior)**2, axis=-1) + np.sum(self.cell_position_variance, axis=-1)
        self.adata.obs['pseudotime'] = self.pseudotime
        self.adata.obs['projection_uncertainty'] = self.uncertainty
        print("Cell psedutime and projection uncertainties stored as 'pseudotime' and 'projection_uncertainty' in self.adata.obs")
                
        self.adata.obs['vitae_new_clustering'] = self.labels_map.iloc[np.argmax(self.cell_position_projected, 1)]['label_names'].to_numpy()
        self.adata.obs['vitae_new_clustering'] = self.adata.obs['vitae_new_clustering'].astype('category')
        print("'vitae_new_clustering' updated based on the projected cell positions.")
        
        connected_comps = nx.node_connected_component(self.backbone, init_node)
        DG = nx.DiGraph(nx.to_directed(self.backbone))
        subG = DG.subgraph(connected_comps)
        DG.remove_edges_from(subG.edges - nx.dfs_edges(DG, init_node))
        self.backbone = DG
        print("Directed trajectory backbone saved as self.backbone.")
        
        if plot_backbone:
            edgewidth = [ d['weight'] for (u,v,d) in DG.edges(data=True)]
            nx.draw_spring(DG, width = edgewidth/np.mean(edgewidth), with_labels = True)
        

        ax = self.visualize_latent(method = method, color='pseudotime', show=False, **kwargs)
        cluster_labels = self.adata.obs['vitae_new_clustering'].to_numpy()
        uni_cluster_labels = list(self.adata.obs['vitae_new_clustering'].cat.categories)
        embed_z = self._adata.obsm[self.dict_method_scname[method]]
        embed_mu = np.zeros((len(uni_cluster_labels), 2))
        for i,l in enumerate(uni_cluster_labels):
            embed_mu[i,:] = np.mean(embed_z[cluster_labels==l], axis=0)
            embed_mu[i,:] = embed_z[cluster_labels==l][np.argmin(np.mean((embed_z[cluster_labels==l] - embed_mu[i,:])**2, axis=1)),:]
        ax = self.inferer.plot_trajectory(ax, embed_z, embed_mu, uni_cluster_labels)
        return ax



    def differential_expression_test(self, alpha: float = 0.05, cell_subset = None):
        '''Differentially gene expression test. All (selected and unselected) genes will be tested 
        Only cells in `selected_cell_subset` will be used, which is useful when one need to
        test differentially expressed genes on a branch of the inferred trajectory.

        Parameters
        ----------
        alpha : float, optional
            The cutoff of p-values.

        Returns
        ----------
        res_df : pandas.DataFrame
            The test results of expressed genes with two columns,
            the estimated coefficients and the adjusted p-values.
        '''
        if not hasattr(self, 'pseudotime'):
            raise ReferenceError("Pseudotime does not exist! Please run 'infer_trajectory' first.")
        if cell_subset is None:
            cell_subset = np.arange(self.X_input.shape[0])
            print("All cells are selected.")

        # Prepare X and Y for regression expression ~ rank(PDT) + covariates
        Y = self.adata.X[cell_subset,:]
#        std_Y = np.std(Y, ddof=1, axis=0, keepdims=True)
#        Y = np.divide(Y-np.mean(Y, axis=0, keepdims=True), std_Y, out=np.empty_like(Y)*np.nan, where=std_Y!=0)
        X = stats.rankdata(self.pseudotime[cell_subset])
        X = ((X-np.mean(X))/np.std(X, ddof=1)).reshape((-1,1))
        if self.c_score is None:
            X = np.c_[np.ones_like(X), X]
        else:
            X = np.c_[np.ones_like(X), X, self.c_score[cell_subset,:]]

        res_df = DE_test(Y, X, self.adata.var_names, alpha)
        return res_df


 

    def evaluate(self, milestone_net, begin_node_true, grouping = None,
                thres: float = 0.5, no_loop: bool = True, cutoff: Optional[float] = None,
                method: str = 'mean', path: Optional[str] = None):
        ''' Evaluate the model.

        Parameters
        ----------
        milestone_net : pd.DataFrame
            The true milestone network. For real data, milestone_net will be a DataFrame of the graph of nodes.
            Eg.

            from|to
            ---|---
            cluster 1 | cluster 1
            cluster 1 | cluster 2

            For synthetic data, milestone_net will be a DataFrame of the (projected)
            positions of cells. The indexes are the orders of cells in the dataset.
            Eg.

            from|to|w
            ---|---|---
            cluster 1 | cluster 1 | 1
            cluster 1 | cluster 2 | 0.1
        begin_node_true : str or int
            The true begin node of the milestone.
        grouping : np.array, optional
            \([N,]\) The labels. For real data, grouping must be provided.

        Returns
        ----------
        res : pd.DataFrame
            The evaluation result.
        '''
        if not hasattr(self, 'le'):
            raise ValueError("No given labels for training.")

        # Evaluate for the whole dataset will ignore selected_cell_subset.
        if len(self.selected_cell_subset)!=len(self.cell_names):
            warnings.warn("Evaluate for the whole dataset.")
        
        # If the begin_node_true, need to encode it by self.le.
        if isinstance(begin_node_true, str):
            begin_node_true = self.le.transform([begin_node_true])[0]
            
        # For generated data, grouping information is already in milestone_net
        if 'w' in milestone_net.columns:
            grouping = None
            
        # If milestone_net is provided, transform them to be numeric.
        if milestone_net is not None:
            milestone_net['from'] = self.le.transform(milestone_net['from'])
            milestone_net['to'] = self.le.transform(milestone_net['to'])
            
        begin_node_pred = int(np.argmin(np.mean((
            self.z[self.labels==begin_node_true,:,np.newaxis] -
            self.mu[np.newaxis,:,:])**2, axis=(0,1))))
        
        G, edges = self.inferer.init_inference(self.cell_position_posterior, self.pc_x, thres, method, no_loop)
        G, w, pseudotime = self.inferer.infer_trajectory(begin_node_pred, self.label_names, cutoff=cutoff, path=path, is_plot=False)
        
        # 1. Topology
        G_pred = nx.Graph()
        G_pred.add_nodes_from(G.nodes)
        G_pred.add_edges_from(G.edges)
        nx.set_node_attributes(G_pred, False, 'is_init')
        G_pred.nodes[begin_node_pred]['is_init'] = True

        G_true = nx.Graph()
        G_true.add_nodes_from(G.nodes)
        # if 'grouping' is not provided, assume 'milestone_net' contains proportions
        if grouping is None:
            G_true.add_edges_from(list(
                milestone_net[~pd.isna(milestone_net['w'])].groupby(['from', 'to']).count().index))
        # otherwise, 'milestone_net' indicates edges
        else:
            if milestone_net is not None:             
                G_true.add_edges_from(list(
                    milestone_net.groupby(['from', 'to']).count().index))
            grouping = self.le.transform(grouping)
        G_true.remove_edges_from(nx.selfloop_edges(G_true))
        nx.set_node_attributes(G_true, False, 'is_init')
        G_true.nodes[begin_node_true]['is_init'] = True
        res = topology(G_true, G_pred)
            
        # 2. Milestones assignment
        if grouping is None:
            milestones_true = milestone_net['from'].values.copy()
            milestones_true[(milestone_net['from']!=milestone_net['to'])
                           &(milestone_net['w']<0.5)] = milestone_net[(milestone_net['from']!=milestone_net['to'])
                                                                      &(milestone_net['w']<0.5)]['to'].values
        else:
            milestones_true = grouping
        milestones_true = milestones_true[pseudotime!=-1]
        milestones_pred = np.argmax(w[pseudotime!=-1,:], axis=1)
        res['ARI'] = (adjusted_rand_score(milestones_true, milestones_pred) + 1)/2
        
        if grouping is None:
            n_samples = len(milestone_net)
            prop = np.zeros((n_samples,n_samples))
            prop[np.arange(n_samples), milestone_net['to']] = 1-milestone_net['w']
            prop[np.arange(n_samples), milestone_net['from']] = np.where(np.isnan(milestone_net['w']), 1, milestone_net['w'])
            res['GRI'] = get_GRI(prop, w)
        else:
            res['GRI'] = get_GRI(grouping, w)
        
        # 3. Correlation between geodesic distances / Pseudotime
        if no_loop:
            if grouping is None:
                pseudotime_true = milestone_net['from'].values + 1 - milestone_net['w'].values
                pseudotime_true[np.isnan(pseudotime_true)] = milestone_net[pd.isna(milestone_net['w'])]['from'].values            
            else:
                pseudotime_true = - np.ones(len(grouping))
                nx.set_edge_attributes(G_true, values = 1, name = 'weight')
                connected_comps = nx.node_connected_component(G_true, begin_node_true)
                subG = G_true.subgraph(connected_comps)
                milestone_net_true = self.inferer.build_milestone_net(subG, begin_node_true)
                if len(milestone_net_true)>0:
                    pseudotime_true[grouping==int(milestone_net_true[0,0])] = 0
                    for i in range(len(milestone_net_true)):
                        pseudotime_true[grouping==int(milestone_net_true[i,1])] = milestone_net_true[i,-1]
            pseudotime_true = pseudotime_true[pseudotime>-1]
            pseudotime_pred = pseudotime[pseudotime>-1]
            res['PDT score'] = (np.corrcoef(pseudotime_true,pseudotime_pred)[0,1]+1)/2
        else:
            res['PDT score'] = np.nan
            
        # 4. Shape
        # score_cos_theta = 0
        # for (_from,_to) in G.edges:
        #     _z = self.z[(w[:,_from]>0) & (w[:,_to]>0),:]
        #     v_1 = _z - self.mu[:,_from]
        #     v_2 = _z - self.mu[:,_to]
        #     cos_theta = np.sum(v_1*v_2, -1)/(np.linalg.norm(v_1,axis=-1)*np.linalg.norm(v_2,axis=-1)+1e-12)

        #     score_cos_theta += np.sum((1-cos_theta)/2)

        # res['score_cos_theta'] = score_cos_theta/(np.sum(np.sum(w>0, axis=-1)==2)+1e-12)
        return res