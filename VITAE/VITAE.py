import warnings
import os
from typing import Optional

import VITAE.model as model
import VITAE.preprocess as preprocess
import VITAE.train as train
from VITAE.inference import Inferer
from VITAE.utils import load_data, get_embedding, get_igraph, louvain_igraph, \
    plot_clusters, plot_marker_gene, DE_test
from VITAE.metric import topology, get_GRI

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.model_selection import train_test_split
import umap
import numpy as np
import scipy as sp
import pandas as pd
import networkx as nx
from scipy import stats


class VITAE():
    """
    Variational Inference for Trajectory by AutoEncoder.
    """
    def __init__(self):
        pass

    def get_data(self, X = None, adata = None, labels = None,
                 covariate = None, cell_names = None, gene_names = None):
        '''Get data for model. 
        
        (1) The user can provide a 2-dim numpy array as the count matrix `X`, either preprocessed or raw. 
        Some extra information `labels`, `cell_names` and `gene_names` (as 1-dim numpy arrays) are optional.

        (2) If the package `scanpy` is installed, then the function can also accept an `annData` input as `adata`. 
        Some extra information `labels`, `cell_names` and `gene_names` are extracted from 
        `adata.obs.cell_types`, `adata.obs_names.values` and `adata.var_names.values`, and
        a 1-dim numpy array `labels` can also be provided if `adata.obs.cell_types` does not exist.

        Covariates can be provided as a 2-dim numpy array.

        Parameters
        ----------
        X : np.array, optional
            \([N, G]\) The counts or expressions data.
        adata : AnnData, optional
            The scanpy object.      
        covariate : np.array, optional
            \([N, s]\) The covariate data.
        labels : np.array, optional
            \([N,]\) The list of labelss for cells.
        cell_names : np.array, optional
            \([N,]\) The list of cell names.
        gene_names : np.array, optional
            \([N,]\) The list of gene names.
        '''
        if adata is None and X is None:
            raise ValueError("Either X or adata should be given!")
        if adata is not None and X is not None:
            warnings.warn("Both X and adata are given, will use adata!")

        self.adata = adata        
        self.raw_X = None if X is None else X.astype(np.float32)
        self.c_score = None if covariate is None else np.array(covariate, np.float32)
        if sp.sparse.issparse(self.raw_X):
            self.raw_X = self.raw_X.toarray()
        self.raw_label_names = None if labels is None else np.array(labels, dtype = str)
        if X is None:
            self.raw_cell_names = None
            self.raw_gene_names = None
        else:            
            self.raw_cell_names = np.array(['c_%d'%i for i in range(self.raw_X.shape[0])]) if \
                cell_names is None else np.array(cell_names, dtype = str)
            self.raw_gene_names = np.array(['g_%d'%i for i in range(self.raw_X.shape[1])]) if \
                gene_names is None else np.array(gene_names, dtype = str)

        
    def preprocess_data(self, processed: bool = False, dimred: bool = False,
                        K: float = 1e4, gene_num: int = 2000, data_type: str = 'UMI', npc: int = 64):
        ''' Data preprocessing - log-normalization, feature selection, and scaling.                    

        If the inputs are preprocessed by users, then `Gaussian` model will be used and PCA will be performed to reduce the input dimension.
        Otherwise, preprocessing will be performed on `X` following Seurat's routine. 
        If `adata` is provided, the preprocession will be done via `scanpy`.

        Parameters
        ----------
        processed : boolean, optional
            Whether adata has been processed. If `processed=True`, then `Gaussian` model will be used.
        dimred : boolean, optional
            Whether the processed adata is after dimension reduction.
        K : float, optional              
            The constant summing gene expression in each cell up to.
        gene_num : int, optional
            The number of feature to select.
        data_type : str, optional
            'UMI', 'non-UMI' and 'Gaussian', default is 'UMI'. If the input is a processed scanpy object, data type is set to Gaussian.
        npc : int, optional
            The number of PCs to retain.
        '''
        if data_type not in set(['UMI', 'non-UMI', 'Gaussian']):
            raise ValueError("Invalid data type, must be one of 'UMI', 'non-UMI', and 'Gaussian'.")

        if (self.adata is not None) & processed:
            self.data_type = 'Gaussian'
        else:
            self.data_type = data_type

        raw_X = self.raw_X.copy() if self.raw_X is not None else None
        self.X_normalized, self.expression, self.X, self.c_score, \
        self.cell_names, self.gene_names, self.selected_gene_names, \
        self.scale_factor, self.labels, self.label_names, \
        self.le, self.gene_scalar = preprocess.preprocess(
            self.adata,
            processed,
            dimred,
            raw_X,
            self.c_score,
            self.raw_label_names,
            self.raw_cell_names,
            self.raw_gene_names,
            K, gene_num, data_type, npc)
        self.dim_origin = self.X.shape[1]
        self.selected_cell_subset = self.cell_names
        self.selected_cell_subset_id = np.arange(len(self.cell_names))
        self.adata = None


    def build_model(self,
        dimensions = [16],
        dim_latent: int = 8,   
        ):
        ''' Initialize the Variational Auto Encoder model.
        
        Parameters
        ----------
        dimensions : list, optional
            The list of dimensions of layers of autoencoder between latent space and original space.
        dim_latent : int, optional
            The dimension of latent space.
        '''
        self.dimensions = dimensions
        self.dim_latent = dim_latent
    
        self.vae = model.VariationalAutoEncoder(
            self.dim_origin, self.dimensions,
            self.dim_latent, self.data_type,
            False if self.c_score is None else True
            )
        
        if hasattr(self, 'inferer'):
            delattr(self, 'inferer')
            

    def save_model(self, path_to_file: str = 'model.checkpoint'):
        '''Saving model weights.
        
        Parameters
        ----------
        path_to_file : str, optional
            The path to weight files of pre-trained or trained model           
        '''
        self.vae.save_weights(path_to_file)
        if hasattr(self, 'cluster_labels') and self.cluster_labels is not None:
            with open(path_to_file+'.label', 'wb') as f:
                np.save(f, self.cluster_labels)
        with open(path_to_file+'.config', 'wb') as f:
            np.save(f, [self.dim_origin,
                        self.dimensions,
                        self.dim_latent,
                        self.data_type,
                        False if self.c_score is None else True])
        if hasattr(self, 'pi'):
            with open(path_to_file+'.inference', 'wb') as f:
                np.save(f, [self.pi, self.mu, self.pc_x,
                            self.w_tilde, self.var_w_tilde,
                            self.D_JS, self.z, self.embed_z, self.inferer.embed_mu])
    

    def load_model(self, path_to_file: str = 'model.checkpoint', load_labels: bool = False):
        '''Load model weights.

        Parameters
        ----------
        path_to_file : str, optional 
            The path to weight files of pre trained or trained model
        load_labels : boolean, optional
            Whether to load clustering labels or not.
            If load_labels is True, then the LatentSpace layer will be initialized basd on the model. 
            If load_labels is False, then the LatentSpace layer will not be initialized.
        ''' 
        if not os.path.exists(path_to_file+'.config'):
            raise AssertionError('Config file not exist!')               
        if load_labels and not os.path.exists(path_to_file+'.label'):
            raise AssertionError('Label file not exist!')

        with open(path_to_file+'.config', 'rb') as f:
            [self.dim_origin, self.dimensions,
            self.dim_latent, self.data_type, has_c] = np.load(f, allow_pickle=True)
        self.vae = model.VariationalAutoEncoder(
            self.dim_origin, self.dimensions, 
            self.dim_latent, self.data_type, has_c
            )

        if load_labels:            
            with open(path_to_file+'.label', 'rb') as f:
                cluster_labels = np.load(f, allow_pickle=True)
            n_clusters = len(np.unique(cluster_labels))
            self.init_latent_space(n_clusters, cluster_labels)
            if os.path.exists(path_to_file+'.inference'):
                with open(path_to_file+'.inference', 'rb') as f:
                    [self.pi, self.mu, self.pc_x, self.w_tilde, self.var_w_tilde,
                        self.D_JS, self.z, self.embed_z, embed_mu] = np.load(f, allow_pickle=True)
                self.inferer.mu = self.mu
                self.inferer.embed_z = self.embed_z
                self.inferer.embed_mu = embed_mu

        self.vae.load_weights(path_to_file)


    def pre_train(self, stratify = False, test_size = 0.1, random_state: int = 0,
            learning_rate: float = 1e-3, batch_size: int = 32, L: int = 1, alpha: float = 0.01,
            num_epoch: int = 300, num_step_per_epoch: Optional[int] = None,
            early_stopping_patience: int = 10, early_stopping_tolerance: float = 1e-3, early_stopping_warmup: int = 0, 
            path_to_weights: Optional[str] = None):
        '''Pretrain the model with specified learning rate.

        Parameters
        ----------
        stratify : np.array, None, or False
            If an array is provided, or `stratify=None` and `self.labels` is available, then they will be used to perform stratified shuffle splitting. Otherwise, general shuffle splitting is used. Set to `False` if `self.labels` is not intented for stratified shuffle splitting.
        test_size : float or int, optional
            The proportion or size of the test set.
        random_state : int, optional
            The random state for data splitting.
        learning_rate : float, optional
            The initial learning rate for the Adam optimizer.
        batch_size : int, optional 
            The batch size for pre-training.
        L : int, optional 
            The number of MC samples.
        alpha : float, optional
            The value of alpha in [0,1] to encourage covariate adjustment. Not used if there is no covariates.
        num_epoch : int, optional 
            The maximum number of epoches.
        num_step_per_epoch : int, optional 
            The number of step per epoch, it will be inferred from number of cells and batch size if it is None.            
        early_stopping_patience : int, optional 
            The maximum number of epoches if there is no improvement.
        early_stopping_tolerance : float, optional 
            The minimum change of loss to be considered as an improvement.
        early_stopping_warmup : int, optional
            The number of warmup epoches.
        path_to_weights : str, optional 
            The path of weight file to be saved; not saving weight if None.
        '''                    
        train.clear_session()
        if stratify is None:
            stratify = self.labels
        elif stratify is False:
            stratify = None   
        id_train, id_test = train_test_split(
                                np.arange(self.X.shape[0]), 
                                test_size=test_size, 
                                stratify=stratify, 
                                random_state=random_state)
        if num_step_per_epoch is None:
            num_step_per_epoch = len(id_train)//batch_size+1
        self.train_dataset = train.warp_dataset(self.X_normalized[id_train], 
                                                None if self.c_score is None else self.c_score[id_train],
                                                batch_size, 
                                                self.X[id_train], 
                                                self.scale_factor[id_train])
        self.test_dataset = train.warp_dataset(self.X_normalized[id_test], 
                                                None if self.c_score is None else self.c_score[id_test],
                                                batch_size, 
                                                self.X[id_test], 
                                                self.scale_factor[id_test])
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
            early_stopping_warmup)

        if path_to_weights is not None:
            self.save_model(path_to_weights)
          

    def get_latent_z(self):
        '''Set a subset of interested cells.

        Returns
        ----------
        z : np.array
            \([N,d]\) The latent means.
        ''' 
        c = None if self.c_score is None else self.c_score[self.selected_cell_subset_id,:]
        return self.vae.get_z(self.X_normalized[self.selected_cell_subset_id,:], c)


    def set_cell_subset(self, selected_cell_names):
        '''Set a subset of interested cells.

        Parameters
        ----------
        selected_cell_names : np.array, optional
            The names of selected cells.
        ''' 
        self.selected_cell_subset = np.unique(selected_cell_names)
        self.selected_cell_subset_id = np.sort(np.where(np.in1d(self.cell_names, selected_cell_names))[0])
        
    
    def refine_pi(self, batch_size: int = 64):  
        '''Refine pi by the its posterior. This function will be effected if 'selected_cell_subset_id' is set.

        Parameters
        ----------
        batch_size - int, optional 
            The batch size when computing \(p(c_i|Y_i,X_i)\).
        
        Returns
        ----------
        pi : np.array
            \([1,K]\) The original pi.
        post_pi : np.array 
            \([1,K]\) The posterior estimate of pi.
        '''      
        if len(self.selected_cell_subset_id)!=len(self.cell_names):
            warnings.warn("Only using a subset of cells to refine pi.")

        c = None if self.c_score is None else self.c_score[self.selected_cell_subset_id,:]
        self.test_dataset = train.warp_dataset(
            self.X_normalized[self.selected_cell_subset_id,:], 
            c,
            batch_size)
        pi, p_c_x = self.vae.get_pc_x(self.test_dataset)

        post_pi = np.mean(p_c_x, axis=0, keepdims=True)        
        self.vae.latent_space.pi.assign(np.log(post_pi+1e-16))
        return pi, post_pi


    def init_latent_space(self, n_clusters: int, cluster_labels = None, mu = None, log_pi = None):
        '''Initialze the latent space.

        Parameters
        ----------
        n_clusters : int
            The number of cluster.
        cluster_labels : np.array, optional
            \([N,]\) The  cluster labels.
        mu : np.array, optional
            \([d,k]\) The value of initial \(\\mu\).
        log_pi : np.array, optional
            \([1,K]\) The value of initial \(\\log(\\pi)\).
        '''             
        z = self.get_latent_z()
        if (mu is None) & (cluster_labels is not None):
            mu = np.zeros((z.shape[1], n_clusters))
            for i,l in enumerate(np.unique(cluster_labels)):
                mu[:,i] = np.mean(z[cluster_labels==l], axis=0)

        self.n_clusters = n_clusters
        self.cluster_labels = None if cluster_labels is None else np.array(cluster_labels)
        self.vae.init_latent_space(n_clusters, mu, log_pi)
        self.inferer = Inferer(self.n_clusters)            


    def train(self, stratify = False, test_size = 0.1, random_state: int = 0,
            learning_rate: float = 1e-3, batch_size: int = 32, 
            L: int = 1, alpha: float = 0.01, beta: float = 1, 
            num_epoch: int = 300, num_step_per_epoch: Optional[int] =  None,
            early_stopping_patience: int = 10, early_stopping_tolerance: float = 1e-3, early_stopping_warmup: int = 5,
            path_to_weights: Optional[str] = None, plot_every_num_epoch: Optional[int] = None, dimred: str = 'umap', **kwargs):
        '''Train the model.

        Parameters
        ----------
        stratify : np.array, None, or False
            If an array is provided, or `stratify=None` and `self.labels` is available, then they will be used to perform stratified shuffle splitting. Otherwise, general shuffle splitting is used. Set to `False` if `self.labels` is not intented for stratified shuffle splitting.
        test_size : float or int, optional
            The proportion or size of the test set.
        random_state : int, optional
            The random state for data splitting.
        learning_rate : float, optional  
            The initial learning rate for the Adam optimizer.
        batch_size : int, optional  
            The batch size for training.
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
            The maximum number of epoches if there is no improvement.
        early_stopping_tolerance : float, optional 
            The minimum change of loss to be considered as an improvement.
        early_stopping_warmup : int, optional 
            The number of warmup epoches.            
        path_to_weights : str, optional 
            The path of weight file to be saved; not saving weight if None.
        plot_every_num_epoch : int, optional 
            Plot the intermediate result every few epoches, or not plotting if it is None.            
        dimred : str, optional 
            The name of dimension reduction algorithms, can be 'umap', 'pca' and 'tsne'. Only used if 'plot_every_num_epoch' is not None. 
        **kwargs :  
            Extra key-value arguments for dimension reduction algorithms.        
        '''        
        if stratify is None:
            stratify = self.labels[self.selected_cell_subset_id]
        elif stratify is False:
            stratify = None    
        id_train, id_test = train_test_split(
                                np.arange(len(self.selected_cell_subset_id)), 
                                test_size=test_size, 
                                stratify=stratify, 
                                random_state=random_state)
        if num_step_per_epoch is None:
            num_step_per_epoch = len(id_train)//batch_size+1
        c = None if self.c_score is None else self.c_score[self.selected_cell_subset_id,:]
        self.train_dataset = train.warp_dataset(self.X_normalized[self.selected_cell_subset_id,:][id_train],
                                                None if c is None else c[id_train],
                                                batch_size, 
                                                self.X[self.selected_cell_subset_id,:][id_train], 
                                                self.scale_factor[self.selected_cell_subset_id][id_train])
        self.test_dataset = train.warp_dataset(self.X_normalized[self.selected_cell_subset_id,:][id_test],
                                                None if c is None else c[id_test],
                                                batch_size, 
                                                self.X[self.selected_cell_subset_id,:][id_test], 
                                                self.scale_factor[self.selected_cell_subset_id][id_test])    
        if plot_every_num_epoch is None:
            self.whole_dataset = None    
        else:
            self.whole_dataset = train.warp_dataset(self.X_normalized[self.selected_cell_subset_id,:], 
                                                    c,
                                                    batch_size)                                    
        self.vae = train.train(
            self.train_dataset,
            self.test_dataset,
            self.whole_dataset,
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
            self.labels[self.selected_cell_subset_id],            
            plot_every_num_epoch,
            dimred='umap', 
            **kwargs            
            )
            
        if path_to_weights is not None:
            self.save_model(path_to_weights)
          

    def init_inference(self, batch_size: int = 32, L: int = 5, 
            dimred: str = 'umap', refit_dimred: bool = True, **kwargs):
        '''Initialze trajectory inference by computing the posterior estimations.        

        Parameters
        ----------
        batch_size : int, optional
            The batch size when doing inference.
        L : int, optional
            The number of MC samples when doing inference.
        dimred : str, optional
            The name of dimension reduction algorithms, can be 'umap', 'pca' and 'tsne'.
        refit_dimred : boolean, optional 
            Whether to refit the dimension reduction algorithm or not.
        **kwargs :  
            Extra key-value arguments for dimension reduction algorithms.              
        '''
        c = None if self.c_score is None else self.c_score[self.selected_cell_subset_id,:]
        self.test_dataset = train.warp_dataset(self.X_normalized[self.selected_cell_subset_id,:], 
                                               c,
                                               batch_size)
        self.pi, self.mu, self.pc_x,\
            self.w_tilde,self.var_w_tilde,self.D_JS,self.z = self.vae.inference(self.test_dataset, L=L)
        if refit_dimred or not hasattr(self.inferer, 'embed_z'):
            self.embed_z = self.inferer.init_embedding(self.z, self.mu, **kwargs)
        else:
            self.embed_z = self.inferer.embed_z
        return None
        

    def select_root(self, days, method: str = 'sum'):
        '''Initialze trajectory inference by computing the posterior estimations.        

        Parameters
        ----------
        day : np.array, optional
            The day information for selected cells used to determine the root vertex.
            The dtype should be 'int' or 'float'.
        method : str, optional
            'sum' or 'mean'. 
            For 'sum', the root is the one with maximal number of cells from the earliest day.
            For 'mean', the root is the one with earliest mean time among cells associated with it.

        Returns
        ----------
        root : int
            The root vertex in the inferred trajectory based on given day information.
        '''
        if days is not None and len(days)!=len(self.selected_cell_subset_id):
            raise ValueError("The length of day information ({}) is not "
                "consistent with the number of selected cells ({})!".format(
                    len(days), len(self.selected_cell_subset_id)))
        if not hasattr(self.inferer, 'embed_z'):
            raise ValueError("Need to call 'init_inference' first!")

        estimated_cell_types = np.argmax(self.w_tilde, axis=-1)
        if method=='sum':
            root = np.argmax([np.sum(days[estimated_cell_types==i]==np.min(days)) for i in range(self.w_tilde.shape[-1])])
        elif method=='mean':
            root = np.argmin([np.mean(days[estimated_cell_types==i]) for i in range(self.w_tilde.shape[-1])])
        else:
            raise ValueError("Method can be either 'sum' or 'mean'!")
        return root

        
    def comp_inference_score(self, method: str = 'modified_map', thres = 0.5, 
            no_loop: bool = False, is_plot: bool = True, plot_labels: bool = True, path: Optional[str] = None):
        ''' Compute edge scores.

        Parameters
        ----------
        method : string, optional
            'mean', 'modified_mean', 'map', or 'modified_map'.
        thres : float, optional 
            The threshold used for filtering edges \(e_{ij}\) that \((n_{i}+n_{j}+e_{ij})/N<thres\), only applied to mean method.
        no_loop : boolean, optional 
            Whether loops are allowed to exist in the graph.
        is_plot : boolean, optional  
            Whether to plot or not.
        plot_labels : boolean, optional  
            Whether to plot label names or not, only used when `is_plot=True`.
        path : string, optional
            The path to save figure, or don't save if it is None.
        
        Returns
        ----------
        G : nx.Graph 
            The weighted graph with weight on each edge indicating its score of existence.
        '''
        G, edges = self.inferer.init_inference(self.w_tilde, self.pc_x, thres, method, no_loop)
        if is_plot:
            self.inferer.plot_clusters(self.cluster_labels, plot_labels=plot_labels, path=path)
        return G
        
        
    def infer_trajectory(self, init_node: int, cutoff: Optional[float] = None, is_plot: bool = True, path: Optional[str] = None):
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

        Returns
        ----------
        G : nx.Graph 
            The modified graph that indicates the inferred trajectory.
        w : np.array
            \([N,k]\) The modified \(\\tilde{w}\).
        pseudotime : np.array
            \([N,]\) The pseudotime based on projected trajectory.
        '''
        G, w, pseudotime = self.inferer.infer_trajectory(init_node, 
                                                         self.label_names[self.selected_cell_subset_id], 
                                                         cutoff, 
                                                         path=path, 
                                                         is_plot=is_plot)
        self.pseudotime = pseudotime
        return G, w, pseudotime


    def differentially_expressed_test(self, alpha: float = 0.05):
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

        # Prepare X and Y for regression expression ~ rank(PDT) + covariates
        Y = self.expression[self.selected_cell_subset_id,:]
        std_Y = np.std(Y, ddof=1, axis=0, keepdims=True)
        Y = np.divide(Y-np.mean(Y, axis=0, keepdims=True), std_Y, out=np.empty_like(Y)*np.nan, where=std_Y!=0)
        X = stats.rankdata(self.pseudotime[self.selected_cell_subset_id])
        X = ((X-np.mean(X))/np.std(X, ddof=1)).reshape((-1,1))
        X = np.c_[np.ones_like(X), X, self.c_score[self.selected_cell_subset_id,:]]

        res_df = DE_test(Y, X, self.gene_names, alpha)
        return res_df


    def plot_marker_gene(self, gene_name: str, refit_dimred: bool = False, dimred: str = 'umap', path: Optional[str] =None, **kwargs):
        '''Plot expression of the given marker gene.

        Parameters
        ----------
        gene_name : str 
            The name of the marker gene.
        refit_dimred : boolean, optional 
            Whether to refit dimension reduction or use the existing embedding after inference.
        dimred : str, optional
            The name of dimension reduction algorithms, can be 'umap', 'pca' and 'tsne'.
        path : str, optional
            The path to save the figure, or not saving if it is None.
        **kwargs :  
            Extra key-value arguments for dimension reduction algorithms.
        '''
        if gene_name not in self.gene_names:
            raise ValueError("Gene '{}' does not exist!".format(gene_name))
        if self.expression is None:
            raise ReferenceError("The expression matrix does not exist!")
        expression = self.expression[self.selected_cell_subset_id,:][:,self.gene_names==gene_name].flatten()
        
        if not hasattr(self, 'embed_z') or refit_dimred:
            z = self.get_latent_z()       
            embed_z = get_embedding(z, dimred, **kwargs)
        else:
            embed_z = self.embed_z
        plot_marker_gene(expression, 
                         gene_name, 
                         embed_z[self.selected_cell_subset_id,:],
                         path)
        return None


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
        
        G, edges = self.inferer.init_inference(self.w_tilde, self.pc_x, thres, method, no_loop)
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