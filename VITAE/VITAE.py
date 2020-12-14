import warnings
import os

import VITAE.model as model
import VITAE.preprocess as preprocess
import VITAE.train as train
from VITAE.inference import Inferer
from VITAE.utils import load_data, get_embedding, get_igraph, louvain_igraph, plot_clusters, plot_marker_gene
from VITAE.metric import topology, get_GRI

from sklearn.metrics.cluster import adjusted_rand_score
from scipy.spatial.distance import pdist as dist
import umap
import numpy as np
import scipy as sp
import pandas as pd
import networkx as nx


class VITAE():
    """
    Variational Inference for Trajectory by AutoEncoder.
    """
    def __init__(self):
        pass

    def get_data(self, X = None, adata = None, labels = None,
                 covariate=None, cell_names = None, gene_names = None):
        ''' get data for model
        Params:
            adata       - a scanpy object
            X:          - 2-dimension np array, counts or expressions data
            covariate   - 2-dimension np array, covariate data
            labels:     - (optional) a list of labelss for cells
            cell_names  - (optional) a list of cell names
            gene_names  - (optional) a list of gene names
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

        
    def preprocess_data(self, processed = False, dimred = False,
                        K = 1e4, gene_num = 2000, data_type = 'UMI', npc = 64):
        ''' data preprocessing, feature selection, log-normalization
            If input with processed scanpy object, data type is set to Gaussian
        Params:
            processed       - whether adata has been processed
            dimred          - whether the processed adata is after dimension reduction
            K               - the constant summing gene expression in each cell up to
            gene_num        - number of feature to select
            data_type       - 'UMI', 'non-UMI' and 'Gaussian', default is 'UMI'
            npc             - Number of PCs
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
            data_type, K, gene_num, npc)
        self.dim_origin = self.X.shape[1]
        self.selected_cell_subset = self.cell_names
        self.selected_cell_subset_id = np.arange(len(self.cell_names))
        self.adata = None


    def build_model(self,
        dimensions = [16],
        dim_latent = 8,   
        ):
        ''' Initialize the Variational Auto Encoder model.
        Params:
            dimensions          - a list of dimensions of layers of autoencoder between latent 
                                  space and original space
            dim_latent          - dimension of latent space
        '''
        self.dimensions = dimensions
        self.dim_latent = dim_latent
    
        self.vae = model.VariationalAutoEncoder(
            self.dim_origin, self.dimensions,
            self.dim_latent, self.data_type,
            False if self.c_score is None else True
            )
        

    def save_model(self, path_to_file='model.checkpoint'):
        '''Saving model weights.
        Params:
            path_to_file - path to weight files of pre-trained or
                           trained model           
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
    

    def load_model(self, path_to_file='model.checkpoint', load_labels=False):
        '''Loading model weights, called after the model is built.
        Params:
            path_to_file - path to weight files of pre trained or
                           trained model
            load_labels  - whether to load clustering labels or not.
                           If load_labels is True, then the LatentSpace layer
                           will be initialized basd on the model. 
                           If load_labels is False, then the LatentSpace layer
                           will not be initialized.
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


    def pre_train(self, learning_rate = 1e-3, batch_size = 32, L = 1, alpha=0.01,
            num_epoch = 300, num_step_per_epoch = None,
            early_stopping_patience = 10, early_stopping_tolerance = 1e-3, early_stopping_warmup = 0, 
            path_to_weights = None):
        '''pre train the model with specified learning rate
        Params:
            learning_rate            - (float) the initial learning rate for the Adam optimizer.
            batch_size               - (int) the batch size for pre-training.
            L                        - (int) the number of MC samples.
            num_epoch                - (int) the maximum number of epoches.
            num_step_per_epoch       - (int) the number of step per epoch, it will be inferred from 
                                            number of cells and batch size if it is None.            
            early_stopping_patience  - (int) the maximum number of epoches if there is no improvement.
            early_stopping_tolerance - (float) the minimum change of loss to be considered as an improvement.
            early_stopping_warmup    - (int) the number of warmup epoches.
            path_to_weights          - (str) the path of weight file to be saved; not saving weight if None.
        '''    
        if num_step_per_epoch is None:
            num_step_per_epoch = self.X.shape[0]//batch_size+1
                
        train.clear_session()
        self.train_dataset = train.warp_dataset(self.X_normalized, 
                                                self.c_score,
                                                batch_size, 
                                                self.X, 
                                                self.scale_factor)
        self.vae = train.pre_train(
            self.train_dataset,
            self.vae,
            learning_rate,
            early_stopping_patience,
            early_stopping_tolerance,
            early_stopping_warmup,
            num_epoch,
            num_step_per_epoch,
            L, alpha)

        if path_to_weights is not None:
            self.save_model(path_to_weights)
          

    def get_latent_z(self):
        c = None if self.c_score is None else self.c_score[self.selected_cell_subset_id,:]
        return self.vae.get_z(self.X_normalized[self.selected_cell_subset_id,:], c)


    def set_cell_subset(self, selected_cell_names):
        self.selected_cell_subset = np.unique(selected_cell_names)
        self.selected_cell_subset_id = np.sort(np.where(np.in1d(self.cell_names, selected_cell_names))[0])
        
    
    def refine_pi(self, batch_size=64):  
        '''
        Refine pi by the its posterior. This function will be effected if 
        'selected_cell_subset_id' is set.
        Params:
            batch_size  - (int) the batch size when computing p(c|Y).
        Returns:
            pi          - (2d array) the original pi.
            post_pi     - (2d array) the posterior estimate of pi.
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


    def init_latent_space(self, n_clusters, cluster_labels=None, mu=None, log_pi=None):
        z = self.get_latent_z()
        if (mu is None) & (cluster_labels is not None):
            mu = np.zeros((z.shape[1], n_clusters))
            for i,l in enumerate(np.unique(cluster_labels)):
                mu[:,i] = np.mean(z[cluster_labels==l], axis=0)

        self.n_clusters = n_clusters
        self.cluster_labels = None if cluster_labels is None else np.array(cluster_labels)
        self.vae.init_latent_space(n_clusters, mu, log_pi)
        self.inferer = Inferer(self.n_clusters)            


    def train(self, learning_rate = 1e-3, batch_size = 32, 
            L = 1, alpha=0.01, beta=1, 
            num_epoch = 300, num_step_per_epoch = None,
            early_stopping_patience = 10, early_stopping_tolerance = 1e-3, early_stopping_warmup = 5,
            path_to_weights = None, plot_every_num_epoch=None, dimred='umap', **kwargs):
        '''
        Training the model.
        Params:
            learning_rate            - (float) the initial learning rate for the Adam optimizer.
            batch_size               - (int) the batch size for training.
            L                        - (int) the number of MC samples.
            alpha                    - (float) the value of alpha in [0,1] to encourage covariate 
                                            adjustment. Not used if there is no covariates.
            beta                     - (float) the value of beta in beta-VAE.
            num_epoch                - (int) the number of epoch.
            num_step_per_epoch       - (int) the number of step per epoch, it will be inferred from 
                                            number of cells and batch size if it is None.
            early_stopping_patience  - (int) the maximum number of epoches if there is no improvement.
            early_stopping_tolerance - (float) the minimum change of loss to be considered as an 
                                            improvement.
            early_stopping_warmup    - (int) the number of warmup epoches.            
            path_to_weights          - (str) the path of weight file to be saved; not saving weight 
                                            if None.
            plot_every_num_epoch     - (int) plot the intermediate result every few epoches, or not 
                                            plotting if it is None.            
            dimred                   - (str) the name of dimension reduction algorithms, can be 'umap', 
                                            'pca' and 'tsne'. Only used if 'plot_every_num_epoch' is not None. 
            **kwargs                 - extra key-value arguments for dimension reduction algorithms.        
        Retruns:
        '''
        if num_step_per_epoch is None:
            num_step_per_epoch = len(self.selected_cell_subset_id)//batch_size+1
            
        c = None if self.c_score is None else self.c_score[self.selected_cell_subset_id,:]
        self.train_dataset = train.warp_dataset(self.X_normalized[self.selected_cell_subset_id,:],
                                                c,
                                                batch_size, 
                                                self.X[self.selected_cell_subset_id,:], 
                                                self.scale_factor[self.selected_cell_subset_id])
        self.test_dataset = train.warp_dataset(self.X_normalized[self.selected_cell_subset_id,:], 
                                               c,
                                               batch_size)
        self.vae = train.train(
            self.train_dataset,
            self.test_dataset,
            self.vae,
            learning_rate,
            early_stopping_patience,
            early_stopping_tolerance,
            early_stopping_warmup,
            num_epoch,
            num_step_per_epoch,
            L,
            alpha,
            beta,
            self.labels[self.selected_cell_subset_id],            
            plot_every_num_epoch,
            dimred='umap', 
            **kwargs            
            )
            
        if path_to_weights is not None:
            self.save_model(path_to_weights)
          

    def init_inference(self, batch_size=32, L=5, 
            dimred='umap', refit_dimred=True, **kwargs):
        '''
        Initialze trajectory inference by computing the posterior estimations.        
        Params:
            batch_size   - (int) batch size when doing inference.
            L            - (int) number of MC samples when doing inference.
            dimred       - (str) name of dimension reduction algorithms, can be 'umap', 'pca' and 'tsne'.
            refit_dimred - (boolean) if refit the dimension reduction algorithm or not.
            **kwargs     - extra key-value arguments for dimension reduction algorithms.              
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
        
        
    def comp_inference_score(self, method='modified_map', thres=0.5, no_loop=False, is_plot=True, path=None):
        '''
        Params:
            thres   - threshold used for filtering edges e_{ij} that
                      (n_{i}+n_{j}+e_{ij})/N<thres, only applied to
                      mean method.            
            method  - (string) 'mean', 'modified_mean', 'map', and 'modified_map'
            no_loop - (boolean) if loops are allowed to exist in the graph.
            is_plot - (boolean) whether to plot or not.
            path    - (string) path to save figure, or don't save if it is None.
        Returns:
            G       - (networkx.Graph) a weighted graph with weight on each edge
                      indicating its score of existence.
        '''
        G, edges = self.inferer.init_inference(self.w_tilde, self.pc_x, thres, method, no_loop)
        if is_plot:
            self.inferer.plot_clusters(self.cluster_labels, path=path)
        return G
        
        
    def infer_trajectory(self, init_node: int, cutoff=None, is_plot=True, path=None):
        '''
        Params:
            init_node  - (int) the initial node for the inferred trajectory.
            cutoff     - (string) threshold for filtering edges with scores less than cutoff.
            is_plot    - (boolean) whether to plot or not.
            path       - (string) path to save figure, or don't save if it is None.
        Returns:
            G          - (networkx.Graph) modified graph that indicates the inferred trajectory
            w          - (numpy.array) modified w_tilde
            pseudotime - (numpy.array) pseudotime based on projected trajectory
        '''
        G, w, pseudotime = self.inferer.infer_trajectory(init_node, 
                                                         self.label_names[self.selected_cell_subset_id], 
                                                         cutoff, 
                                                         path=path, 
                                                         is_plot=is_plot)
        return G, w, pseudotime

    
    def plot_marker_gene(self, gene_name: str, refit_dimred=False, dimred='umap', path=None, **kwargs):
        '''
        Plot expression of the given marker gene.
        Params:
            gene_name    - (str) name of the marker gene.
            refit_dimred - (boolean) whether to refit dimension reduction or use the existing embedding 
                                after inference.
            dimred       - (str) name of dimension reduction algorithms, can be 'umap', 'pca' and 'tsne'.
            path         - (str) path to save the figure, or not saving if it is None.
            **kwargs     - extra key-value arguments for dimension reduction algorithms.
        Returns:
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


    def evaluate(self, milestone_net, begin_node_true, grouping=None,
                thres=0.5, no_loop=True, cutoff=None,
                method='mean', path=None):
        '''
        Params:
            milestone_net   - True milestone network.
                              For real data, milestone_net will be a DataFrame of the graph of nodes.
                              Eg.
                                  from         to
                                  cluster 1    cluster 2
                                  cluster 2    cluster 3
                              For synthetic data, milestone_net will be a DataFrame of the (projected)
                              positions of cells. The indexes are the orders of cells in the dataset.
                              Eg.
                                  from         to          w
                                  cluster 1    cluster 1   1
                                  cluster 1    cluster 2   0.1
            begin_node_true - True begin node of the milestone.
            grouping        - For real data, grouping must be provided.
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
            res['PDT score'] = np.corrcoef(pseudotime_true,pseudotime_pred)[0,1]
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