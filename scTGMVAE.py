import numpy as np
import scipy as sp
import pandas as pd
import networkx as nx
from sklearn.metrics.cluster import adjusted_rand_score
import warnings
import os

import model
import preprocess
import train
from inference import Inferer
from utils import get_igraph, louvain_igraph, plot_clusters, load_data, plot_marker_gene
from metric import topology, get_RI_continuous
from scipy.spatial.distance import pdist as dist
import umap

class scTGMVAE():
    """
    class for single-cell Trajectory Variational Autoencoder.
    """
    def __init__(self):
        pass

    def get_data(self, X, labels = None, covariate=None, cell_names = None, gene_names = None):
        ''' get data for model
        Params:
            X:          - 2-dimension np array, counts or expressions data
            covariate   - 2-dimension np array, covariate data
            labels:     - (optional) a list of labelss for cells
            cell_names  - (optional) a list of cell names
            gene_names  - (optional) a list of gene names
        '''
        self.raw_X = X.astype(np.float32)
        self.c_score = None if covariate is None else np.array(covariate, np.float32)
        if sp.sparse.issparse(self.raw_X):
            self.raw_X = self.raw_X.toarray()
        self.raw_label_names = None if labels is None else np.array(labels, dtype = str)
        self.raw_cell_names = None if cell_names is None else np.array(cell_names, dtype = str)
        self.raw_gene_names = None if gene_names is None else np.array(gene_names, dtype = str)

        
    def preprocess_data(self, K = 1e4, gene_num = 2000, data_type = 'UMI', npc = 64):
        ''' data preprocessing, feature selection, log-normalization
            This step will transform both X and X_normalized into PCs and equal if Gaussian_input
        Params:
            K               - the constant summing gene expression in each cell up to
            gene_num        - number of feature to select
            data_type       - 'UMI', 'non-UMI' and 'Gaussian', default is 'UMI'
            npc             - Number of PCs
        '''
        if data_type not in set(['UMI', 'non-UMI', 'Gaussian']):
            raise ValueError("Invalid data type, must be one of 'UMI', 'non-UMI', and 'Gaussian'.")

        self.data_type = data_type
        self.X_normalized, self.X, self.c_score, self.cell_names, self.gene_names, \
        self.scale_factor, self.labels, self.label_names, \
        self.le, self.gene_scalar = preprocess.preprocess(
            self.raw_X.copy(),
            self.c_score,
            self.raw_label_names,
            self.raw_cell_names,
            self.raw_gene_names,
            data_type, K, gene_num, npc)
        self.dim_origin = self.X.shape[1]
        self.selected_cell_subset = self.cell_names
        self.selected_cell_subset_id = np.arange(len(self.cell_names))

        
    def build_model(self,
        dimensions = [16],
        dim_latent = 8,   
        ):
        ''' get parameters, wrap up training dataset and initialize the Variational Auto Encoder model
        Params:
            dimensions          - a list of dimensions of layers of autoencoder between latent 
                                  space and original space
            dim_latent          - dimension of latent space
        '''
        self.dimensions = dimensions
        self.dim_latent = dim_latent
    
        self.vae = model.VariationalAutoEncoder(
            self.dim_origin,
            self.dimensions,
            self.dim_latent,
            self.data_type,
            False if self.c_score is None else True
            )
        
        
    # save and load trained model parameters
    # path: path of checkpoints files
    def save_model(self, path_to_file='model.checkpoint'):
        self.vae.save_weights(path_to_file)
    
    
    def load_model(self, path_to_file='model.checkpoint', n_clusters=None):
        '''
        Params:
            path_to_file - path to weight files of pre trained or
                           trained model
            n_clusters   - if n_cluster is provided, then the GMM layer
                           will be initialized or re-initialized. For loading
                           a trained model when the GMM layer is not
                           initialized, n_cluster is required.
        '''
        if n_clusters is not None:
            self.init_GMM(n_clusters)
        self.vae.load_weights(path_to_file)


    def pre_train(self, learning_rate = 1e-3, batch_size = 32, L = 1,
            num_epoch = 300, num_step_per_epoch = None,
            early_stopping_patience = 10, early_stopping_tolerance = 1e-3, early_stopping_warmup = 0, 
            path_to_weights = None):
        '''pre train the model with specified learning rate
        Params:
            learning_rate            - the initial learning rate for the Adam optimizer.
            batch_size               - the batch size for pre-training.
            L                        - the number of MC samples.
            num_epoch                - the maximum number of epoches.
            num_step_per_epoch       - the number of step per epoch, it will be inferred from number of cells and batch size if it is None
            early_stopping_tolerance - the minimum change of loss to be considered as an improvement.
            early_stopping_patience  - the maximum number of epoches if there is no improvement.
            early_stopping_warmup    - the number of warmup epoches.
            path_to_weights          - the path of weight file to be saved; not saving weight if None.
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
            L)

        if path_to_weights is not None:
            self.save_model(path_to_weights)
          

    def get_latent_z(self):
        c = None if self.c_score is None else self.c_score[self.selected_cell_subset_id,:]
        return self.vae.get_z(self.X_normalized[self.selected_cell_subset_id,:], c)


    def set_cell_subset(self, selected_cell_names):
        self.selected_cell_subset = selected_cell_names
        self.selected_cell_subset_id = np.sort(np.where(np.in1d(selected_cell_names, self.cell_names))[0])
        
    
    def init_GMM(self, n_clusters, cluster_labels=None, mu=None, log_pi=None):
        self.n_clusters = n_clusters
        self.cluster_labels = None if cluster_labels is None else np.array(cluster_labels)
        self.vae.init_GMM(n_clusters, mu, log_pi)
        self.inferer = Inferer(self.n_clusters)            


    # train the model with specified learning rate
    def train(self, learning_rate = 1e-3, batch_size = 32, L = 1,
            num_epoch = 300, num_step_per_epoch = None,
            early_stopping_patience = 10, early_stopping_tolerance = 1e-3, early_stopping_warmup = 0,
            weight=None, plot_every_num_epoch=None,
            path_to_weights = None):
        
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
            self.labels[self.selected_cell_subset_id],
            weight,
            plot_every_num_epoch
            )
            
        if path_to_weights is not None:
            self.save_model(path_to_weights)
          

    def init_inference(self, batch_size=32, L=5):
        '''
        Initialze trajectory inference by computing the posterior estimations.
        '''
        c = None if self.c_score is None else self.c_score[self.selected_cell_subset_id,:]
        self.test_dataset = train.warp_dataset(self.X_normalized[self.selected_cell_subset_id,:], 
                                               c,
                                               batch_size)
        self.pi, self.mu, self.c, self.pc_x,\
            self.p_wc_x,self.w,self.var_w,self.wc,self.var_wc,\
            self.w_tilde,self.var_w_tilde,self.D_JS,self.z = self.vae.inference(self.test_dataset, L=L)
        self.embed_z = self.inferer.init_embedding(self.z, self.mu)
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

    
    def plot_marker_gene(self, gene_name: str, path=None):
        if gene_name not in self.gene_names:
            raise ValueError("Gene name '{}' not in selected genes!".format(gene_name))
        expression = self.X_normalized[self.selected_cell_subset_id,:][:,self.gene_names==gene_name].flatten()
        
        if not hasattr(self, 'embed_z'):
            self.z = self.get_latent_z()            
            self.embed_z = umap.UMAP().fit_transform(self.z)
        plot_marker_gene(expression, 
                         gene_name, 
                         self.embed_z,
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
        res['score_ARI_discrete'] = (adjusted_rand_score(milestones_true, milestones_pred) + 1)/2
        
        if grouping is None:
            n_samples = len(milestone_net)
            prop = np.zeros((n_samples,n_samples))
            prop[np.arange(n_samples), milestone_net['to']] = 1-milestone_net['w']
            prop[np.arange(n_samples), milestone_net['from']] = np.where(np.isnan(milestone_net['w']), 1, milestone_net['w'])
            res['score_RI_continuous'] = get_RI_continuous(prop, w)
        else:
            res['score_RI_continuous'] = get_RI_continuous(grouping, w)
        
        # 3. Correlation between geodesic distances / Pseudotime
        if grouping is None:
            pseudotime_ture = milestone_net['from'].values + 1 - milestone_net['w'].values
            pseudotime_ture[np.isnan(pseudotime_ture)] = milestone_net[pd.isna(milestone_net['w'])]['from'].values
            pseudotime_ture = pseudotime_ture[pseudotime>-1]
            pseudotime_pred = pseudotime[pseudotime>-1]
            res['score_cor'] = np.corrcoef(pseudotime_ture,pseudotime_pred)[0,1]
        
        # 4. Shape
        score_cos_theta = 0
        for (_from,_to) in G.edges:
            _z = self.z[(w[:,_from]>0) & (w[:,_to]>0),:]
            v_1 = _z - self.mu[:,_from]
            v_2 = _z - self.mu[:,_to]
            cos_theta = np.sum(v_1*v_2, -1)/(np.linalg.norm(v_1,axis=-1)*np.linalg.norm(v_2,axis=-1)+1e-12)

            score_cos_theta += np.sum((1-cos_theta)/2)

        res['score_cos_theta'] = score_cos_theta/(np.sum(np.sum(w>0, axis=-1)==2)+1e-12)
        return res


    def plot_output(self, init_node, batchsize = 32, cutoff=None, gene=None, thres=0.5, method='mean'):
        # dim_red
        z = self.get_latent_z()
        embed_z = umap.UMAP().fit_transform(z)
        np.savetxt('dimred.csv', embed_z)

        # cell_ids
        cell_ids = self.cell_names[self.selected_cell_subset_id]
        np.savetxt('cell_ids.csv', cell_ids, fmt="%s")

        # feature_ids (gene)
        feature_ids = self.gene_names
        np.savetxt('feature_ids.csv', feature_ids, fmt="%s")

        # grouping
        np.savetxt('grouping.csv', self.label_names[self.selected_cell_subset_id], fmt="%s")

        # milestone_network
        self.init_inference(batch_size=batchsize, L=300)
        G = self.comp_inference_score(no_loop=True, method=method, thres=thres)
        G, modified_w_tilde, pseudotime = self.inferer.infer_trajectory(init_node, 
                                                                        self.label_names[self.selected_cell_subset_id], 
                                                                        cutoff, 
                                                                        is_plot = False)
        from_to = self.inferer.build_milestone_net(G, init_node)[:,:2]
        fromm = from_to[:,0][from_to[:,0] != None]
        to = from_to[:,1][from_to[:,0] != None]
        dd = np.zeros((self.n_clusters,self.n_clusters))
        dd[np.triu_indices(self.n_clusters,1)]=dist(self.mu.T)
        dd += dd.T
        length = [dd[fromm[i], to[i]] for i in range(len(fromm))]
        fromm = ['M'+str(j) for j in fromm]
        to = ['M'+str(j) for j in to]
        milestone_network = pd.DataFrame({'from': fromm, 'to': to, 'length': length, 'directed': [True] * len(to)})
        milestone_network.to_csv('milestone_network.csv', index = False)

        # milestone_percentage
        cell_id = []
        milestone_id = []
        percentage = []
        for i in range(len(z)):
            ind = np.where(modified_w_tilde[i,:]!=0)[0]
            cell_id += np.repeat(cell_ids[i], len(ind)).tolist()
            milestone_id += ['M'+str(j) for j in ind]
            percentage += modified_w_tilde[i, ind].tolist()
        milestone_percentages = pd.DataFrame({'cell_id': cell_id, 'milestone_id': milestone_id, 'percentage': percentage})
        milestone_percentages.to_csv('milestone_percentages.csv', index = False)

        # pseudotime
        np.savetxt('pseudotime.csv', pseudotime)

        # posterior variance
        np.savetxt('pos_var.csv', self.var_w_tilde.sum(axis=1))        

        # gene_express
        if gene is not None:
            np.savetxt('gene_express.csv', self.X_normalized[self.selected_cell_subset_id,:][:,self.gene_names == gene])
