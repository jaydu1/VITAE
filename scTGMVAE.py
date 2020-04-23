import model
import preprocess
import train
import numpy as np
from inference import Inferer
import os

class scTGMVAE():
    """
    class for Gaussian Mixture Model for trajectory analysis
    step of analysis:
    get_data ---> data_preprocess ---> get_params ---> 
    model_init ---> pre_train ---> init_GMM_plot ---> train_together ---> 
    inference
    """
    def __init__(self):
        pass


	# get data for model
    # X: 2-dimension np array, original counts data
    # grouping: a list of labels for cells
    def get_data(self, X, grouping = None):
        self.raw_X = X
        self.grouping = np.array(grouping, dtype=str)


    # data preprocessing, feature selection, log-normalization
    # K: the constant summing gene expression in each cell up to
    # gene_num: number of feature to select
    def preprocess_data(self, K = 1e4, gene_num = 2000):
        self.X_normalized, self.X, self.scale_factor, self.label, self.le = preprocess.preprocess(self.raw_X.copy(), self.grouping, K, gene_num)
        self.dim_origin = self.X.shape[1]


    # get parameters, wrap up training dataset and initialize the Variational Auto Encoder model
    # n_clusters: number of Gaussian Mixtures, number of cell types
    # dimensions: a list of dimensions of layers of autoencoder between latent space and original space
    # dim_latent: dimension of latent space
    # data_type: 'UMI' and 'non-UMI', default is 'UMI'
    # NUM_EPOCH_PRE: number of epochs for pre training
    # NUM_EPOCH: number of epochs for training
    # NUM_STEP_PER_EPOCH: number of steps in each epoch, default is n/BATCH_SIZE+1
    def build_model(self,
        dimensions = [16], 
        dim_latent = 8,
        L = 5,
        data_type = 'UMI',
        save_weights = False,
        path_to_weights_pretrain = 'pre_train.checkpoint',
        path_to_weights_train = 'train.checkpoint'
        ):
        self.dimensions = dimensions
        self.dim_latent = dim_latent
        self.L = L
        self.data_type = data_type
        self.save_weights = save_weights
        self.path_to_weights_pretrain = path_to_weights_pretrain
        self.path_to_weights_train = path_to_weights_train
    
        self.vae = model.VariationalAutoEncoder(
            self.dim_origin, 
            self.dimensions, 
            self.dim_latent,
            self.L,
            self.data_type)
        
        
    # save and load trained model parameters
    # path: path of checkpoints files
    def save_model(self, path_to_file='model.checkpoint'):
        self.vae.save_weights(path_to_file)
    
    
    def load_model(self, path_to_file='model.checkpoint'):
        self.vae.load_weights(path_to_file)


    # pre train the model with specified learning rate
    def pre_train(self, learning_rate = 1e-3, batch_size = 32,
            num_epoch = 300, num_step_per_epoch = None,
            early_stopping_patience = 10, early_stopping_tolerance = 1e-3, L=None):
            
        if num_step_per_epoch is None:
            num_step_per_epoch = self.X.shape[0]//batch_size+1
                
        train.clear_session()
        self.train_dataset = train.warp_dataset(self.X, self.X_normalized, self.scale_factor, batch_size)
        self.vae = train.pre_train(
            self.train_dataset, 
            self.vae, 
            learning_rate, 
            early_stopping_patience,
            early_stopping_tolerance,
            num_epoch,
            num_step_per_epoch,
            L)
        if self.save_weights:
            self.save_model(self.path_to_weights_pretrain)
          
          
    # initialize parameters in GMM after pre train
    # plot the UMAP latent space after pre train
#    def init_GMM_plot(self):
#        self.vae = train.init_GMM(self.vae, self.X_normalized, self.n_clusters)
#        train.plot_pre_train(self.vae, self.X_normalized, self.label)

    def get_latent_z(self):
        return self.vae.get_z(self.X_normalized)


    def init_GMM(self, n_clusters, mu, Sigma=None, pi=None):
        self.n_clusters = n_clusters
        self.vae.init_GMM(n_clusters, mu, Sigma, pi)
        self.inferer = Inferer(self.n_clusters)


    # train the model with specified learning rate
    def train(self, learning_rate = 1e-3, batch_size = 32,
            num_epoch = 300, num_step_per_epoch = None,
            early_stopping_patience = 10, early_stopping_tolerance = 1e-3, L=None, weight=None):
        
        if num_step_per_epoch is None:
            num_step_per_epoch = self.X.shape[0]//batch_size+1
            
        self.train_dataset = train.warp_dataset(self.X, self.X_normalized, self.scale_factor, batch_size)
        self.vae = train.train(
            self.train_dataset, 
            self.vae, 
            learning_rate,
            early_stopping_patience,
            early_stopping_tolerance,
            num_epoch,
            num_step_per_epoch,
            L,
            self.label,
            self.X_normalized,
            weight
            )
        if self.save_weights:
            self.save_model(self.path_to_weights_train)
          
          
    # train the model with specified learning rate
    def train_all(self, learning_rate = 1e-3, batch_size = 32,
            num_epoch = 300, num_step_per_epoch = None,
            early_stopping_patience = 10, early_stopping_tolerance = 1e-3):
        '''
        To pretrain and train the model by using same parameters for pre_train() and train().
        '''
        train.clear_session()
        self.pre_train(learning_rate,
            batch_size,
            num_epoch,
            num_step_per_epoch,
            early_stopping_patience,
            early_stopping_tolerance)
        self.init_GMM_plot()
        self.train(learning_rate,
            batch_size,
            num_epoch,
            num_step_per_epoch,
            early_stopping_patience,
            early_stopping_tolerance)


    # inference for trajectory
    def init_inference(self, metric='max_relative_score', no_loop=False):
        pi,mu,c,w,var_w,wc,var_wc,z,proj_z = self.vae(self.X_normalized, inference=True)
        
        cluster_center = [int((self.n_clusters+(1-i)/2)*i) for i in range(self.n_clusters)]
        edges = [i for i in np.unique(c) if i not in cluster_center]
        if len(edges)==0:
            proj_c, proj_z_M = c, None
        else:
            proj_c, proj_z_M = self.vae.get_proj_z(edges)
        
        self.inferer.init_inference(c, w, mu, z, proj_c, proj_z_M,
            metric=metric, no_loop=no_loop)
        return w, var_w, wc, var_wc
        
    def plot_trajectory(self, cutoff=None):
        self.inferer.plot_trajectory(self.grouping, cutoff=cutoff)
        
        
    def plot_pseudotime(self, init_node):
        self.inferer.plot_pseudotime(init_node)
