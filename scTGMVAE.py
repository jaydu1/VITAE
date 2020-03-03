import model
import preprocess
import train
import numpy as np
from inference import Inferer

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
        self.X = X
        self.grouping = grouping        

    # data preprocessing, feature selection, log-normalization
    # K: the constant summing gene expression in each cell up to
    # gene_num: number of feature to select
    def preprocess_data(self, K = 1e4, gene_num = 2000):
        self.X_normalized, self.X, self.scale_factor, self.label, self.le = preprocess.preprocess(self.X, self.grouping, K, gene_num)
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
        n_clusters = 3, 
        dimensions = [16], 
        dim_latent = 8, 
        data_type = 'UMI',
        EARLY_STOPPING_PATIENCE = 10,
        EARLY_STOPPING_TOLERANCE = 1e-3,
        BATCH_SIZE = 32,
        NUM_EPOCH_PRE = 300,
        NUM_STEP_PER_EPOCH = None,
        NUM_EPOCH = 1000
        ):
        self.n_clusters = n_clusters
        self.dimensions = dimensions
        self.dim_latent = dim_latent
        self.data_type = data_type
        self.EARLY_STOPPING_PATIENCE = EARLY_STOPPING_PATIENCE
        self.EARLY_STOPPING_TOLERANCE= EARLY_STOPPING_TOLERANCE
        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_EPOCH_PRE = NUM_EPOCH_PRE
        self.NUM_EPOCH = NUM_EPOCH
        if NUM_STEP_PER_EPOCH is None:
            self.NUM_STEP_PER_EPOCH = self.X.shape[0]//BATCH_SIZE+1
        else:
            self.NUM_STEP_PER_EPOCH = NUM_STEP_PER_EPOCH

        self.train_dataset = train.warp_dataset(self.X, self.X_normalized, self.scale_factor, self.BATCH_SIZE, self.data_type)
    
        self.vae = model.VariationalAutoEncoder(
            self.n_clusters, 
            self.dim_origin, 
            self.dimensions, 
            self.dim_latent,
            self.data_type)
        
        self.inferer = Inferer(self.n_clusters)
        
    # save and load trained model parameters
    # path: path of checkpoints files
    def save_model(self, path):
        self.vae.save_weights(path)
    
    def load_model(self, path):
        self.vae.load_weights(path)

    # pre train the model with specified learning rate
    def pre_train(self, learning_rate = 1e-4):
        train.clear_session()
        self.vae = train.pre_train(
            self.train_dataset, 
            self.vae, 
            learning_rate, 
            self.EARLY_STOPPING_PATIENCE, 
            self.EARLY_STOPPING_TOLERANCE, 
            self.NUM_EPOCH_PRE, 
            self.NUM_STEP_PER_EPOCH)

    # initialize parameters in GMM after pre train
    # plot the UMAP latent space after pre train
    def init_GMM_plot(self):
        self.vae = train.init_GMM(self.vae, self.X_normalized, self.n_clusters)
        train.plot_pre_train(self.vae, self.X_normalized, self.label)

    # train the model with specified learning rate
    def train_together(self, learning_rate = 1e-4):
        self.vae = train.trainTogether(
            self.train_dataset, 
            self.vae, 
            learning_rate, 
            self.EARLY_STOPPING_PATIENCE, 
            self.EARLY_STOPPING_TOLERANCE, 
            self.NUM_EPOCH, 
            self.NUM_STEP_PER_EPOCH,
            self.label,
            self.X_normalized)

    # train the model with specified learning rate
    def train(self, pre_train_learning_rate = 1e-4, train_learning_rate = 1e-4):
        self.pre_train(pre_train_learning_rate)
        self.init_GMM_plot()
        self.train_together(train_learning_rate)

    # inference for trajectory
    def init_inference(self, metric='max_relative_score', no_loop=False):
        pi,mu,c,w,var_w,wc,var_wc,z,proj_z = self.vae(self.X_normalized, inference=True)
        
        cluster_center = [int((self.n_clusters+(1-i)/2)*i) for i in range(self.n_clusters)]
        edges = [i for i in np.unique(c) if i not in cluster_center]
        proj_c, proj_z_M = self.vae.get_proj_z(edges)
        
        self.inferer.init_inference(c, w, mu, z, proj_c, proj_z_M,
            metric=metric, no_loop=no_loop)
        
        
    def plot_trajectory(self, cutoff=None):
        self.inferer.plot_trajectory(self.grouping, cutoff=cutoff)
        
        
    def plot_pseudotime(self, init_node):
        self.inferer.plot_pseudotime(init_node)
