import model
import preprocess
import inference
import util
import train

class scTGMVAE():
	"""docstring for scTGMVAE"""
	def __init__(self):
		pass

	def get_data(self, X, grouping = None):
		self.X = X
		self.grouping = grouping		

	def data_preprocess(self, K = 1e4, gene_num = 2000):
		self.X_normalized, self.X, self.scale_factor, self.label, self.le = preprocess.preprocess(self.X, self.grouping, K, gene_num)

	def get_params(self, 
		n_clusters = 3, 
		dim_origin = self.X.shape[1], 
		dimensions = [16], 
		dim_latent = 8, 
		data_type = 'UMI',
		EARLY_STOPPING_PATIENCE = 10,
		EARLY_STOPPING_TOLERANCE = 1e-3,
		BATCH_SIZE = 32,
		NUM_EPOCH_PRE = 300,
		NUM_STEP_PER_EPOCH = self.X.shape[0]//BATCH_SIZE+1,
		NUM_EPOCH = 1000
		):
		self.n_clusters = n_clusters
		self.dim_origin = dim_origin
		self.dimensions = dimensions
		self.dim_latent = dim_latent
		self.data_type = data_type
		self.EARLY_STOPPING_PATIENCE = EARLY_STOPPING_PATIENCE
		self.EARLY_STOPPING_TOLERANCE= EARLY_STOPPING_TOLERANCE
		self.BATCH_SIZE = BATCH_SIZE
		self.NUM_EPOCH_PRE = NUM_EPOCH_PRE
		self.NUM_STEP_PER_EPOCH = NUM_STEP_PER_EPOCH
		self.NUM_EPOCH = NUM_EPOCH

	def model_init(self):
		self.vae = model.VariationalAutoEncoder(
			self.n_clusters, 
			self.dim_origin, 
			self.dimensions, 
			self.dim_latent,
			self.data_type)
		self.train_dataset = train.warp_dataset(self.X, self.X_normalized, self.scale_factor, self.BATCH_SIZE, self.data_type)

	def pre_train(self, learning_rate = 1e-4):
		train.clear_session()
		self.vae = train.pre_train(
			self.train_dataset, self.vae, 
			learining_rate, 
			self.patience, 
			self.tolerance, 
			self.NUM_EPOCH_PRE, 
			self.NUM_STEP_PER_EPOCH)

	def init_GMM_plot(self):
		self.vae = train.init_GMM(self.vae, self.X_normalized, self.n_clusters)
		train.plot_pre_train(self,vae, self.X_normalized)

	def train_together(self, learining_rate = 1e-4):
		self.vae = train.trainTogether(
			self.train_dataset, 
			self.vae, 
			learining_rate, 
			self.EARLY_STOPPING_PATIENCE, 
			self.EARLY_STOPPING_TOLERANCE, 
			self.NUM_EPOCH, 
			self.NUM_STEP_PER_EPOCH)

	def train(self, pre_train_learning_rate = 1e-4, train_learning_rate = 1e-4):
		self.pre_train(pre_train_learining_rate)
		self.init_GMM_plot()
		self.train_together(train_learning_rate)


	def inference(self):
		pass

