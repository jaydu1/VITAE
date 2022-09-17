import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, BatchNormalization
import tensorflow_probability as tfp
from tensorflow.keras.utils import Progbar
from .utils import compute_mmd, _nelem, _nan2zero, _nan2inf, _reduce_mean
from tensorflow.keras import backend as K
import time

 
class cdf_layer(Layer):
    '''
    The Normal cdf layer with custom gradients.
    '''
    def __init__(self):
        '''
        '''
        super(cdf_layer, self).__init__()
        
    @tf.function
    def call(self, x):
        return self.func(x)
        
    @tf.custom_gradient
    def func(self, x):
        '''Return cdf(x) and pdf(x).

        Parameters
        ----------
        x : tf.Tensor
            The input tensor.
        
        Returns
        ----------
        f : tf.Tensor
            cdf(x).
        grad : tf.Tensor
            pdf(x).
        '''   
        dist = tfp.distributions.Normal(
            loc = tf.constant(0.0, tf.keras.backend.floatx()), 
            scale = tf.constant(1.0, tf.keras.backend.floatx()), 
            allow_nan_stats=False)
        f = dist.cdf(x)
        def grad(dy):
            gradient = dist.prob(x)
            return dy * gradient
        return f, grad
    

class Sampling(Layer):
    """Sampling latent variable \(z\) from \(N(\\mu_z, \\log \\sigma_z^2\)).    
    Used in Encoder.
    """
    def __init__(self, seed=0, **kwargs):
        super(Sampling, self).__init__(**kwargs)
        self.seed = seed

    @tf.function
    def call(self, z_mean, z_log_var):
        '''Return cdf(x) and pdf(x).

        Parameters
        ----------
        z_mean : tf.Tensor
            \([B, L, d]\) The mean of \(z\).
        z_log_var : tf.Tensor
            \([B, L, d]\) The log-variance of \(z\).

        Returns
        ----------
        z : tf.Tensor
            \([B, L, d]\) The sampled \(z\).
        '''   
   #     seed = tfp.util.SeedStream(self.seed, salt="random_normal")
   #     epsilon = tf.random.normal(shape = tf.shape(z_mean), seed=seed(), dtype=tf.keras.backend.floatx())
        epsilon = tf.random.normal(shape = tf.shape(z_mean), dtype=tf.keras.backend.floatx())
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        z = tf.clip_by_value(z, -1e6, 1e6)
        return z

class Encoder(Layer):
    '''
    Encoder, model \(p(Z_i|Y_i,X_i)\).
    '''
    def __init__(self, dimensions, dim_latent, name='encoder', **kwargs):
        '''
        Parameters
        ----------
        dimensions : np.array
            The dimensions of hidden layers of the encoder.
        dim_latent : int
            The latent dimension of the encoder.
        name : str, optional
            The name of the layer.
        **kwargs : 
            Extra keyword arguments.
        ''' 
        super(Encoder, self).__init__(name = name, **kwargs)
        self.dense_layers = [Dense(dim, activation = tf.nn.leaky_relu,
                                          name = 'encoder_%i'%(i+1)) \
                             for (i, dim) in enumerate(dimensions)]
        self.batch_norm_layers = [BatchNormalization(center=False) \
                                    for _ in range(len((dimensions)))]
        self.batch_norm_layers.append(BatchNormalization(center=False))
        self.latent_mean = Dense(dim_latent, name = 'latent_mean')
        self.latent_log_var = Dense(dim_latent, name = 'latent_log_var')
        self.sampling = Sampling()
    
    @tf.function
    def call(self, x, L=1, is_training=True):
        '''Encode the inputs and get the latent variables.

        Parameters
        ----------
        x : tf.Tensor
            \([B, L, d]\) The input.
        L : int, optional
            The number of MC samples.
        is_training : boolean, optional
            Whether in the training or inference mode.
        
        Returns
        ----------
        z_mean : tf.Tensor
            \([B, L, d]\) The mean of \(z\).
        z_log_var : tf.Tensor
            \([B, L, d]\) The log-variance of \(z\).
        z : tf.Tensor
            \([B, L, d]\) The sampled \(z\).
        '''         
        for dense, bn in zip(self.dense_layers, self.batch_norm_layers):
            x = dense(x)
            x = bn(x, training=is_training)
        z_mean = self.batch_norm_layers[-1](self.latent_mean(x), training=is_training)
        z_log_var = self.latent_log_var(x)
        _z_mean = tf.tile(tf.expand_dims(z_mean, 1), (1,L,1))
        _z_log_var = tf.tile(tf.expand_dims(z_log_var, 1), (1,L,1))
        z = self.sampling(_z_mean, _z_log_var)
        return z_mean, z_log_var, z


class Decoder(Layer):
    '''
    Decoder, model \(p(Y_i|Z_i,X_i)\).
    '''
    def __init__(self, dimensions, dim_origin, data_type = 'UMI', 
                name = 'decoder', **kwargs):
        '''
        Parameters
        ----------
        dimensions : np.array
            The dimensions of hidden layers of the encoder.
        dim_origin : int
            The output dimension of the decoder.
        data_type : str, optional
            `'UMI'`, `'non-UMI'`, or `'Gaussian'`.
        name : str, optional
            The name of the layer.
        '''
        super(Decoder, self).__init__(name = name, **kwargs)
        self.data_type = data_type
        self.dense_layers = [Dense(dim, activation = tf.nn.leaky_relu,
                                          name = 'decoder_%i'%(i+1)) \
                             for (i,dim) in enumerate(dimensions)]
        self.batch_norm_layers = [BatchNormalization(center=False) \
                                    for _ in range(len((dimensions)))]

        if data_type=='Gaussian':
            self.nu_z = Dense(dim_origin, name = 'nu_z')
            # common variance
            self.log_tau = tf.Variable(tf.zeros([1, dim_origin], dtype=tf.keras.backend.floatx()),
                                 constraint = lambda t: tf.clip_by_value(t,-30.,6.),
                                 name = "log_tau")
        else:
            self.log_lambda_z = Dense(dim_origin, name = 'log_lambda_z')

            # dispersion parameter
            self.log_r = tf.Variable(tf.zeros([1, dim_origin], dtype=tf.keras.backend.floatx()),
                                     constraint = lambda t: tf.clip_by_value(t,-30.,6.),
                                     name = "log_r")
            
            if self.data_type == 'non-UMI':
                self.phi = Dense(dim_origin, activation = 'sigmoid', name = "phi")
          
    @tf.function  
    def call(self, z, is_training=True):
        '''Decode the latent variables and get the reconstructions.

        Parameters
        ----------
        z : tf.Tensor
            \([B, L, d]\) the sampled \(z\).
        is_training : boolean, optional
            whether in the training or inference mode.

        When `data_type=='Gaussian'`:

        Returns
        ----------
        nu_z : tf.Tensor
            \([B, L, G]\) The mean of \(Y_i|Z_i,X_i\).
        tau : tf.Tensor
            \([1, G]\) The variance of \(Y_i|Z_i,X_i\).

        When `data_type=='UMI'`:

        Returns
        ----------
        lambda_z : tf.Tensor
            \([B, L, G]\) The mean of \(Y_i|Z_i,X_i\).
        r : tf.Tensor
            \([1, G]\) The dispersion parameters of \(Y_i|Z_i,X_i\).

        When `data_type=='non-UMI'`:

        Returns
        ----------
        lambda_z : tf.Tensor
            \([B, L, G]\) The mean of \(Y_i|Z_i,X_i\).
        r : tf.Tensor
            \([1, G]\) The dispersion parameters of \(Y_i|Z_i,X_i\).
        phi_z : tf.Tensor
            \([1, G]\) The zero inflated parameters of \(Y_i|Z_i,X_i\).
        '''
        for dense, bn in zip(self.dense_layers, self.batch_norm_layers):
            z = dense(z)
            z = bn(z, training=is_training)
        if self.data_type=='Gaussian':
            nu_z = self.nu_z(z)
            tau = tf.exp(self.log_tau)
            return nu_z, tau
        else:
            lambda_z = tf.math.exp(
                tf.clip_by_value(self.log_lambda_z(z), -30., 6.)
                )
            r = tf.exp(self.log_r)
            if self.data_type=='UMI':
                return lambda_z, r
            else:
                return lambda_z, r, self.phi(z)


class LatentSpace(Layer):
    '''
    Layer for the Latent Space.
    '''
    def __init__(self, n_clusters, dim_latent,
            name = 'LatentSpace', seed=0, **kwargs):
        '''
        Parameters
        ----------
        n_clusters : int
            The number of vertices in the latent space.
        dim_latent : int
            The latent dimension.
        M : int, optional
            The discretized number of uniform(0,1).
        name : str, optional
            The name of the layer.
        **kwargs : 
            Extra keyword arguments.
        '''
        super(LatentSpace, self).__init__(name=name, **kwargs)
        self.dim_latent = dim_latent
        self.n_states = n_clusters
        self.n_categories = int(n_clusters*(n_clusters+1)/2)

        # nonzero indexes
        # A = [0,0,...,0  , 1,1,...,1,   ...]
        # B = [0,1,...,k-1, 1,2,...,k-1, ...]
        self.A, self.B = np.nonzero(np.triu(np.ones(n_clusters)))
        self.A = tf.convert_to_tensor(self.A, tf.int32)
        self.B = tf.convert_to_tensor(self.B, tf.int32)
        self.clusters_ind = tf.boolean_mask(
            tf.range(0,self.n_categories,1), self.A==self.B)

        # [pi_1, ... , pi_K] in R^(n_categories)
        self.pi = tf.Variable(tf.ones([1, self.n_categories], dtype=tf.keras.backend.floatx()) / self.n_categories,
                                name = 'pi')
        
        # [mu_1, ... , mu_K] in R^(dim_latent * n_clusters)
        self.mu = tf.Variable(tf.random.uniform([self.dim_latent, self.n_states],
                                                minval = -1, maxval = 1, seed=seed, dtype=tf.keras.backend.floatx()),
                                name = 'mu')
        self.cdf_layer = cdf_layer()       
        
    def initialize(self, mu, log_pi):
        '''Initialize the latent space.

        Parameters
        ----------
        mu : np.array
            \([d, k]\) The position matrix.
        log_pi : np.array
            \([1, K]\) \(\\log\\pi\).
        '''
        # Initialize parameters of the latent space
        if mu is not None:
            self.mu.assign(mu)
        if log_pi is not None:
            self.pi.assign(log_pi)

    def normalize(self):
        '''Normalize \(\\pi\).
        '''
        self.pi = tf.nn.softmax(self.pi)

    @tf.function
    def _get_normal_params(self, z, pi):
        batch_size = tf.shape(z)[0]
        L = tf.shape(z)[1]
        
        # [batch_size, L, n_categories]
        if pi is None:
            # [batch_size, L, n_states]
            temp_pi = tf.tile(
                tf.expand_dims(tf.nn.softmax(self.pi), 1),
                (batch_size,L,1))
        else:
            temp_pi = tf.expand_dims(tf.nn.softmax(pi), 1)

        # [batch_size, L, d, n_categories]
        alpha_zc = tf.expand_dims(tf.expand_dims(
            tf.gather(self.mu, self.B, axis=1) - tf.gather(self.mu, self.A, axis=1), 0), 0)
        beta_zc = tf.expand_dims(z,-1) - \
            tf.expand_dims(tf.expand_dims(
            tf.gather(self.mu, self.B, axis=1), 0), 0)
            
        # [batch_size, L, n_categories]
        _inv_sig = tf.reduce_sum(alpha_zc * alpha_zc, axis=2)
        _nu = - tf.reduce_sum(alpha_zc * beta_zc, axis=2) * tf.math.reciprocal_no_nan(_inv_sig)
        _t = - tf.reduce_sum(beta_zc * beta_zc, axis=2) + _nu**2*_inv_sig
        return temp_pi, beta_zc, _inv_sig, _nu, _t
    
    @tf.function
    def _get_pz(self, temp_pi, _inv_sig, beta_zc, log_p_z_c_L):
        # [batch_size, L, n_categories]
        log_p_zc_L = - 0.5 * self.dim_latent * tf.math.log(tf.constant(2 * np.pi, tf.keras.backend.floatx())) + \
            tf.math.log(temp_pi) + \
            tf.where(_inv_sig==0, 
                    - 0.5 * tf.reduce_sum(beta_zc**2, axis=2), 
                    log_p_z_c_L)
        
        # [batch_size, L, 1]
        log_p_z_L = tf.reduce_logsumexp(log_p_zc_L, axis=-1, keepdims=True)
        
        # [1, ]
        log_p_z = tf.reduce_mean(log_p_z_L)
        return log_p_zc_L, log_p_z_L, log_p_z
    
    @tf.function
    def _get_posterior_c(self, log_p_zc_L, log_p_z_L):
        L = tf.shape(log_p_z_L)[1]

        # log_p_c_x     -   predicted probability distribution
        # [batch_size, n_categories]
        log_p_c_x = tf.reduce_logsumexp(
                        log_p_zc_L - log_p_z_L,
                    axis=1) - tf.math.log(tf.cast(L, tf.keras.backend.floatx()))
        return log_p_c_x

    @tf.function
    def _get_inference(self, z, log_p_z_L, temp_pi, _inv_sig, _nu, beta_zc, log_eta0, eta1, eta2):
        batch_size = tf.shape(z)[0]
        L = tf.shape(z)[1]
        dist = tfp.distributions.Normal(
            loc = tf.constant(0.0, tf.keras.backend.floatx()), 
            scale = tf.constant(1.0, tf.keras.backend.floatx()), 
            allow_nan_stats=False)
        
        # [batch_size, L, n_categories, n_clusters]
        _inv_sig = tf.expand_dims(_inv_sig, -1)
        _sig = tf.tile(tf.clip_by_value(tf.math.reciprocal_no_nan(_inv_sig), 1e-12, 1e30), (1,1,1,self.n_states))
        log_eta0 = tf.tile(tf.expand_dims(log_eta0, -1), (1,1,1,self.n_states))
        eta1 = tf.tile(tf.expand_dims(eta1, -1), (1,1,1,self.n_states))
        eta2 = tf.tile(tf.expand_dims(eta2, -1), (1,1,1,self.n_states))
        _nu = tf.tile(tf.expand_dims(_nu, -1), (1,1,1,1))
        A = tf.tile(tf.expand_dims(tf.expand_dims(
            tf.one_hot(self.A, self.n_states, dtype=tf.keras.backend.floatx()), 
            0),0), (batch_size,L,1,1))
        B = tf.tile(tf.expand_dims(tf.expand_dims(
            tf.one_hot(self.B, self.n_states, dtype=tf.keras.backend.floatx()), 
            0),0), (batch_size,L,1,1))
        temp_pi = tf.expand_dims(temp_pi, -1)

        # w_tilde [batch_size, L, n_clusters]
        w_tilde = log_eta0 + tf.math.log(
            tf.clip_by_value(
                (dist.cdf(eta1) - dist.cdf(eta2)) * (_nu * A + (1-_nu) * B)  -
                (dist.prob(eta1) - dist.prob(eta2)) * tf.math.sqrt(_sig) * (A - B), 
                0.0, 1e30)
            )
        w_tilde = - 0.5 * self.dim_latent * tf.math.log(tf.constant(2 * np.pi, tf.keras.backend.floatx())) + \
            tf.math.log(temp_pi) + \
            tf.where(_inv_sig==0, 
                    tf.where(B==1, - 0.5 * tf.expand_dims(tf.reduce_sum(beta_zc**2, axis=2), -1), -np.inf), 
                    w_tilde)
        w_tilde = tf.exp(tf.reduce_logsumexp(w_tilde, 2) - log_p_z_L)

        # tf.debugging.assert_greater_equal(
        #     tf.reduce_sum(w_tilde, -1), tf.ones([batch_size, L], dtype=tf.keras.backend.floatx())*0.99, 
        #     message='Wrong w_tilde', summarize=None, name=None
        # )
        
        # var_w_tilde [batch_size, L, n_clusters]
        var_w_tilde = log_eta0 + tf.math.log(
            tf.clip_by_value(
                (dist.cdf(eta1) -  dist.cdf(eta2)) * ((_sig + _nu**2) * (A+B) + (1-2*_nu) * B)  -
                (dist.prob(eta1) - dist.prob(eta2)) * tf.math.sqrt(_sig) * (_nu *(A+B)-B )*2 -
                (eta1*dist.prob(eta1) - eta2*dist.prob(eta2)) * _sig *(A+B), 
                0.0, 1e30)
            )
        var_w_tilde = - 0.5 * self.dim_latent * tf.math.log(tf.constant(2 * np.pi, tf.keras.backend.floatx())) + \
            tf.math.log(temp_pi) + \
            tf.where(_inv_sig==0, 
                    tf.where(B==1, - 0.5 * tf.expand_dims(tf.reduce_sum(beta_zc**2, axis=2), -1), -np.inf), 
                    var_w_tilde) 
        var_w_tilde = tf.exp(tf.reduce_logsumexp(var_w_tilde, 2) - log_p_z_L) - w_tilde**2  


        w_tilde = tf.reduce_mean(w_tilde, 1)
        var_w_tilde = tf.reduce_mean(var_w_tilde, 1)
        return w_tilde, var_w_tilde

    def get_pz(self, z, eps, pi):
        '''Get \(\\log p(Z_i|Y_i,X_i)\).

        Parameters
        ----------
        z : tf.Tensor
            \([B, L, d]\) The latent variables.

        Returns
        ----------
        temp_pi : tf.Tensor
            \([B, L, K]\) \(\\pi\).
        _inv_sig : tf.Tensor
            \([B, L, K]\) \(\\sigma_{Z_ic_i}^{-1}\).
        _nu : tf.Tensor
            \([B, L, K]\) \(\\nu_{Z_ic_i}\).
        beta_zc : tf.Tensor
            \([B, L, d, K]\) \(\\beta_{Z_ic_i}\).
        log_eta0 : tf.Tensor
            \([B, L, K]\) \(\\log\\eta_{Z_ic_i,0}\).
        eta1 : tf.Tensor
            \([B, L, K]\) \(\\eta_{Z_ic_i,1}\).
        eta2 : tf.Tensor
            \([B, L, K]\) \(\\eta_{Z_ic_i,2}\).
        log_p_zc_L : tf.Tensor
            \([B, L, K]\) \(\\log p(Z_i,c_i|Y_i,X_i)\).
        log_p_z_L : tf.Tensor
            \([B, L]\) \(\\log p(Z_i|Y_i,X_i)\).
        log_p_z : tf.Tensor
            \([B, 1]\) The estimated \(\\log p(Z_i|Y_i,X_i)\). 
        '''        
        temp_pi, beta_zc, _inv_sig, _nu, _t = self._get_normal_params(z, pi)
        temp_pi = tf.clip_by_value(temp_pi, eps, 1.0)

        log_eta0 = 0.5 * (tf.math.log(tf.constant(2 * np.pi, tf.keras.backend.floatx())) - \
                    tf.math.log(tf.clip_by_value(_inv_sig, 1e-12, 1e30)) + _t)
        eta1 = (1-_nu) * tf.math.sqrt(tf.clip_by_value(_inv_sig, 1e-12, 1e30))
        eta2 = -_nu * tf.math.sqrt(tf.clip_by_value(_inv_sig, 1e-12, 1e30))

        log_p_z_c_L =  log_eta0 + tf.math.log(tf.clip_by_value(
            self.cdf_layer(eta1) - self.cdf_layer(eta2),
            eps, 1e30))
        
        log_p_zc_L, log_p_z_L, log_p_z = self._get_pz(temp_pi, _inv_sig, beta_zc, log_p_z_c_L)
        return temp_pi, _inv_sig, _nu, beta_zc, log_eta0, eta1, eta2, log_p_zc_L, log_p_z_L, log_p_z

    def get_posterior_c(self, z):
        '''Get \(p(c_i|Y_i,X_i)\).

        Parameters
        ----------
        z : tf.Tensor
            \([B, L, d]\) The latent variables.

        Returns
        ----------
        p_c_x : np.array
            \([B, K]\) \(p(c_i|Y_i,X_i)\).
        '''  
        _,_,_,_,_,_,_, log_p_zc_L, log_p_z_L, _ = self.get_pz(z)
        log_p_c_x = self._get_posterior_c(log_p_zc_L, log_p_z_L)
        p_c_x = tf.exp(log_p_c_x).numpy()
        return p_c_x

    def call(self, z, pi=None, inference=False):
        '''Get posterior estimations.

        Parameters
        ----------
        z : tf.Tensor
            \([B, L, d]\) The latent variables.
        inference : boolean
            Whether in training or inference mode.

        When `inference=False`:

        Returns
        ----------
        log_p_z_L : tf.Tensor
            \([B, 1]\) The estimated \(\\log p(Z_i|Y_i,X_i)\).

        When `inference=True`:

        Returns
        ----------
        res : dict
            The dict of posterior estimations - \(p(c_i|Y_i,X_i)\), \(c\), \(E(\\tilde{w}_i|Y_i,X_i)\), \(Var(\\tilde{w}_i|Y_i,X_i)\), \(D_{JS}\).
        '''                 
        eps = 1e-16 if not inference else 0.
        temp_pi, _inv_sig, _nu, beta_zc, log_eta0, eta1, eta2, log_p_zc_L, log_p_z_L, log_p_z = self.get_pz(z, eps, pi)

        if not inference:
            return log_p_z
        else:
            log_p_c_x = self._get_posterior_c(log_p_zc_L, log_p_z_L)
            w_tilde, var_w_tilde = self._get_inference(z, log_p_z_L, temp_pi, _inv_sig, _nu, beta_zc, log_eta0, eta1, eta2)
            
            res = {}
            res['p_c_x'] = tf.exp(log_p_c_x).numpy()
            res['w_tilde'] = w_tilde.numpy()
            res['var_w_tilde'] = var_w_tilde.numpy()
            return res
            
            
class VariationalAutoEncoder(tf.keras.Model):
    """
    Combines the encoder, decoder and LatentSpace into an end-to-end model for training and inference.
    """
    def __init__(self, dim_origin, dimensions, dim_latent,
                 data_type = 'UMI', has_cov=False,
                 name = 'autoencoder', **kwargs):
        '''
        Parameters
        ----------
        dim_origin : int
            The output dimension of the decoder.        
        dimensions : np.array
            The dimensions of hidden layers of the encoder.
        dim_latent : int
            The latent dimension.
        data_type : str, optional
            `'UMI'`, `'non-UMI'`, or `'Gaussian'`.
        has_cov : boolean
            Whether has covariates or not.
        gamma : float, optional
            The weights of the MMD loss
        name : str, optional
            The name of the layer.
        **kwargs : 
            Extra keyword arguments.
        '''
        super(VariationalAutoEncoder, self).__init__(name = name, **kwargs)
        self.data_type = data_type
        self.dim_origin = dim_origin
        self.dim_latent = dim_latent
        self.encoder = Encoder(dimensions, dim_latent)
        self.decoder = Decoder(dimensions[::-1], dim_origin, data_type, data_type)        
        self.has_cov = has_cov
        
    def init_latent_space(self, n_clusters, mu, log_pi=None):
        '''Initialze the latent space.

        Parameters
        ----------
        n_clusters : int
            The number of vertices in the latent space.
        mu : np.array
            \([d, k]\) The position matrix.
        log_pi : np.array, optional
            \([1, K]\) \(\\log\\pi\).
        '''
        self.n_states = n_clusters
        self.latent_space = LatentSpace(self.n_states, self.dim_latent)
        self.latent_space.initialize(mu, log_pi)
        self.pilayer = None

    def create_pilayer(self):
        self.pilayer = Dense(self.latent_space.n_categories, name = 'pi_layer')

    def call(self, x_normalized, c_score, x = None, scale_factor = 1,
             pre_train = False, L=1, alpha=0.0, gamma = 0.0, phi = 1.0, conditions = None, pi_cov = None):
        '''Feed forward through encoder, LatentSpace layer and decoder.

        Parameters
        ----------
        x_normalized : np.array
            \([B, G]\) The preprocessed data.
        c_score : np.array
            \([B, s]\) The covariates \(X_i\), only used when `has_cov=True`.
        x : np.array, optional
            \([B, G]\) The original count data \(Y_i\), only used when data_type is not `'Gaussian'`.
        scale_factor : np.array, optional
            \([B, ]\) The scale factors, only used when data_type is not `'Gaussian'`.
        pre_train : boolean, optional
            Whether in the pre-training phare or not.
        L : int, optional
            The number of MC samples.
        alpha : float, optional
            The penalty parameter for covariates adjustment.
        gamma : float, optional
            The weight of mmd loss
        phi : float, optional
            The weight of Jacob norm of the encoder.
        conditions: str or list, optional
            The conditions of different cells from the selected batch

        Returns
        ----------
        losses : float
            the loss.
        '''

        if not pre_train and self.latent_space is None:
            raise ReferenceError('Have not initialized the latent space.')
                    
        if self.has_cov:
            x_normalized = tf.concat([x_normalized, c_score], -1)
        else:
            x_normalized
        _, z_log_var, z = self.encoder(x_normalized, L)

        if gamma == 0:
            mmd_loss = 0.0
        else:
            mmd_loss = self._get_total_mmd_loss(conditions,z,gamma)

        z_in = tf.concat([z, tf.tile(tf.expand_dims(c_score,1), (1,L,1))], -1) if self.has_cov else z
        
        x = tf.tile(tf.expand_dims(x, 1), (1,L,1))
        reconstruction_z_loss = self._get_reconstruction_loss(x, z_in, scale_factor, L)
        
        if self.has_cov and alpha>0.0:
            zero_in = tf.concat([tf.zeros([z.shape[0],1,z.shape[2]], dtype=tf.keras.backend.floatx()), 
                                tf.tile(tf.expand_dims(c_score,1), (1,1,1))], -1)
            reconstruction_zero_loss = self._get_reconstruction_loss(x, zero_in, scale_factor, 1)
            reconstruction_z_loss = (1-alpha)*reconstruction_z_loss + alpha*reconstruction_zero_loss

        self.add_loss(reconstruction_z_loss)
        J_norm = self._get_Jacob(x_normalized, L)
        self.add_loss((phi * J_norm))
        # gamma weight has been used when call _mmd_loss function.
        self.add_loss(mmd_loss)

        if not pre_train:
            pi = self.pilayer(pi_cov) if self.pilayer is not None else None
            log_p_z = self.latent_space(z, pi, inference=False)

            # - E_q[log p(z)]
            self.add_loss(- log_p_z)

            # - Eq[log q(z|x)]
            E_qzx = - tf.reduce_mean(
                            0.5 * self.dim_latent *
                            (tf.math.log(tf.constant(2 * np.pi, tf.keras.backend.floatx())) + 1.0) +
                            0.5 * tf.reduce_sum(z_log_var, axis=-1)
                            )
            self.add_loss(E_qzx)
        return self.losses
    
    @tf.function
    def _get_reconstruction_loss(self, x, z_in, scale_factor, L):
        if self.data_type=='Gaussian':
            # Gaussian Log-Likelihood Loss function
            nu_z, tau = self.decoder(z_in)
            neg_E_Gaus = 0.5 * tf.math.log(tf.clip_by_value(tau, 1e-12, 1e30)) + 0.5 * tf.math.square(x - nu_z) / tau
            neg_E_Gaus = tf.reduce_mean(tf.reduce_sum(neg_E_Gaus, axis=-1))

            return neg_E_Gaus
        else:
            if self.data_type == 'UMI':
                x_hat, r = self.decoder(z_in)
            else:
                x_hat, r, phi = self.decoder(z_in)

            x_hat = x_hat*tf.expand_dims(scale_factor, -1)

            # Negative Log-Likelihood Loss function

            # Ref for NB & ZINB loss functions:
            # https://github.com/gokceneraslan/neuralnet_countmodels/blob/master/Count%20models%20with%20neuralnets.ipynb
            # Negative Binomial loss

            neg_E_nb = tf.math.lgamma(r) + tf.math.lgamma(x+1.0) \
                        - tf.math.lgamma(x+r) + \
                        (r+x) * tf.math.log(1.0 + (x_hat/r)) + \
                        x * (tf.math.log(r) - tf.math.log(tf.clip_by_value(x_hat, 1e-12, 1e30)))
            
            if self.data_type == 'non-UMI':
                # Zero-Inflated Negative Binomial loss
                nb_case = neg_E_nb - tf.math.log(tf.clip_by_value(1.0-phi, 1e-12, 1e30))
                zero_case = - tf.math.log(tf.clip_by_value(
                    phi + (1.0-phi) * tf.pow(r * tf.math.reciprocal_no_nan(r + x_hat), r),
                    1e-12, 1e30))
                neg_E_nb = tf.where(tf.less(x, 1e-8), zero_case, nb_case)

            neg_E_nb = tf.reduce_mean(tf.reduce_sum(neg_E_nb, axis=-1))
            return neg_E_nb

    def _get_total_mmd_loss(self,conditions,z,gamma):
        mmd_loss = 0.0
        conditions = tf.cast(conditions,tf.int32)
        n_group = conditions.shape[1]

        for i in range(n_group):
            sub_conditions = conditions[:, i]
            # 0 means not participant in mmd
            z_cond = z[sub_conditions != 0]
            sub_conditions = sub_conditions[sub_conditions != 0]
            n_sub_group = tf.unique(sub_conditions)[0].shape[0]
            real_labels = K.reshape(sub_conditions, (-1,)).numpy()
            unique_set = list(set(real_labels))
            reindex_dict = dict(zip(unique_set, range(n_sub_group)))
            real_labels = [reindex_dict[x] for x in real_labels]
            real_labels = tf.convert_to_tensor(real_labels,dtype=tf.int32)

            if (n_sub_group == 1) | (n_sub_group == 0):
                _loss = 0
            else:
                _loss = self._mmd_loss(real_labels=real_labels, y_pred=z_cond, gamma=gamma,
                                       n_conditions=n_sub_group,
                                       kernel_method='multi-scale-rbf',
                                       computation_method="general")
            mmd_loss = mmd_loss + _loss
        return mmd_loss

    # each loop the inputed shape is changed. Can not use @tf.function
    # tf graph requires static shape and tensor dtype
    def _mmd_loss(self, real_labels, y_pred, gamma, n_conditions, kernel_method='multi-scale-rbf',
                  computation_method="general"):
        conditions_mmd = tf.dynamic_partition(y_pred, real_labels, num_partitions=n_conditions)
        loss = 0.0
        if computation_method.isdigit():
            boundary = int(computation_method)
            ## every pair of groups will calculate a distance
            for i in range(boundary):
                for j in range(boundary, n_conditions):
                    loss += _nan2zero(compute_mmd(conditions_mmd[i], conditions_mmd[j], kernel_method))
        else:
            for i in range(len(conditions_mmd)):
                for j in range(i):
                    loss += _nan2zero(compute_mmd(conditions_mmd[i], conditions_mmd[j], kernel_method))

        # print("The loss is ", loss)
        return gamma * loss

    @tf.function
    def _get_Jacob(self, x, L):
        with tf.GradientTape() as g:
            g.watch(x)
            z_mean, z_log_var, z = self.encoder(x, L)
            # y_mean, y_log_var = self.decoder(z)
        ## just jacobian will cause shape (batch,16,batch,64) matrix
        J = g.batch_jacobian(z, x)
        J_norm = tf.norm(J)
        # tf.print(J_norm)

        return J_norm
    
    def get_z(self, x_normalized, c_score):    
        '''Get \(q(Z_i|Y_i,X_i)\).

        Parameters
        ----------
        x_normalized : int
            \([B, G]\) The preprocessed data.
        c_score : np.array
            \([B, s]\) The covariates \(X_i\), only used when `has_cov=True`.

        Returns
        ----------
        z_mean : np.array
            \([B, d]\) The latent mean.
        '''    
        x_normalized = x_normalized if (not self.has_cov or c_score is None) else tf.concat([x_normalized, c_score], -1)
        z_mean, _, _ = self.encoder(x_normalized, 1, False)
        return z_mean.numpy()

    def get_pc_x(self, test_dataset):
        '''Get \(p(c_i|Y_i,X_i)\).

        Parameters
        ----------
        test_dataset : tf.Dataset
            the dataset object.

        Returns
        ----------
        pi_norm : np.array
            \([1, K]\) The estimated \(\\pi\).
        p_c_x : np.array
            \([N, ]\) The estimated \(p(c_i|Y_i,X_i)\).
        '''    
        if self.latent_space is None:
            raise ReferenceError('Have not initialized the latent space.')
        
        pi_norm = tf.nn.softmax(self.latent_space.pi).numpy()
        p_c_x = []
        for x,c_score in test_dataset:
            x = tf.concat([x, c_score], -1) if self.has_cov else x
            _, _, z = self.encoder(x, 1, False)
            _p_c_x = self.latent_space.get_posterior_c(z)            
            p_c_x.append(_p_c_x)
        p_c_x = np.concatenate(p_c_x)         
        return pi_norm, p_c_x

    def inference(self, test_dataset, L=1):
        '''Get \(p(c_i|Y_i,X_i)\).

        Parameters
        ----------
        test_dataset : tf.Dataset
            The dataset object.
        L : int
            The number of MC samples.

        Returns
        ----------
        pi_norm  : np.array
            \([1, K]\) The estimated \(\\pi\).
        mu : np.array
            \([d, k]\) The estimated \(\\mu\).
        p_c_x : np.array
            \([N, ]\) The estimated \(p(c_i|Y_i,X_i)\).
        w_tilde : np.array
            \([N, k]\) The estimated \(E(\\tilde{w}_i|Y_i,X_i)\).
        var_w_tilde  : np.array 
            \([N, k]\) The estimated \(Var(\\tilde{w}_i|Y_i,X_i)\).
        z_mean : np.array
            \([N, d]\) The estimated latent mean.
        '''   
        if self.latent_space is None:
            raise ReferenceError('Have not initialized the latent space.')
        
        print('Computing posterior estimations over mini-batches.')
        progbar = Progbar(test_dataset.cardinality().numpy())
        pi_norm = tf.nn.softmax(self.latent_space.pi).numpy()
        mu = self.latent_space.mu.numpy()
        z_mean = []
        p_c_x = []
        w_tilde = []
        var_w_tilde = []
        for step, (x,c_score, _, _) in enumerate(test_dataset):
            x = tf.concat([x, c_score], -1) if self.has_cov else x
            _z_mean, _, z = self.encoder(x, L, False)
            res = self.latent_space(z, inference=True)
            
            z_mean.append(_z_mean.numpy())
            p_c_x.append(res['p_c_x'])            
            w_tilde.append(res['w_tilde'])
            var_w_tilde.append(res['var_w_tilde'])
            progbar.update(step+1)

        z_mean = np.concatenate(z_mean)
        p_c_x = np.concatenate(p_c_x)
        w_tilde = np.concatenate(w_tilde)
        w_tilde /= np.sum(w_tilde, axis=1, keepdims=True)
        var_w_tilde = np.concatenate(var_w_tilde)
        return pi_norm, mu, p_c_x, w_tilde, var_w_tilde, z_mean
