import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, BatchNormalization
import tensorflow_probability as tfp

 
class cdf_layer(Layer):
    def __init__(self):
        super(cdf_layer, self).__init__()
        
    @tf.function
    def call(self, x):
        return self.func(x)
        
    @tf.custom_gradient
    def func(self, x):
        dist = tfp.distributions.Normal(loc=0., scale=1., allow_nan_stats=False)
        f = dist.cdf(x)
        def grad(dy):
            gradient = dist.prob(x)
            return dy * gradient
        return f, grad
    

class Sampling(Layer):
    """
    Sampling latent variable z by (z_mean, z_log_var).    
    Used in Encoder.
    """
    @tf.function
    def call(self, z_mean, z_log_var):
        epsilon = tf.random.normal(shape = tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        z = tf.clip_by_value(z, -1e6, 1e6)
        return z

class Encoder(Layer):
    '''
    Encoder, model q(z|x).
    '''
    def __init__(self, dimensions, dim_latent, name='encoder', **kwargs):
        '''
        Input:
          dimensions  - list of dimensions of layers in dense layers of
                       encoder expcept the latent layer.
          dim_latent  - dimension of latent layer.
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
        '''
        Input :
            x           - input                     [batch_size, dim_origin]
        Output:
            z_mean      - mean of p(z|x)            [batch_size, dim_latent]
            z_log_var   - log of variance of p(z|x) [batch_size, dim_latent]
            z           - sampled z                 [batch_size, L, dim_latent]
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
    Decoder, model p(x|z).
    '''
    def __init__(self, dimensions, dim_origin, data_type = 'UMI', 
                name = 'decoder', **kwargs):
        '''
        Input:
            dimensions      - list of dimensions of layers in dense layers of
                                decoder expcept the output layer.
            dim_origin      - dimension of output layer.
            data_type       - 'UMI', 'non-UMI' and 'Gaussian'
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
            self.log_tau = tf.Variable(tf.zeros([1, dim_origin]),
                                 constraint = lambda t: tf.clip_by_value(t,-30.,6.),
                                 name = "log_tau")
        else:
            self.log_lambda_z = Dense(dim_origin, name = 'log_lambda_z')

            # dispersion parameter
            self.log_r = tf.Variable(tf.zeros([1, dim_origin]),
                                     constraint = lambda t: tf.clip_by_value(t,-30.,6.),
                                     name = "log_r")
            
            if self.data_type == 'non-UMI':
                self.phi = Dense(dim_origin, activation = 'sigmoid', name = "phi")
          
    @tf.function  
    def call(self, z, is_training=True):
        '''
        Input :
            z           - latent variables  [batch_size, L, dim_origin]
        Output:
            nu_z        - x_hat if Gaussian
            tau         - common variance if Gaussian
            lambda_z    - x_hat             [batch_size, L, dim_origin]
            r           - dispersion parameter
                                            [1,          L, dim_origin]
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
    It contains parameters related to model assumptions.
    '''
    def __init__(self, n_clusters, dim_latent, M = 50, name = 'LatentSpace', **kwargs):
        '''
        Input:
          dim_latent   - dimension of latent layer.
          dim_origin - dimension of output layer.
          M            - number of samples for w.
        '''
        super(LatentSpace, self).__init__(name=name, **kwargs)
        self.dim_latent = dim_latent
        self.n_clusters = n_clusters
        self.n_states = int(n_clusters*(n_clusters+1)/2)

        # nonzero indexes
        # A = [0,0,...,0  , 1,1,...,1,   ...]
        # B = [0,1,...,k-1, 1,2,...,k-1, ...]
        self.A, self.B = np.nonzero(np.triu(np.ones(n_clusters)))
        self.A = tf.convert_to_tensor(self.A, tf.int32)
        self.B = tf.convert_to_tensor(self.B, tf.int32)
        self.clusters_ind = tf.boolean_mask(
            tf.range(0,self.n_states,1), self.A==self.B)

        # Uniform random variable
        self.M = tf.convert_to_tensor(M, tf.int32)
        self.w =  tf.convert_to_tensor(
            np.resize(np.arange(0, M)/M, (1, M)), tf.float32)

        # [pi_1, ... , pi_K] in R^(n_states)
        self.pi = tf.Variable(tf.ones([1, self.n_states]) / self.n_states,
                                name = 'pi')
        
        # [mu_1, ... , mu_K] in R^(dim_latent * n_clusters)
        self.mu = tf.Variable(tf.random.uniform([self.dim_latent, self.n_clusters],
                                                minval = -1, maxval = 1),
                                name = 'mu')
        self.cdf_layer = cdf_layer()       
        
    def initialize(self, mu, pi):
        # Initialize parameters of the latent space
        if mu is not None:
            self.mu.assign(mu)
        if pi is not None:
            self.pi.assign(pi)

    def normalize(self):
        self.pi = tf.nn.softmax(self.pi)

    @tf.function
    def _get_normal_params(self, z):
        batch_size = tf.shape(z)[0]
        L = tf.shape(z)[1]
        
        # [batch_size, L, n_states]
        temp_pi = tf.tile(
            tf.expand_dims(tf.nn.softmax(self.pi), 1),
            (batch_size,L,1))
                        
        # [batch_size, L, d, n_states]
        a1 = tf.expand_dims(tf.expand_dims(
            tf.gather(self.mu, self.B, axis=1) - tf.gather(self.mu, self.A, axis=1), 0), 0)
        a2 = tf.expand_dims(z,-1) - \
            tf.expand_dims(tf.expand_dims(
            tf.gather(self.mu, self.B, axis=1), 0), 0)
            
        # [batch_size, L, n_states]
        _inv_sig = tf.reduce_sum(a1 * a1, axis=2)
        _mu = - tf.reduce_sum(a1 * a2, axis=2)*tf.math.reciprocal_no_nan(_inv_sig)
        _t = - tf.reduce_sum(a2 * a2, axis=2) + _mu**2*_inv_sig
        return temp_pi, a2, _inv_sig, _mu, _t
    
    @tf.function
    def _get_pz(self, temp_pi, _inv_sig, a2, log_p_z_c_L):
        # [batch_size, L, n_states]
        log_p_zc_L = - 0.5 * self.dim_latent * tf.math.log(2 * np.pi) + \
            tf.math.log(temp_pi+1e-12) + \
            tf.where(_inv_sig==0, 
                    - 0.5 * tf.reduce_sum(a2**2, axis=2), 
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
        # [batch_size, n_states]
        log_p_c_x = tf.reduce_logsumexp(
                        log_p_zc_L - log_p_z_L,
                    axis=1) - tf.math.log(tf.cast(L, tf.float32))
        return log_p_c_x

    @tf.function
    def _get_inference(self, z, log_p_zc_L, log_p_z_L):
        batch_size = tf.shape(z)[0]
        L = tf.shape(z)[1]
        
        # [batch_size, L, dim_latent, n_states, M]
        temp_Z = tf.tile(tf.expand_dims(tf.expand_dims(z,-1), -1),
                    (1,1,1,self.n_states,self.M))
        temp_mu = tf.tile(
                        tf.expand_dims(tf.expand_dims(
                            tf.gather(self.mu, self.A, axis=1),0),-1),
                        (batch_size,1,1,self.M)) * self.w + \
                tf.tile(
                        tf.expand_dims(tf.expand_dims(
                            tf.gather(self.mu, self.B, axis=1),0),-1),
                        (batch_size,1,1,self.M)) * (1-self.w)
        temp_mu = tf.tile(tf.expand_dims(temp_mu, 1), (1,L,1,1,1))
            
        # [batch_size, L, n_states, M]
        temp_pi = tf.tile(tf.expand_dims(
                        tf.expand_dims(tf.nn.softmax(self.pi), -1), 1),
                        (batch_size,L,1,self.M))
        
        # [batch_size, L, n_states, M]
        log_p_zc_w = - 0.5 * self.dim_latent * tf.math.log(2 * np.pi) + \
                        tf.math.log(temp_pi+1e-12) - \
                        tf.reduce_sum(tf.math.square(temp_Z - temp_mu), 2)/2  # this omits a term -logM for avoiding numerical issue
                        
        # [batch_size, L]
        log_p_z_L = tf.expand_dims(tf.reduce_logsumexp(log_p_zc_w, axis=[2,3]), -1) # this omits a term -logM for avoiding numerical issue
                    
        # [batch_size, n_states, M]
        p_wc_x = tf.exp(tf.reduce_logsumexp(
                    log_p_zc_w -
                    tf.expand_dims(log_p_z_L, -1),
                    1) - tf.math.log(tf.cast(L, tf.float32)))
                    
        # [batch_size, n_states, n_clusters, M]
        _w = tf.tile(tf.expand_dims(
                tf.tile(tf.expand_dims(
                    tf.one_hot(self.A, self.n_clusters),
                    -1), (1,1,self.M)) * self.w +
                tf.tile(tf.expand_dims(
                    tf.one_hot(self.B, self.n_clusters),
                    -1), (1,1,self.M)) * (1-self.w), 0), (batch_size,1,1,1))
                    
        # [batch_size, n_clusters]
        w_tilde = tf.reduce_sum(
            _w  * \
            tf.tile(tf.expand_dims(p_wc_x, 2), (1,1,self.n_clusters,1)),
            (1,3))
            
        var_w_tilde = tf.reduce_sum(
            tf.math.square(_w)  *
            tf.tile(tf.expand_dims(p_wc_x, 2), (1,1,self.n_clusters,1)),
            (1,3)) - tf.square(w_tilde)    
        
        _wtilde = tf.expand_dims(tf.expand_dims(w_tilde, 1), -1)
        D_JS = tf.reduce_sum(
            tf.math.sqrt(
                0.5 * tf.reduce_mean(
                    _w * tf.math.log(
                        _w/(0.5 * (_w + _wtilde) + 1e-12) + 1e-12) +
                    _wtilde * tf.math.log(
                        _wtilde/(0.5 * (_w + _wtilde) + 1e-12) + 1e-12),
                    axis=2)) * \
            p_wc_x, (1,2)) / tf.cast(self.M, tf.float32)
        return w_tilde, var_w_tilde, D_JS
    
    def get_pz(self, z):
        temp_pi, a2, _inv_sig, _mu, _t = self._get_normal_params(z)
        
        log_p_z_c_L =  0.5 * (tf.math.log(2 * np.pi) - \
                        tf.math.log(_inv_sig+1e-12) + \
                        _t
                        ) + \
                        tf.math.log(self.cdf_layer((1-_mu)*tf.math.sqrt(_inv_sig+1e-12)) - 
                                    self.cdf_layer(-_mu*tf.math.sqrt(_inv_sig+1e-12)) + 1e-12)
        
        log_p_zc_L, log_p_z_L, log_p_z = self._get_pz(temp_pi, _inv_sig, a2, log_p_z_c_L)
        return log_p_zc_L, log_p_z_L, log_p_z

    def call(self, z, inference=False):
        '''
        Input :
                z       - latent variables outputed by the encoder
                          [batch_size, L, dim_latent]
        Output:       
            inference = True:     
                log_p_z - MC samples for log p(z)=log sum_{c}p(z|c)*p(c)
                          [batch_size, ]
            inference = False:
                res     - results contains estimations for 
                            p(c|x), E(w|x), Var(w|x), E(w|x,c), Var(w|x,c), 
                            c, E(w_tilde), Var(w_tilde), D_JS
        '''               
        log_p_zc_L, log_p_z_L, log_p_z = self.get_pz(z)

        if not inference:
            return log_p_z
        else:
            log_p_c_x = self._get_posterior_c(log_p_zc_L, log_p_z_L)
            w_tilde, var_w_tilde, D_JS = self._get_inference(z, log_p_zc_L, log_p_z_L)
            
            res = {}
            res['p_c_x'] = tf.exp(log_p_c_x).numpy()
            res['w_tilde'] = w_tilde.numpy()
            res['var_w_tilde'] = var_w_tilde.numpy()
            res['D_JS'] = D_JS.numpy()
            return res

    def get_posterior_c(self, z):
        log_p_zc_L, log_p_z_L, _ = self.get_pz(z)
        log_p_c_x = self._get_posterior_c(log_p_zc_L, log_p_z_L)
        p_c_x = tf.exp(log_p_c_x).numpy()
        return p_c_x

    def get_proj_z(self, c):
        '''
        Args:
            c - Numpy array of indexes [1,*]
        '''
        proj_c = np.tile(c, (self.M,1)).T.flatten()
        proj_z_M = tf.transpose(
                        tf.gather(self.mu, tf.gather(self.A, proj_c), axis=1) * 
                        tf.tile(self.w, (1,len(c))) + 
                        tf.gather(self.mu, tf.gather(self.B, proj_c), axis=1) * 
                        (1-tf.tile(self.w, (1,len(c))))
                    )
        return proj_c, proj_z_M.numpy()
            
            
class VariationalAutoEncoder(tf.keras.Model):
    """
    Combines the encoder, decoder and LatentSpace into an end-to-end model for training.
    """
    def __init__(self, dim_origin, dimensions, dim_latent,
                 data_type = 'UMI', has_cov=False,
                 name = 'autoencoder', **kwargs):
        '''
        Args:
            n_clusters      -   Number of clusters.
            dim_origin      -   Dim of input.
            dimensions      -   List of dimensions of layers of the encoder. Assume
                               symmetric network sturcture of encoder and decoder.
            dim_latent      -   Dimension of latent layer.
            data_type       -   Type of count data.
                              'UMI' for negative binomial loss;
                              'non-UMI' for zero-inflated negative binomial loss.
            has_cov         - has covariate or not
        '''
        super(VariationalAutoEncoder, self).__init__(name = name, **kwargs)
        self.data_type = data_type
        self.dim_origin = dim_origin
        self.dim_latent = dim_latent
        self.encoder = Encoder(dimensions, dim_latent)
        self.decoder = Decoder(dimensions[::-1], dim_origin, data_type, data_type)        
        self.has_cov = has_cov
        
    def init_latent_space(self, n_clusters, mu, pi=None):
        self.n_clusters = n_clusters
        self.latent_space = LatentSpace(self.n_clusters, self.dim_latent)
        self.latent_space.initialize(mu, pi)

    def call(self, x_normalized, c_score, x = None, scale_factor = 1,
             pre_train = False, L=1, alpha=0.0):
        # Feed forward through encoder, LatentSpace layer and decoder.
        if not pre_train and self.latent_space is None:
            raise ReferenceError('Have not initialized the latent space.')
                    
        x_normalized = tf.concat([x_normalized, c_score], -1) if self.has_cov else x_normalized
        _, z_log_var, z = self.encoder(x_normalized, L)
                
        z_in = tf.concat([z, tf.tile(tf.expand_dims(c_score,1), (1,L,1))], -1) if self.has_cov else z
        
        x = tf.tile(tf.expand_dims(x, 1), (1,L,1))
        reconstruction_z_loss = self.get_reconstruction_loss(x, z_in, scale_factor, L)
        
        if self.has_cov and alpha>0.0:
            zero_in = tf.concat([tf.zeros([z.shape[0],1,z.shape[2]]), tf.tile(tf.expand_dims(c_score,1), (1,1,1))], -1)
            reconstruction_zero_loss = self.get_reconstruction_loss(x, zero_in, scale_factor, 1)
            reconstruction_z_loss = (1-alpha)*reconstruction_z_loss + alpha*reconstruction_zero_loss
        
        self.add_loss(reconstruction_z_loss)

        if not pre_train:        
            log_p_z = self.latent_space(z, inference=False)

            # - E_q[log p(z)]
            self.add_loss(- log_p_z)

            # - Eq[log q(z|x)]
            E_qzx = - tf.reduce_mean(
                            0.5 * self.dim_latent *
                            (tf.math.log(2 * np.pi) + 1) +
                            0.5 * tf.reduce_sum(z_log_var, axis=-1)
                            )
            self.add_loss(E_qzx)
        return self.losses
    
    @tf.function
    def get_reconstruction_loss(self, x, z_in, scale_factor, L):
        if self.data_type=='Gaussian':
            # Gaussian Log-Likelihood Loss function
            nu_z, tau = self.decoder(z_in)
            neg_E_Gaus = 0.5 * tf.math.log(tau + 1e-12) + 0.5 * tf.math.square(x - nu_z) / tau
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
                        x * (tf.math.log(r) - tf.math.log(x_hat+1e-12))
            
            if self.data_type == 'non-UMI':
                # Zero-Inflated Negative Binomial loss
                nb_case = neg_E_nb - tf.math.log(1.0-phi+1e-12)
                zero_case = - tf.math.log(phi + (1.0-phi) *
                                     tf.pow(r/(r + x_hat + 1e-12), r) + 1e-12)
                neg_E_nb = tf.where(tf.less(x, 1e-8), zero_case, nb_case)

            neg_E_nb =  tf.reduce_mean(tf.reduce_sum(neg_E_nb, axis=-1))
            
            return neg_E_nb
    
    def get_z(self, x_normalized, c_score):        
        x_normalized = x_normalized if (not self.has_cov or c_score is None) else tf.concat([x_normalized, c_score], -1)
        z_mean, _, _ = self.encoder(x_normalized, 1, False)
        return z_mean.numpy()
    
    def get_proj_z(self, c):
        '''
        Args:
            c - List of indexes of edges
        '''
        return self.latent_space.get_proj_z(c)

    def get_pc_x(self, test_dataset):
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
        if self.latent_space is None:
            raise ReferenceError('Have not initialized the latent space.')
            
        pi_norm = tf.nn.softmax(self.latent_space.pi).numpy()
        mu = self.latent_space.mu.numpy()
        z_mean = []
        p_c_x = []
        w_tilde = []
        var_w_tilde = []
        D_JS = []
        for x,c_score in test_dataset:
            x = tf.concat([x, c_score], -1) if self.has_cov else x
            _z_mean, _, z = self.encoder(x, L, False)
            res = self.latent_space(z, inference=True)
            
            z_mean.append(_z_mean.numpy())
            p_c_x.append(res['p_c_x'])            
            w_tilde.append(res['w_tilde'])
            var_w_tilde.append(res['var_w_tilde'])
            D_JS.append(res['D_JS'])
        
        D_JS = np.concatenate(D_JS)
        z_mean = np.concatenate(z_mean)
        p_c_x = np.concatenate(p_c_x)
        w_tilde = np.concatenate(w_tilde)
        var_w_tilde = np.concatenate(var_w_tilde)
        return pi_norm, mu, p_c_x, w_tilde, var_w_tilde, D_JS, z_mean
