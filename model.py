import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, BatchNormalization
import numpy as np
import math


class Sampling(Layer):
    """
    Sampling latent variable z by (z_mean, z_log_var).    
    Used in Encoder.
    """
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


class GMM(Layer):
    '''
    GMM layer.
    It contains parameters related to model assumptions of GMM.
    '''
    def __init__(self, n_clusters, dim_latent, M = 100, name = 'GMM', **kwargs):
        '''
        Input:
          dim_latent   - dimension of latent layer.
          dim_origin - dimension of output layer.
          M            - number of samples for w.
        '''
        super(GMM, self).__init__(name=name, **kwargs)
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

    def initialize(self, mu, pi):
        # Initialize parameters of GMM
        if mu is not None:
            self.mu.assign(mu)
        if pi is not None:
            self.pi.assign(pi)

    def normalize(self):
        self.pi = tf.nn.softmax(self.pi)

    def call(self, z, inference=False):
        '''
        Input :
                z       - latent variables outputed by the encoder
                          [batch_size, L, dim_latent]
        Output:
                p_z     - MC samples for p(z)=sum_{c,w}(p_zc_w)/M
                          [batch_size, ]
                p_zc_w  - MC samples for p(z,c|w)=p(z|w,c)p(c) (L=1)
                          [batch_size, n_states, M]
        '''
        batch_size = tf.shape(z)[0]
        L = tf.shape(z)[1]
        
        res = {}
        
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
        
        # [batch_size, L, M]
        log_p_zc_w = - 0.5 * self.dim_latent * tf.math.log(2 * math.pi) + \
                        tf.math.log(temp_pi+1e-30) - \
                        tf.reduce_sum(tf.math.square(temp_Z - temp_mu), 2)/2  # this omits a term -logM for avoiding numerical issue
        
        # [batch_size, L]
        log_p_z_L = tf.reduce_logsumexp(log_p_zc_w, axis=[2,3]) # this omits a term -logM for avoiding numerical issue
                        
        # [1, ]
        log_p_z = tf.reduce_mean(log_p_z_L) - tf.math.log(tf.cast(self.M, tf.float32))
                            
        if not inference:
            return log_p_z
        else:
            # log_p_c_x     -   predicted probability distribution
            # [batch_size, n_states]
            log_p_c_x = tf.reduce_logsumexp(
                            tf.reduce_logsumexp(
                                log_p_zc_w, axis=-1) -
                            tf.expand_dims(log_p_z_L, -1),
                        axis=1) - tf.math.log(tf.cast(L, tf.float32))
            res['p_c_x'] = tf.exp(log_p_c_x).numpy()
            
            # c         -   predicted clusters
            c = tf.math.argmax(log_p_c_x, axis=-1)
            c = tf.cast(c, 'int32')

            # w         -   E(w|x)
            # [batch_size, M]
            p_w_x = tf.reduce_mean(tf.exp(
                        tf.reduce_logsumexp(log_p_zc_w, axis=2) -
                        tf.expand_dims(log_p_z_L, -1)
                        ), axis=1) # this omits a term M for avoiding numerical issue
            
            # [batch_size, ]
            w =  tf.reduce_sum(
                    tf.tile(self.w, (batch_size,1)) * p_w_x, axis=-1
                    )
            res['w_x'] = w.numpy()
            
            # var_w     -   Var(w|x)
            var_w =  tf.reduce_sum(
                        tf.square(
                            tf.tile(self.w, (batch_size,1))
                            ) * p_w_x, axis=-1
                        ) - tf.square(w)
            res['var_w_x'] = var_w.numpy()
            
            # w|c       -   E(w|x,c)
            # [batch_size, L, M]
            map_log_p_zc_w = tf.gather_nd(log_p_zc_w,
                                tf.reshape(
                                tf.stack([tf.repeat(tf.range(batch_size), L),
                                        tf.tile(tf.range(L), [batch_size]),
                                        tf.repeat(c, L)], 1), [batch_size, L, -1])) # this omits a term M for avoiding numerical issue
            
            # [batch_size, M]
            p_w_xc = tf.exp(tf.reduce_logsumexp(
                    map_log_p_zc_w -
                    tf.expand_dims(log_p_z_L, -1), axis=1) -
                    tf.expand_dims(tf.gather_nd(log_p_c_x,
                                list(zip(np.arange(batch_size),
                                        c.numpy()))), -1)) / tf.cast(L, tf.float32)
                
            wc = tf.reduce_sum(
                        tf.tile(self.w, (batch_size,1)) * p_w_xc, axis=-1
                    )
            res['w_xc'] = wc.numpy()
                        
            # var_w|c   -   Var(w|x,c)
            var_wc = tf.reduce_mean(
                        tf.square(
                            tf.tile(self.w, (batch_size,1))
                            ) * p_w_xc, axis=-1
                        ) - tf.square(wc)
            res['var_w_xc'] = var_wc.numpy()
            
            c = tf.where(wc>1e-3, c,
                        tf.gather(self.clusters_ind, tf.gather(self.B, c)))
            c = tf.where(wc<1-1e-3, c,
                        tf.gather(self.clusters_ind, tf.gather(self.A, c)))
            res['c'] = c.numpy()
            
            # [batch_size, n_states, M]
            p_wc_x = tf.exp(tf.reduce_logsumexp(
                        log_p_zc_w -
                        tf.expand_dims(tf.expand_dims(log_p_z_L, -1), -1),
                        1) - tf.math.log(tf.cast(L, tf.float32)))
            res['p_wc_x'] = p_wc_x.numpy()
            
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
            res['w_tilde'] = w_tilde.numpy()
            
            var_w_tilde = tf.reduce_sum(
                tf.math.square(_w)  *
                tf.tile(tf.expand_dims(p_wc_x, 2), (1,1,self.n_clusters,1)),
                (1,3)) - tf.square(w_tilde)
            res['var_w_tilde'] = var_w_tilde.numpy()
            
            # Jensen–Shannon divergence
            _wtilde = tf.expand_dims(tf.expand_dims(w_tilde, 1), -1)
            res['D_JS'] = tf.reduce_sum(
                tf.math.sqrt(
                    0.5 * tf.reduce_mean(
                        _w * tf.math.log(
                            _w/(0.5 * (_w + _wtilde) + 1e-12) + 1e-12) +
                        _wtilde * tf.math.log(
                            _wtilde/(0.5 * (_w + _wtilde) + 1e-12) + 1e-12),
                        axis=2)) * \
                p_wc_x, (1,2)) / tf.cast(self.M, tf.float32).numpy()
            return res

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
    Combines the encoder, decoder and GMM into an end-to-end model for training.
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
        
    def init_GMM(self, n_clusters, mu, pi=None):
        self.n_clusters = n_clusters
        self.GMM = GMM(self.n_clusters, self.dim_latent)
        self.GMM.initialize(mu, pi)

    def call(self, x_normalized, c_score, x = None, scale_factor = 1,
             pre_train = False, L=1):
        # Feed forward through encoder, GMM layer and decoder.
        if not pre_train and self.GMM is None:
            raise ReferenceError('Have not initialized GMM.')
                    
        x_normalized = tf.concat([x_normalized, c_score], -1) if self.has_cov else x_normalized
        _, z_log_var, z = self.encoder(x_normalized, L)
                
        z_in = tf.concat([z, tf.tile(tf.expand_dims(c_score,1), (1,L,1))], -1) if self.has_cov else z
        
        if self.data_type=='Gaussian':
            # Gaussian Log-Likelihood Loss function
            nu_z, tau = self.decoder(z_in)
            x = tf.tile(tf.expand_dims(x, 1), (1,L,1))
            neg_E_Gaus = 0.5 * tf.math.log(tau + 1e-12) + 0.5 * tf.math.square(x - nu_z) / tau
            neg_E_Gaus =  tf.reduce_mean(tf.reduce_sum(neg_E_Gaus, axis=-1))
            self.add_loss(neg_E_Gaus)

        else:
            if self.data_type == 'UMI':
                x_hat, r = self.decoder(z_in)
            else:
                x_hat, r, phi = self.decoder(z_in)

            x_hat = x_hat*tf.expand_dims(scale_factor, -1)
            x = tf.tile(tf.expand_dims(x, 1), (1,L,1))
            # Negative Log-Likelihood Loss function

            # Ref for NB & ZINB loss functions:
            # https://github.com/gokceneraslan/neuralnet_countmodels/blob/master/Count%20models%20with%20neuralnets.ipynb
            # Negative Binomial loss

            neg_E_nb = tf.math.lgamma(r) + tf.math.lgamma(x+1.0) \
                        - tf.math.lgamma(x+r) + \
                        (r+x) * tf.math.log(1.0 + (x_hat/r)) + \
                        x * (tf.math.log(r) - tf.math.log(x_hat+1e-30))
            
            if self.data_type == 'non-UMI':
                # Zero-Inflated Negative Binomial loss
                nb_case = neg_E_nb - tf.math.log(1.0-phi+1e-30)
                zero_case = - tf.math.log(phi + (1.0-phi) *
                                     tf.pow(r/(r + x_hat + 1e-30), r) + 1e-30)
                neg_E_nb = tf.where(tf.less(x, 1e-8), zero_case, nb_case)

            neg_E_nb =  tf.reduce_mean(tf.reduce_sum(neg_E_nb, axis=-1))
            self.add_loss(neg_E_nb)

        if not pre_train:            
            log_p_z = self.GMM(z, inference=False)

            # - E_q[log p(z)]
            self.add_loss(- log_p_z)

            # - Eq[log q(z|x)]
            E_qzx = - tf.reduce_mean(
                            0.5 * self.dim_latent *
                            (tf.math.log(2 * math.pi) + 1) +
                            0.5 * tf.reduce_sum(z_log_var, axis=-1)
                            )
            self.add_loss(E_qzx)
        return self.losses
    
    def get_z(self, x_normalized, c_score):        
        x_normalized = x_normalized if (not self.has_cov or c_score is None) else tf.concat([x_normalized, c_score], -1)
        z_mean, _, _ = self.encoder(x_normalized, 1, False)
        return z_mean.numpy()
    
    def get_proj_z(self, c):
        '''
        Args:
            c - List of indexes of edges
        '''
        return self.GMM.get_proj_z(c)

    def inference(self, test_dataset, L=1):
        if self.GMM is None:
            raise ReferenceError('Have not initialized GMM.')
            
        pi_norm = tf.nn.softmax(self.GMM.pi).numpy()
        mu = self.GMM.mu.numpy()
        z_mean = []
        result = []
        p_c_x = []
        p_wc_x = []
        w_tilde = []
        var_w_tilde = []
        for x,c_score in test_dataset:
            x = tf.concat([x, c_score], -1) if self.has_cov else x
            _z_mean, _, z = self.encoder(x, L, False)
            res = self.GMM(z, inference=True)
            result.append(np.c_[res['c'], res['w_x'], res['var_w_x'], res['w_xc'], res['var_w_xc'], res['D_JS']])
            p_c_x.append(res['p_c_x'])
            z_mean.append(_z_mean.numpy())
            w_tilde.append(res['w_tilde'])
            var_w_tilde.append(res['var_w_tilde'])
            p_wc_x.append(res['p_wc_x'])
        c, w_x, var_w_x, w_xc, var_w_xc, D_JS = np.hsplit(np.concatenate(result), 6)
        c = c[:,0].astype(np.int32)
        w_x = w_x[:,0]
        var_w_x = var_w_x[:,0]
        w_xc = w_xc[:,0]
        var_w_xc = var_w_xc[:,0]
        D_JS = D_JS[:,0]
        z_mean = np.concatenate(z_mean)
        p_c_x = np.concatenate(p_c_x)
        w_tilde = np.concatenate(w_tilde)
        var_w_tilde = np.concatenate(var_w_tilde)
        p_wc_x = np.concatenate(p_wc_x)
        return pi_norm, mu, c, p_c_x, p_wc_x, w_x, var_w_x, w_xc, var_w_xc, w_tilde, var_w_tilde, D_JS, z_mean
