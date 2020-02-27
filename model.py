import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, BatchNormalization
import tensorflow.keras.backend as K
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
        self.dense_layers = [Dense(dim, activation = 'relu',
                                          name = 'encoder_%i'%(i+1)) \
                             for (i, dim) in enumerate(dimensions)]
        self.batch_norm_layers = [BatchNormalization() \
                                    for _ in range(len((dimensions)))]
        self.latent_mean = Dense(dim_latent, name = 'latent_mean')
        self.latent_log_var = Dense(dim_latent, name = 'latent_log_var')
        self.sampling = Sampling()

    def call(self, x):
        '''
        Input :
            x           - input                     [batch_size, dim_origin]
        Output:
            z_mean      - mean of p(z|x)            [batch_size, dim_latent]
            z_log_var   - log of variance of p(z|x) [batch_size, dim_latent]
            z           - sampled z                 [batch_size, dim_latent]
        '''
        for dense, bn in zip(self.dense_layers, self.batch_norm_layers):
            x = dense(x)
            x = bn(x)
        z_mean = self.latent_mean(x)
        z_log_var = self.latent_log_var(x)
        z = self.sampling(z_mean, z_log_var)
        return z_mean, z_log_var, z


class Decoder(Layer):
    '''
    Decoder, model p(x|z).
    '''
    def __init__(self, dimensions, dim_origin,
                 data_type = 'UMI', name = 'decoder', **kwargs):
        '''
        Input:
            dimensions  - list of dimensions of layers in dense layers of
                                decoder expcept the output layer.
            dim_origin  - dimension of output layer.
        '''
        super(Decoder, self).__init__(name = name, **kwargs)
        self.data_type = data_type
        self.dense_layers = [Dense(dim, activation = 'relu',
                                          name = 'decoder_%i'%(i+1)) \
                             for (i,dim) in enumerate(dimensions)]
        self.batch_norm_layers = [BatchNormalization() \
                                    for _ in range(len((dimensions)))]
        self.log_lambda_z = Dense(dim_origin, name = 'log_lambda_z')
                
        # dispersion parameter
        self.log_r = tf.Variable(tf.zeros([1, dim_origin]),
                                 constraint = lambda t: tf.clip_by_value(t,-30.,6.),
                                 name = "log_r")
        
        if self.data_type == 'non-UMI':
            self.phi = Dense(dim_origin, activation = 'sigmoid', name = "phi")
            
    def call(self, z):
        '''
        Input :
            z           - latent variables  [batch_size, dim_origin]
        Output:
            lambda_z    - x_hat             [batch_size, dim_origin]
            r           - dispersion parameter
                                            [1,          dim_origin]
        '''
        for dense, bn in zip(self.dense_layers, self.batch_norm_layers):
            z = dense(z)
            z = bn(z)
        lambda_z = tf.math.exp(self.log_lambda_z(z))
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
                                name = "mu")
                
        # [diag(Sigma_1), ... , diag(Sigma_K)] in R^(dim_latent * n_clusters)
        # self.Sigma = tf.Variable(tf.ones([self.dim_latent, self.n_clusters]),
        #                          constraint=lambda x: tf.clip_by_value(x, 1e-30, np.infty),
        #                          name="Sigma")

    def initialize(self, mu, Sigma, pi):
        # Initialize mu and sigma computed by GMM
        self.pi.assign(pi)
        self.mu.assign(mu)

        # self.Sigma.assign(Sigma)

    def call(self, z, cluster=False, inference=False):
        '''
        Input :
                z       - latent variables outputed by the encoder
                          [batch_size, dim_latent]
        Output:
                p_z     - MC samples for p(z)=sum_{c,w}(p_zc_w)/M
                          [batch_size, ]
                p_zc_w  - MC samples for p(z,c|w)=p(z|w,c)p(c) (L=1)
                          [batch_size, n_states, M]
        '''
        batch_size = tf.shape(z)[0]
        
        if cluster:
            # [batch_size, dim_latent, n_clusters]
            temp_Z = tf.tile(tf.expand_dims(z,-1),
                        (1,1,self.n_clusters))
            temp_mu = tf.tile(
                            tf.expand_dims(
                                self.mu,0),
                            (batch_size,1,1))
            distance = tf.sqrt(tf.reduce_sum(tf.math.square(temp_Z - temp_mu),1))
            c = tf.argmin(distance, 1)
            cluster_loss = tf.reduce_mean(
                tf.reduce_max(tf.exp(-distance),1)/
                (tf.reduce_sum(tf.exp(-distance),1)+1e-30)
                )

            return self.mu, c, cluster_loss
        else:
            # [batch_size, dim_latent, n_states, M]
            temp_Z = tf.tile(tf.expand_dims(tf.expand_dims(z,-1), -1),
                        (1,1,self.n_states,self.M))
            temp_mu = tf.tile(
                            tf.expand_dims(tf.expand_dims(
                                tf.gather(self.mu, self.A, axis=1),0),-1),
                            (batch_size,1,1,self.M)) * self.w + \
                    tf.tile(
                            tf.expand_dims(tf.expand_dims(
                                tf.gather(self.mu, self.B, axis=1),0),-1),
                            (batch_size,1,1,self.M)) * (1-self.w)
                
            # [batch_size, n_states, M]
            temp_pi = tf.tile(
                            tf.expand_dims(tf.nn.softmax(self.pi), -1),
                            (batch_size, 1, self.M))
            log_p_zc_w = - 0.5 * self.dim_latent * tf.math.log(2 * math.pi) + \
                            tf.math.log(temp_pi+1e-30) - \
                            tf.reduce_sum(tf.math.square(temp_Z - temp_mu)/2, 1)
            # [batch_size, ]
            log_p_z = tf.math.reduce_logsumexp(log_p_zc_w, axis=[1,2]) \
                        - tf.math.log(tf.cast(self.M, tf.float32))
                                
            if inference:
                # p_c_x     -   predicted probability distribution
                p_c_x = tf.math.reduce_logsumexp(log_p_zc_w, axis=2)

                # c         -   predicted clusters
                c = tf.math.argmax(
                    tf.math.reduce_logsumexp(log_p_zc_w, axis=2), axis=1)
                c = tf.cast(c, 'int32')

                # w         -   E(w|x)
                w =  tf.reduce_sum(
                        tf.tile(self.w, (batch_size,1)) *
                        tf.exp(tf.math.reduce_logsumexp(log_p_zc_w, axis=1)) /
                        tf.expand_dims(tf.exp(log_p_z)+1e-30, -1) /
                        tf.cast(self.M, tf.float32), axis=-1
                    )
                
                # var_w     -   Var(w|x)
                var_w =  tf.reduce_sum(
                            tf.square(
                                tf.tile(self.w, (batch_size,1)) -
                                tf.expand_dims(w, -1)) *
                            tf.exp(tf.math.reduce_logsumexp(log_p_zc_w, axis=1)) /
                            tf.expand_dims(tf.exp(log_p_z)+1e-30, -1) /
                            tf.cast(self.M, tf.float32), axis=-1
                        )
                # w|c       -   E(w|x,c)
                map_log_p_zc_w = tf.gather_nd(log_p_zc_w,
                                              list(zip(np.arange(batch_size),
                                                       c.numpy())))
                wc = tf.reduce_sum(
                            tf.tile(self.w, (batch_size,1)) *
                            tf.exp(map_log_p_zc_w -
                                tf.math.reduce_logsumexp(
                                    map_log_p_zc_w, axis=1, keepdims=True)),
                            axis=-1
                        )

                # var_w|c   -   Var(w|x,c)
                var_wc = tf.reduce_sum(
                            tf.square(tf.tile(self.w, (batch_size,1)) -
                                      tf.expand_dims(wc, -1)) *
                            tf.exp(map_log_p_zc_w -
                               tf.math.reduce_logsumexp(
                                   map_log_p_zc_w, axis=1, keepdims=True)),
                            axis=-1
                        )
                
                # c = tf.where(w>0, c,
                #              tf.gather(self.clusters_ind, tf.gather(self.A, c)))
                # c = tf.where(w<1, c,
                #              tf.gather(self.clusters_ind, tf.gather(self.B, c)))
                      
                # proj_z    -   projection of z to the segment of two clusters
                #               in the latent space
                proj_z = tf.transpose(
                    tf.gather(self.mu, tf.gather(self.A, c), axis=1) *
                    tf.expand_dims(w, 0) +
                    tf.gather(self.mu, tf.gather(self.B, c), axis=1) *
                    (1-tf.expand_dims(w, 0)))
                return (tf.nn.softmax(self.pi).numpy(), self.mu.numpy(), 
                        log_p_z.numpy(), c.numpy(), w.numpy(), var_w.numpy(), 
                        wc.numpy(), var_wc.numpy(), proj_z.numpy())
            else:
                return tf.nn.softmax(self.pi), self.mu, log_p_z

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
    def __init__(self, n_clusters, dim_origin, dimensions, dim_latent,
                 data_type = 'UMI', name = 'autoencoder', **kwargs):
        '''
        Args:
            n_clusters  -   Number of clusters.
            dim_origin  -   Dim of input.
            dimensions  -   List of dimensions of layers of the encoder. Assume
                            symmetric network sturcture of encoder and decoder.
            dim_latent  -   Dimension of latent layer.
            data_type   -   Type of count data.
                            'UMI' for negative binomial loss;
                            'non-UMI' for zero-inflated negative binomial loss.
        '''
        super(VariationalAutoEncoder, self).__init__(name = name, **kwargs)
        self.data_type = data_type
        self.dim_origin = dim_origin
        self.dim_latent = dim_latent
        self.n_clusters = n_clusters

        self.encoder = Encoder(dimensions, dim_latent)
        self.GMM = GMM(n_clusters, dim_latent)
        self.decoder = Decoder(dimensions[::-1], dim_origin, data_type)

    def call(self, x_normalized, x = None, scale_factor = 1,
             cluster = False, inference = False, pre_train = False):
        # Feed forward through encoder, GMM layer and decoder.
        z_mean, z_log_var, z = self.encoder(x_normalized)

        if self.data_type == 'UMI':
            x_hat, r = self.decoder(z)
        else:
            x_hat, r, phi = self.decoder(z)

        if cluster:
            mu, c, cluster_loss = self.GMM(z, cluster=cluster)
            self.add_loss(cluster_loss)
            return None
        elif inference:
            pi_norm, mu, log_p_z, c, w, var_w, wc, var_wc, proj_z = self.GMM(z_mean, inference=inference)
            return pi_norm, mu, c, w, var_w, wc, var_wc, z_mean, proj_z

        x_hat = x_hat*scale_factor

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
        if pre_train:
            return None
        else:
            pi_norm, mu, log_p_z = self.GMM(z, inference=inference)

            # - E_q[log p(z)]
            neg_E_pz = tf.reduce_mean(- log_p_z)
            self.add_loss(neg_E_pz)

            # - Eq[log q(z|x)]
            E_qzx = - tf.reduce_mean(
                            0.5 * self.dim_latent *
                            (tf.math.log(2 * math.pi) + 1) +
                            0.5 * tf.reduce_sum(z_log_var, axis=-1)
                            )
            self.add_loss(E_qzx)
            return None
    
    def get_proj_z(self, c):
        '''
        Args:
            c - List of indexes of edges
        '''
        return self.GMM.get_proj_z(c)