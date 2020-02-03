import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, BatchNormalization
from tensorflow.keras.utils import Progbar
import tensorflow.keras.backend as K

from sklearn.mixture import GaussianMixture

import numpy as np
import pandas as pd


class Sampling(Layer):
    """
    Sampling latent variable z by (z_mean, z_log_var).    
    Used in Encoder.
    """
    def call(self, z_mean, z_log_var):
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
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
        super(Encoder, self).__init__(name=name, **kwargs)      
        self.dense_layers = [Dense(dim, activation='relu', 
                                          name='encoder_%i'%(i+1)) \
                             for (i,dim) in enumerate(dimensions)]
        self.batch_norm_layers = [BatchNormalization() \
                                    for _ in range(len((dimensions)))]
        self.latent_mean = Dense(dim_latent, name='latent_mean')
        self.latent_log_var = Dense(dim_latent, name='latent_log_var')
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
                 name='decoder', **kwargs):
        '''
        Input:
            dimensions  - list of dimensions of layers in dense layers of 
                                decoder expcept the output layer.
            dim_origin  - dimension of output layer.
        '''
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_layers = [Dense(dim, activation='relu', 
                                          name='decoder_%i'%(i+1)) \
                             for (i,dim) in enumerate(dimensions)]
        self.batch_norm_layers = [BatchNormalization() \
                                    for _ in range(len((dimensions)))]                             
        self.log_lambda_z = Dense(dim_origin, name='log_lambda_z')
        
        # dispersion parameter
        self.log_r = tf.Variable(tf.zeros([1, dim_origin]), 
                                name="log_r",)

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
        r = tf.clip_by_value(tf.exp(self.log_r), 1e-30, 1e6)
        return lambda_z, r


class GMM(Layer):
    '''
    GMM layer. 
    It contains parameters related to model assumptions of GMM. 
    '''
    def __init__(self, n_clusters, dim_latent, M=100, name='GMM', **kwargs):
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
            np.resize(np.arange(0,M)/M, (1, M)), tf.float32)

        # [pi_1, ... , pi_K] in R^(n_states)
        self.pi = tf.Variable(tf.ones([1, self.n_states]) / self.n_states, 
                                name='pi')
        
        # [mu_1, ... , mu_K] in R^(dim_latent * n_clusters)
        self.mu = tf.Variable(tf.random.uniform([self.dim_latent, self.n_clusters],
                                                minval=-1,maxval=1),
                                name="mu")        
                
        # [diag(Sigma_1), ... , diag(Sigma_K)] in R^(dim_latent * n_clusters)
        # self.Sigma = tf.Variable(tf.ones([self.dim_latent, self.n_clusters]), 
        #                          constraint=lambda x: tf.clip_by_value(x, 1e-30, np.infty),
        #                          name="Sigma")
    
    def initialize(self, mu, Sigma):
        # Initialize mu and sigma computed by GMM
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
            distance = tf.reduce_mean(tf.math.square(temp_Z - temp_mu), 1)
            c = tf.argmin(distance, 1)
            cluster_loss = tf.reduce_mean(
                tf.reduce_max(tf.exp(-distance),1)/(tf.reduce_sum(tf.exp(-distance),1)+1e-30) 
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
                            tf.reduce_sum(tf.math.square(temp_Z - temp_mu) / 2, 1)                        
            log_p_z = tf.math.reduce_logsumexp(log_p_zc_w, axis=[1,2]) \
                        - tf.math.log(tf.cast(self.M, tf.float32))
                                
            if inference:
                c = tf.math.argmax(
                    tf.math.reduce_logsumexp(log_p_zc_w, axis=2), axis=1)
                c = tf.cast(c, 'int32')
                w =  tf.squeeze(
                    tf.matmul(self.w, tf.exp(tf.transpose(tf.math.reduce_logsumexp(log_p_zc_w, axis=1)))) /
                    tf.expand_dims(tf.exp(log_p_z)+1e-30, 0) / 
                    tf.cast(self.M, tf.float32)
                    )                
                c = tf.where(w>0, c, tf.gather(self.clusters_ind, tf.gather(self.A, c)))
                c = tf.where(w<1, c, tf.gather(self.clusters_ind, tf.gather(self.B, c)))                
                proj_z = tf.transpose(
                    tf.gather(self.mu, tf.gather(self.A, c), axis=1) * 
                    tf.expand_dims(w, 0) + 
                    tf.gather(self.mu, tf.gather(self.B, c), axis=1) * 
                    (1-tf.expand_dims(w, 0)))
                return tf.nn.softmax(self.pi), self.mu, log_p_z, c, w, proj_z
            else:
                return tf.nn.softmax(self.pi), self.mu, log_p_z


class VariationalAutoEncoder(tf.keras.Model):
    """
    Combines the encoder, decoder and GMM into an end-to-end model for training.
    """
    def __init__(self, n_clusters, dim_origin, dimensions, dim_latent,
                 name='autoencoder', **kwargs):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)

        self.dim_origin = dim_origin
        self.dim_latent = dim_latent 
        self.n_clusters = n_clusters

        self.encoder = Encoder(dimensions, dim_latent)
        self.GMM = GMM(n_clusters, dim_latent)
        self.decoder = Decoder(dimensions[::-1], dim_origin)
    
    def call(self, x_normalized, x=None, scale_factor=1, cluster=False, inference=False, pre_train=False):
        # Feed forward through encoder, GMM layer and decoder.
        z_mean, z_log_var, z = self.encoder(x_normalized)        

        if pre_train:
            x_hat, r = self.decoder(z)        
            x_hat = x_hat*scale_factor

            # Negative Log-Likelihood Loss function
            neg_E_nb =  tf.reduce_mean(
                            tf.reduce_sum(
                                tf.math.lgamma(r) + tf.math.lgamma(x+1.0)
                                - tf.math.lgamma(x+r) +
                                # what if x_hat=0 (x=0)?
                                (r+x) * tf.math.log(1.0 + (x_hat/r)) +
                                x * (tf.math.log(r) - tf.math.log(x_hat+1e-30)),
                            axis=-1),
                            name='- E log p(x|z)'
                        )    
            self.add_loss(neg_E_nb)
            return None

        if cluster:
            mu, c, cluster_loss = self.GMM(z, cluster=cluster)
        elif inference:
            pi_norm, mu, log_p_z, c, w, proj_z = self.GMM(z, inference=inference)
            return  pi_norm, mu, c, w, z, proj_z
        else:
            pi_norm, mu, log_p_z = self.GMM(z, inference=inference)

        if cluster:
            self.add_loss(cluster_loss)
            return None
        else:
            x_hat, r = self.decoder(z)        
            x_hat = x_hat*scale_factor

            # Negative Log-Likelihood Loss function
            neg_E_nb =  tf.reduce_mean(
                            tf.reduce_sum(
                                tf.math.lgamma(r) + tf.math.lgamma(x+1.0)
                                - tf.math.lgamma(x+r) +
                                # what if x_hat=0 (x=0)?
                                (r+x) * tf.math.log(1.0 + (x_hat/r)) +
                                x * (tf.math.log(r) - tf.math.log(x_hat+1e-30)),
                            axis=-1),
                            name='- E log p(x|z)'
                        )    
            self.add_loss(neg_E_nb)

            neg_E_pz = tf.reduce_mean(- log_p_z,
                                    name='- E log p(z)')
            self.add_loss(neg_E_pz)

            E_qzx = - tf.reduce_mean(                        
                            0.5 * self.dim_latent * (tf.math.log(2 * math.pi) + 1) +
                            0.5 * tf.reduce_sum(z_log_var, axis=-1),
                            name = 'E log q(z|x)')                        
            self.add_loss(E_qzx)
