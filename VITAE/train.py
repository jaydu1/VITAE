# -*- coding: utf-8 -*-
from typing import Optional

from VITAE.utils import Early_Stopping, get_embedding

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import Progbar


def clear_session():
    '''Clear Tensorflow sessions.
    '''
    tf.keras.backend.clear_session()
    return None

    
def warp_dataset(X_normalized, c_score, batch_size:int, X=None, scale_factor=None):
    '''Get Tensorflow datasets.

    Parameters
    ----------
    X_normalized : np.array
        \([N, G]\) the preprocessed data.
    c_score : float, optional
        the normalizing constant.
    batch_size : int
        the batch size.
    X : np.array, optional
        \([N, G]\) the raw count data.
    scale_factor : np.array, optional
        \([N, ]\) the raw count data.

    Returns
    ----------
    dataset : tf.Dataset
        the Tensorflow Dataset object.
    '''
    # fake c_score
    if c_score is None:
        c_score = np.zeros((X_normalized.shape[0],1), np.float32)
        
    if X is not None:
        train_dataset = tf.data.Dataset.from_tensor_slices((X, X_normalized, c_score, scale_factor))
        train_dataset = train_dataset.shuffle(buffer_size = X.shape[0],
                                        reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        return train_dataset
    else:
        test_dataset = tf.data.Dataset.from_tensor_slices((X_normalized, 
                                                          c_score)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        return test_dataset


def pre_train(train_dataset, vae, learning_rate: float, L: int, alpha: float,
              num_epoch_pre: int, num_step_per_epoch: int, 
              early_stopping_patience: int, early_stopping_tolerance: int, early_stopping_warmup: int):
    '''Pretraining.

    Parameters
    ----------
    train_dataset : tf.Dataset
        the Tensorflow Dataset object.
    vae : VariationalAutoEncoder
        the model.
    learning_rate : float
        the initial learning rate for the Adam optimizer.
    L : int
        the number of MC samples.
    alpha : float, optional
        the value of alpha in [0,1] to encourage covariate adjustment. Not used if there is no covariates.
    num_epoch_pre : int
        the maximum number of epoches.
    num_step_per_epoch : int
        the number of step per epoch, it will be inferred from number of cells and batch size if it is None.            
    early_stopping_patience : int
        the maximum number of epoches if there is no improvement.
    early_stopping_tolerance : float
        the minimum change of loss to be considered as an improvement.
    early_stopping_warmup : int, optional
        the number of warmup epoches.

    Returns
    ----------
    vae : VariationalAutoEncoder
        the pretrained model.
    '''    
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    loss_metric = tf.keras.metrics.Mean()
    early_stopping = Early_Stopping(patience=early_stopping_patience, tolerance=early_stopping_tolerance, warmup=early_stopping_warmup)

    for epoch in range(num_epoch_pre):
        progbar = Progbar(num_step_per_epoch)
        
        print('Pretrain - Start of epoch %d' % (epoch,))

        # Iterate over the batches of the dataset.
        for step, (x_batch, x_norm_batch, c_score, x_scale_factor) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                losses = vae(x_norm_batch, c_score, x_batch, x_scale_factor, pre_train=True, L=L, alpha=alpha)
                # Compute reconstruction loss
                loss = tf.reduce_sum(losses[0])
            grads = tape.gradient(loss, vae.trainable_weights,
                        unconnected_gradients=tf.UnconnectedGradients.ZERO)
            optimizer.apply_gradients(zip(grads, vae.trainable_weights))                                
            loss_metric(loss)
            
            if (step+1)%10==0 or step+1==num_step_per_epoch:
                progbar.update(step+1, [('Reconstructed Loss', float(loss))])
        if early_stopping(float(loss_metric.result())):
            print('Early stopping.')
            break
        print(' Training loss over epoch: %s' % (float(loss_metric.result()),))
        loss_metric.reset_states()

    print('Pretrain Done.')
    return vae


def train(train_dataset, test_dataset, vae,
        learning_rate: float, 
        L: int, alpha: float, beta: float,
        num_epoch: int, num_step_per_epoch: int, 
        early_stopping_patience: int, early_stopping_tolerance: float, early_stopping_warmup: int,         
        labels, plot_every_num_epoch: Optional[int] = None, dimred: str = 'umap', **kwargs):
    '''Training.

    Parameters
    ----------
    train_dataset : tf.Dataset
        the Tensorflow Dataset object.
    test_dataset : tf.Dataset
        the Tensorflow Dataset object for visualizations.
    vae : VariationalAutoEncoder
        the model.
    learning_rate : float
        the initial learning rate for the Adam optimizer.
    L : int
        the number of MC samples.
    alpha : float
        the value of alpha in [0,1] to encourage covariate adjustment. Not used if there is no covariates.
    beta : float
        the value of beta in beta-VAE.
    num_epoch : int
        the maximum number of epoches.
    num_step_per_epoch : int
        the number of step per epoch, it will be inferred from number of cells and batch size if it is None.            
    early_stopping_patience : int
        the maximum number of epoches if there is no improvement.
    early_stopping_tolerance : float, optional 
        the minimum change of loss to be considered as an improvement.
    early_stopping_warmup : int
        the number of warmup epoches.
    labels: np.array
        the labels for visualizations of the intermediate results.
    plot_every_num_epoch : int, optional 
        plot the intermediate result every few epoches, or not plotting if it is None.            
    dimred : str, optional 
        the name of dimension reduction algorithms, can be 'umap', 'pca' and 'tsne'. Only used if 'plot_every_num_epoch' is not None. 
    **kwargs : 
        extra key-value arguments for dimension reduction algorithms.    

    Returns
    ----------
    vae : VariationalAutoEncoder
        the trained model.
    '''   
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss_total = tf.keras.metrics.Mean()
    loss_neg_E_nb = tf.keras.metrics.Mean()
    loss_neg_E_pz = tf.keras.metrics.Mean()
    loss_E_qzx = tf.keras.metrics.Mean()
    early_stopping = Early_Stopping(patience = early_stopping_patience, tolerance = early_stopping_tolerance, warmup=early_stopping_warmup)

    print('Warmup:%d'%early_stopping_warmup)
    weight = np.array([1,beta,beta], dtype=np.float32)
    weight = tf.convert_to_tensor(weight)
    
    for epoch in range(num_epoch):
        print('Start of epoch %d' % (epoch,))
        progbar = Progbar(num_step_per_epoch)
        
        # Iterate over the batches of the dataset.
        for step, (x_batch, x_norm_batch, c_score, x_scale_factor) in enumerate(train_dataset):
            if epoch<early_stopping_warmup:
                with tf.GradientTape() as tape:
                    losses = vae(x_norm_batch, c_score, x_batch, x_scale_factor, L=L, alpha=alpha)
                    # Compute reconstruction loss
                    loss = tf.reduce_sum(losses[1:])
                grads = tape.gradient(loss, vae.latent_space.trainable_weights,
                            unconnected_gradients=tf.UnconnectedGradients.ZERO)
                optimizer.apply_gradients(zip(grads, vae.latent_space.trainable_weights))
            else:
                with tf.GradientTape() as tape:
                    losses = vae(x_norm_batch, c_score, x_batch, x_scale_factor, L=L, alpha=alpha)
                    # Compute reconstruction loss
                    loss = tf.reduce_sum(losses*weight)
                grads = tape.gradient(loss, vae.trainable_weights,
                            unconnected_gradients=tf.UnconnectedGradients.ZERO)
                optimizer.apply_gradients(zip(grads, vae.trainable_weights))

            loss_total(loss)
            loss_neg_E_nb(losses[0])
            loss_neg_E_pz(losses[1])
            loss_E_qzx(losses[2])

            if (step+1)%10==0 or step+1==num_step_per_epoch:
                progbar.update(step+1, [
                        ('loss_neg_E_nb'    ,   float(losses[0])),
                        ('loss_neg_E_pz'    ,   float(losses[1])),
                        ('loss_E_qzx   '    ,   float(losses[2])),
                        ('loss_total'       ,   float(loss_total.result()))
                        ])
        
        if early_stopping(float(loss_total.result())):
            print('Early stopping.')
            break
        print(' Training loss over epoch: %s' % (float(loss_total.result())))
        print('% 4.6f, % 4.6f, % 4.6f' % (float(loss_neg_E_nb.result()),
                                          float(loss_neg_E_pz.result()),
                                          float(loss_E_qzx.result())))
        loss_total.reset_states()
        loss_neg_E_nb.reset_states()
        loss_neg_E_pz.reset_states()
        loss_E_qzx.reset_states()

        if plot_every_num_epoch is not None and (epoch%plot_every_num_epoch==0 or epoch==num_epoch-1):
            _, mu, _, w_tilde, _, _, z_mean = vae.inference(test_dataset, 1)
            c = np.argmax(w_tilde, axis=-1)
            
            concate_z = np.concatenate((z_mean, mu.T), axis=0)
            u = get_embedding(concate_z, dimred, **kwargs)
            uz = u[:len(z_mean),:]
            um = u[len(z_mean):,:]            
            
            if labels is None:
                fig, ax1 = plt.subplots(1, figsize=(7, 6))
            else:
                fig, (ax1,ax2) = plt.subplots(1,2, figsize=(16, 6))
            
                ax2.scatter(uz[:,0], uz[:,1], c = labels, s = 2)
                ax2.set_title('Ground Truth')
            
            ax1.scatter(uz[:,0], uz[:,1], c = c, s = 2, alpha = 0.5)
            ax1.set_title('Prediction')
            cluster_center = [(len(um)+(1-i)/2)*i for i in range(len(um))]
            ax1.scatter(um[:,0], um[:,1], c=cluster_center, s=100, marker='s')
            plt.show()

    print('Training Done!')

    return vae
