# -*- coding: utf-8 -*-
from typing import Optional

from VITAE.utils import Early_Stopping
from numba.core.types.scalars import Boolean

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Progbar


def clear_session():
    '''Clear Tensorflow sessions.
    '''
    tf.keras.backend.clear_session()
    return None

    
def warp_dataset(X_normalized, c_score, batch_size:int, X=None, scale_factor=None, seed=0, conditions = None, pi_cov = None):
    '''Get Tensorflow datasets.

    Parameters
    ----------
    X_normalized : np.array
        \([N, G]\) The preprocessed data.
    c_score : float, optional
        The normalizing constant.
    batch_size : int
        The batch size.
    X : np.array, optional
        \([N, G]\) The raw count data.
    scale_factor : np.array, optional
        \([N, ]\) The raw count data.
    seed : int, optional
        The random seed for data shuffling.
    conditions: str or list, optional
        The conditions of different cells

    Returns
    ----------
    dataset : tf.Dataset
        The Tensorflow Dataset object.
    '''
    # fake c_score
    if c_score is None:
        c_score = np.zeros((X_normalized.shape[0],1), tf.keras.backend.floatx())
        
    if X is not None:
        train_dataset = tf.data.Dataset.from_tensor_slices((X, X_normalized, c_score, scale_factor, conditions, pi_cov))
        train_dataset = train_dataset.shuffle(buffer_size = X.shape[0], seed=seed,
                                        reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        return train_dataset
    else:
        test_dataset = tf.data.Dataset.from_tensor_slices((X_normalized, 
                                                          c_score, conditions, pi_cov)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        return test_dataset


def pre_train(train_dataset, test_dataset, vae, learning_rate: float, L: int, alpha: float, gamma: float, phi: float,
              num_epoch: int, num_step_per_epoch: int, 
              es_patience: int, es_tolerance: int, es_relative: bool,
              verbose: bool = True, conditions = None):
    '''Pretraining.

    Parameters
    ----------
    train_dataset : tf.Dataset
        The Tensorflow Dataset object.
    test_dataset : tf.Dataset
        The Tensorflow Dataset object.
    vae : VariationalAutoEncoder
        The model.
    learning_rate : float
        The initial learning rate for the Adam optimizer.
    L : int
        The number of MC samples.
    alpha : float, optional
        The value of alpha in [0,1] to encourage covariate adjustment. Not used if there is no covariates.
    phi : float, optional
        The weight of Jocob norm of the encoder.
    num_epoch : int
        The maximum number of epoches.
    num_step_per_epoch : int
        The number of step per epoch, it will be inferred from number of cells and batch size if it is None.            
    es_patience : int
        The maximum number of epoches if there is no improvement.
    es_tolerance : float
        The minimum change of loss to be considered as an improvement.
    es_relative : bool, optional
        Whether monitor the relative change of loss or not.        
    es_warmup : int, optional
        The number of warmup epoches.
    conditions : str or list
        The conditions of different cells

    Returns
    ----------
    vae : VariationalAutoEncoder
        The pretrained model.
    '''    
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    loss_train = tf.keras.metrics.Mean()
    loss_test = tf.keras.metrics.Mean()
    early_stopping = Early_Stopping(patience=es_patience, tolerance=es_tolerance, relative=es_relative)

    if not verbose:
        progbar = Progbar(num_epoch)
    for epoch in range(num_epoch):

        if verbose:
            progbar = Progbar(num_step_per_epoch)
            print('Pretrain - Start of epoch %d' % (epoch,))
        else:
            if (epoch+1)%2==0 or epoch+1==num_epoch:
                    progbar.update(epoch+1)

        # Iterate over the batches of the dataset.
        for step, (x_batch, x_norm_batch, c_score, x_scale_factor, x_condition, _) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                losses = vae(x_norm_batch, c_score, x_batch, x_scale_factor, pre_train=True, L=L, alpha=alpha, gamma = gamma, phi = phi, conditions = x_condition)
                # Compute reconstruction loss
                loss = tf.reduce_sum(losses[0:3]) # neg_ll, Jacob, mmd_loss
            grads = tape.gradient(loss, vae.trainable_weights,
                        unconnected_gradients=tf.UnconnectedGradients.ZERO)
            optimizer.apply_gradients(zip(grads, vae.trainable_weights))                                
            loss_train(loss)
            
            if verbose:
                if (step+1)%10==0 or step+1==num_step_per_epoch:
                    progbar.update(step + 1, [
                        ('loss_neg_E_nb', float(losses[0])),
                        ('loss_Jacob', float(losses[1])),
                        ('loss_MMD', float(losses[2])),
                        ('loss_total', float(loss))
                    ])
                
        for step, (x_batch, x_norm_batch, c_score, x_scale_factor, x_condition, _) in enumerate(test_dataset):
            losses = vae(x_norm_batch, c_score, x_batch, x_scale_factor, pre_train=True, L=L, alpha=alpha, gamma = gamma, phi = phi, conditions = x_condition)
            loss = tf.reduce_sum(losses[0:3]) # neg_ll, Jacob, mmd_loss
            loss_test(loss)

        if verbose:
            print(' Training loss over epoch: %.4f. Testing loss over epoch: %.4f' % (float(loss_train.result()),
                                                                            float(loss_test.result())))
        if early_stopping(float(loss_test.result())):
            print('Early stopping.')
            break
        loss_train.reset_states()
        loss_test.reset_states()

    print('Pretrain Done.')
    return vae


def train(train_dataset, test_dataset, vae,
        learning_rate: float, 
        L: int, alpha: float, beta: float, gamma: float, phi: float,
        num_epoch: int, num_step_per_epoch: int, 
        es_patience: int, es_tolerance: float, es_relative: bool, es_warmup: int, 
        verbose: bool = False, pi_cov = None, **kwargs):
    '''Training.

    Parameters
    ----------
    train_dataset : tf.Dataset
        The Tensorflow Dataset object.
    test_dataset : tf.Dataset
        The Tensorflow Dataset object.
    vae : VariationalAutoEncoder
        The model.
    learning_rate : float
        The initial learning rate for the Adam optimizer.
    L : int
        The number of MC samples.
    alpha : float
        The value of alpha in [0,1] to encourage covariate adjustment. Not used if there is no covariates.
    beta : float
        The value of beta in beta-VAE.
    gamma : float
        The weight of mmd_loss.
    phi : float
        The weight of Jacob norm of the encoder.
    num_epoch : int
        The maximum number of epoches.
    num_step_per_epoch : int
        The number of step per epoch, it will be inferred from number of cells and batch size if it is None.            
    es_patience : int
        The maximum number of epoches if there is no improvement.
    es_tolerance : float, optional 
        The minimum change of loss to be considered as an improvement.
    es_relative : bool, optional
        Whether monitor the relative change of loss or not.          
    es_warmup : int
        The number of warmup epoches.
    **kwargs : 
        Extra key-value arguments for dimension reduction algorithms.    

    Returns
    ----------
    vae : VariationalAutoEncoder
        The trained model.
    '''   
    optimizer_ = tf.keras.optimizers.Adam(learning_rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss_test = [tf.keras.metrics.Mean() for _ in range(6)]
    loss_train = [tf.keras.metrics.Mean() for _ in range(6)]
    early_stopping = Early_Stopping(patience = es_patience, tolerance = es_tolerance, relative=es_relative, warmup=es_warmup)

    print('Warmup:%d'%es_warmup)
    weight = np.array([1,1,1,beta,beta], dtype=tf.keras.backend.floatx())
    weight = tf.convert_to_tensor(weight)
    
    if not verbose:
        progbar = Progbar(num_epoch)
    for epoch in range(num_epoch):

        if verbose:
            progbar = Progbar(num_step_per_epoch)
            print('Start of epoch %d' % (epoch,))
        else:
            if (epoch+1)%2==0 or epoch+1==num_epoch:
                    progbar.update(epoch+1)

        
        # Iterate over the batches of the dataset.
        for step, (x_batch, x_norm_batch, c_score, x_scale_factor, x_condition, pi_cov) in enumerate(train_dataset):
            if epoch<es_warmup:
                with tf.GradientTape() as tape:
                    losses = vae(x_norm_batch, c_score, x_batch, x_scale_factor, L=L, alpha=alpha, gamma = gamma,phi = phi, conditions = x_condition, pi_cov = pi_cov)
                    # Compute reconstruction loss
                    loss = tf.reduce_sum(losses[3:])
                grads = tape.gradient(loss, vae.latent_space.trainable_weights,
                            unconnected_gradients=tf.UnconnectedGradients.ZERO)
                optimizer_.apply_gradients(zip(grads, vae.latent_space.trainable_weights))
            else:
                with tf.GradientTape() as tape:
                    losses = vae(x_norm_batch, c_score, x_batch, x_scale_factor, L=L, alpha=alpha, gamma = gamma, phi = phi, conditions = x_condition, pi_cov = pi_cov)
                    # Compute reconstruction loss
                    loss = tf.reduce_sum(losses*weight)
                grads = tape.gradient(loss, vae.trainable_weights,
                            unconnected_gradients=tf.UnconnectedGradients.ZERO)
                optimizer.apply_gradients(zip(grads, vae.trainable_weights))

            loss_train[0](losses[0])
            loss_train[1](losses[1])
            loss_train[2](losses[2])
            loss_train[3](losses[3])
            loss_train[4](losses[4])
            loss_train[5](loss)
            
            if verbose:
                if (step+1)%10==0 or step+1==num_step_per_epoch:
                    progbar.update(step+1, [
                            ('loss_neg_E_nb'    ,   float(losses[0])),
                            ('loss_Jacob', float(losses[1])),
                            ('loss_MMD', float(losses[2])),
                            ('loss_neg_E_pz'    ,   float(losses[3])),
                            ('loss_E_qzx   '    ,   float(losses[4])),
                            ('loss_total'       ,   float(loss))
                            ])
                        
        for step, (x_batch, x_norm_batch, c_score, x_scale_factor, x_condition, pi_cov) in enumerate(test_dataset):
            losses = vae(x_norm_batch, c_score, x_batch, x_scale_factor, L=L, alpha=alpha, gamma = gamma, phi = phi, conditions = x_condition, pi_cov = pi_cov)
            loss = tf.reduce_sum(losses*weight)
            loss_test[0](losses[0])
            loss_test[1](losses[1])
            loss_test[2](losses[2])
            loss_test[3](losses[3])
            loss_test[4](losses[4])
            loss_test[5](loss)
            
        if early_stopping(float(loss_test[5].result())):
            print('Early stopping.')
            break
        
        if verbose:
            print(' Training loss over epoch: %.4f (%.4f, %.4f, %.4f, %.4f, %.4f) Testing loss over epoch: %.4f (%.4f, %.4f, %.4f, %.4f, %.4f)' % (
                float(loss_train[5].result()),
                float(loss_train[0].result()),
                float(loss_train[1].result()),
                float(loss_train[2].result()),
                float(loss_train[3].result()),
                float(loss_train[4].result()),
                float(loss_test[5].result()),
                float(loss_test[0].result()),
                float(loss_test[1].result()),
                float(loss_test[2].result()),
                float(loss_test[3].result()),
                float(loss_test[4].result())))

        [l.reset_states() for l in loss_train]
        [l.reset_states() for l in loss_test]


    print('Training Done!')

    return vae
