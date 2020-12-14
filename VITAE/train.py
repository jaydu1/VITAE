# -*- coding: utf-8 -*-
from VITAE.utils import Early_Stopping, get_embedding

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import Progbar


def clear_session():
    tf.keras.backend.clear_session()
    return None

    
def warp_dataset(X_normalized, c_score, BATCH_SIZE, X=None, Scale_factor=None):
    # fake c_score
    if c_score is None:
        c_score = np.zeros((X_normalized.shape[0],1), np.float32)
        
    if X is not None:
        train_dataset = tf.data.Dataset.from_tensor_slices((X, X_normalized, c_score, Scale_factor))
        train_dataset = train_dataset.shuffle(buffer_size = X.shape[0],
                                        reshuffle_each_iteration=True).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
        return train_dataset
    else:
        test_dataset = tf.data.Dataset.from_tensor_slices((X_normalized, 
                                                          c_score)).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
        return test_dataset


def pre_train(train_dataset, vae,
              learning_rate, patience, tolerance, warmup, 
              NUM_EPOCH_PRE, NUM_STEP_PER_EPOCH, L, alpha):
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    loss_metric = tf.keras.metrics.Mean()
    early_stopping = Early_Stopping(patience=patience, tolerance=tolerance, warmup=warmup)

    for epoch in range(NUM_EPOCH_PRE):
        progbar = Progbar(NUM_STEP_PER_EPOCH)
        
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
            
            if (step+1)%10==0 or step+1==NUM_STEP_PER_EPOCH:
                progbar.update(step+1, [('Reconstructed Loss', float(loss))])
        if early_stopping(float(loss_metric.result())):
            print('Early stopping.')
            break
        print(' Training loss over epoch: %s' % (float(loss_metric.result()),))
        loss_metric.reset_states()

    print('Pretrain Done.')
    return vae


def train(train_dataset, test_dataset, vae,
        learning_rate, patience, tolerance, warmup, NUM_EPOCH, NUM_STEP_PER_EPOCH, 
        L, alpha, beta,
        labels, plot_every_num_epoch=None, dimred='umap', **kwargs):
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss_total = tf.keras.metrics.Mean()
    loss_neg_E_nb = tf.keras.metrics.Mean()
    loss_neg_E_pz = tf.keras.metrics.Mean()
    loss_E_qzx = tf.keras.metrics.Mean()
    early_stopping = Early_Stopping(patience = patience, tolerance = tolerance, warmup=warmup)

    print('Warmup:%d'%warmup)
    weight = np.array([1,beta,beta], dtype=np.float32)
    weight = tf.convert_to_tensor(weight)
    
    for epoch in range(NUM_EPOCH):
        print('Start of epoch %d' % (epoch,))
        progbar = Progbar(NUM_STEP_PER_EPOCH)
        
        # Iterate over the batches of the dataset.
        for step, (x_batch, x_norm_batch, c_score, x_scale_factor) in enumerate(train_dataset):
            if epoch<warmup:
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

            if (step+1)%10==0 or step+1==NUM_STEP_PER_EPOCH:
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

        if plot_every_num_epoch is not None and (epoch%plot_every_num_epoch==0 or epoch==NUM_EPOCH-1):
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
