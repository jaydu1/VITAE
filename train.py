# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.utils import Progbar
import numpy as np
import umap
import matplotlib.pyplot as plt
from utils import Early_Stopping

def clear_session():
    tf.keras.backend.clear_session()
    return None

    
def warp_dataset(X_normalized, BATCH_SIZE, X=None, Scale_factor=None):
    if X is not None:
        train_dataset = tf.data.Dataset.from_tensor_slices((X, X_normalized, Scale_factor))
        train_dataset = train_dataset.shuffle(buffer_size = X.shape[0], reshuffle_each_iteration=True).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
        return train_dataset
    else:
        test_dataset = tf.data.Dataset.from_tensor_slices(X_normalized).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
        return test_dataset


def pre_train(train_dataset, vae, learning_rate, patience, tolerance, NUM_EPOCH_PRE, NUM_STEP_PER_EPOCH, L):
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    loss_metric = tf.keras.metrics.Mean()
    early_stopping = Early_Stopping(patience = patience, tolerance = tolerance)
    for epoch in range(NUM_EPOCH_PRE):
        progbar = Progbar(NUM_STEP_PER_EPOCH)
        
        print('Pretrain - Start of epoch %d' % (epoch,))

        # Iterate over the batches of the dataset.
        for step, (x_batch, x_norm_batch, x_scale_factor) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                _ = vae(x_norm_batch, x_batch, x_scale_factor, pre_train=True, L=L)
                # Compute reconstruction loss
                loss = tf.reduce_sum(vae.losses[0]) 
                
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


def plot_pre_train(vae, X_normalized, label):
    print('-------UMAP for latent space after preTrain:-------')
    z_mean, _,_ = vae.encoder(X_normalized)
    fit = umap.UMAP()
    u = fit.fit_transform(tf.concat((z_mean,tf.transpose(vae.GMM.mu)),axis=0))
    uz = u[:len(label),:]
    um = u[len(label):,:]
    plt.figure(figsize=(20,10))
    ax = plt.subplot(111)
    scatter = plt.scatter(uz[:,0], uz[:,1], c=label, alpha = 0.8, s=10)
    ax.set_title('Prediction')        
    legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
    ax.add_artist(legend1)
    plt.scatter(um[:,0], um[:,1], c='black', s=200, marker='s')
    plt.show()


def train(train_dataset, test_dataset, vae,
        learning_rate, patience, tolerance, NUM_EPOCH, NUM_STEP_PER_EPOCH, L,
        labels, weight, plot_every_num_epoch=None):
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss_total = tf.keras.metrics.Mean()
    loss_neg_E_nb = tf.keras.metrics.Mean()
    loss_neg_E_pz = tf.keras.metrics.Mean()
    loss_E_qzx = tf.keras.metrics.Mean()
    early_stopping = Early_Stopping(patience = patience, tolerance = tolerance)
    if weight is None:
        weight = np.ones(3, dtype=np.float32)
    else:
        weight = np.array(weight, dtype=np.float32)
    weight = tf.convert_to_tensor(weight)
    
    for epoch in range(NUM_EPOCH):
        print('Start of epoch %d' % (epoch,))
        progbar = Progbar(NUM_STEP_PER_EPOCH)
        
        # Iterate over the batches of the dataset.
        for step, (x_batch, x_norm_batch, x_scale_factor) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                _ = vae(x_norm_batch, x_batch, x_scale_factor, L=L)
                loss = tf.reduce_sum(vae.losses*weight)  
        
            grads = tape.gradient(loss, vae.trainable_weights,
                                  unconnected_gradients=tf.UnconnectedGradients.ZERO)
            optimizer.apply_gradients(zip(grads, vae.trainable_weights))
            loss_total(loss)
            loss_neg_E_nb(vae.losses[0])
            loss_neg_E_pz(vae.losses[1])
            loss_E_qzx(vae.losses[2])

            if (step+1)%10==0 or step+1==NUM_STEP_PER_EPOCH:
                progbar.update(step+1, [
                        ('loss_neg_E_nb'    ,   float(vae.losses[0])),
                        ('loss_neg_E_pz'    ,   float(vae.losses[1])),
                        ('loss_E_qzx   '    ,   float(vae.losses[2])),
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
            pi,mu,c,w,var_w,wc,var_wc,_,_,z_mean = vae.inference(test_dataset, 1)

            fit = umap.UMAP()
            u = fit.fit_transform(tf.concat((z_mean,tf.transpose(mu)),axis=0))
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
