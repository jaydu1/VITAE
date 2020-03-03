# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.utils import Progbar

from sklearn.mixture import GaussianMixture

import numpy as np
import umap

import matplotlib.pyplot as plt

from utils import Early_Stopping

def clear_session():
    tf.keras.backend.clear_session()
    return None


def warp_dataset(X, X_normalized, Scale_factor, BATCH_SIZE, data_type):
    train_dataset = tf.data.Dataset.from_tensor_slices((X, X_normalized, Scale_factor))
    train_dataset = train_dataset.shuffle(buffer_size = X.shape[0]).batch(BATCH_SIZE)
    return train_dataset


def pre_train(train_dataset, vae, learning_rate, patience, tolerance, NUM_EPOCH_PRE, NUM_STEP_PER_EPOCH):
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    loss_metric = tf.keras.metrics.Mean()
    early_stopping = Early_Stopping(patience = patience, tolerance = tolerance)
    for epoch in range(NUM_EPOCH_PRE):
        progbar = Progbar(NUM_STEP_PER_EPOCH)
        
        print('Pretrain - Start of epoch %d' % (epoch,))

        # Iterate over the batches of the dataset.
        for step, (x_batch, x_norm_batch, x_scale_factor) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                _ = vae(x_norm_batch, x_batch, x_scale_factor, pre_train=True)
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


def init_GMM(vae, X_normalized, NUM_CLUSTER):
    z_mean, _,_ = vae.encoder(X_normalized)
    gmm = GaussianMixture(n_components=NUM_CLUSTER, covariance_type='diag')
    gmm.fit(z_mean)
    means_0 = gmm.means_
    covs_0 = gmm.covariances_
    pred_c = gmm.predict(z_mean)
    n_states = int((NUM_CLUSTER+1)*NUM_CLUSTER/2)
    pi = np.zeros((1,n_states))
    cluster_center = [int((NUM_CLUSTER+(1-i)/2)*i) for i in range(NUM_CLUSTER)]
    for i in range(NUM_CLUSTER):
        pi[0, cluster_center[i]] = np.sum(pred_c == i)
    pi = pi / np.sum(pi)
    vae.GMM.initialize(means_0.T, covs_0, pi)
    print('GMM Initialization Done.')
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


def trainTogether(train_dataset, vae, learning_rate, patience, tolerance, NUM_EPOCH, NUM_STEP_PER_EPOCH, label, X_normalized):
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss_total = tf.keras.metrics.Mean()
    loss_neg_E_nb = tf.keras.metrics.Mean()
    loss_neg_E_pz = tf.keras.metrics.Mean()
    loss_E_qzx = tf.keras.metrics.Mean()
    early_stopping = Early_Stopping(patience = patience, tolerance = tolerance)
    for epoch in range(NUM_EPOCH):
        print('Start of epoch %d' % (epoch,))
        progbar = Progbar(NUM_STEP_PER_EPOCH)
        
        # Iterate over the batches of the dataset.
        for step, (x_batch, x_norm_batch, x_scale_factor) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                _ = vae(x_norm_batch, x_batch, x_scale_factor)
                loss = tf.reduce_sum(vae.losses)  

            grads = tape.gradient(loss, vae.trainable_weights,
                                  unconnected_gradients=tf.UnconnectedGradients.ZERO)
            optimizer.apply_gradients(zip(grads, vae.trainable_weights))
            #vae.GMM.normalize()
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

        if epoch%10==0 or epoch==NUM_EPOCH-1:        
            pi,mu,c,w,var_w,wc,var_wc,z_mean,proj_z = vae(X_normalized, inference=True)

            fit = umap.UMAP()
            u = fit.fit_transform(tf.concat((z_mean,tf.transpose(mu)),axis=0))
            uz = u[:len(label),:]
            um = u[len(label):,:]

            plt.figure(figsize=(16,6))
            ax = plt.subplot(121)
            plt.scatter(uz[:,0], uz[:,1], c = label, s = 2)
            ax.set_title('Ground Truth')

            ax = plt.subplot(122)
            plt.scatter(uz[:,0], uz[:,1], c = c, s = 2, alpha = 0.5)
            ax.set_title('Prediction')        
            # legend1 = ax.legend(*scatter.legend_elements(),
            #                     loc="lower left", title="Classes")
            # ax.add_artist(legend1)
            cluster_center = [(len(um)+(1-i)/2)*i for i in range(len(um))]        
            plt.scatter(um[:,0], um[:,1], c=cluster_center, s=100, marker='s')
            plt.savefig('%d.png'%epoch, dpi=300)
            plt.show()

    print('Training Done!')

    return vae
