# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, BatchNormalization
from tensorflow.keras.utils import Progbar
import tensorflow.keras.backend as K

from sklearn.mixture import GaussianMixture

import numpy as np
import pandas as pd
import umap

import matplotlib.pyplot as plt

from model import VariationalAutoEncoder
from utils import Early_Stopping


# ---------------------------------------------------------------------------- #
# Specify Parameters
# ---------------------------------------------------------------------------- #
tf.keras.backend.clear_session()

# Hyperparamter
BATCH_SIZE = 32
NUM_EPOCH_PRE = 300
NUM_STEP_PER_EPOCH = X.shape[0]//BATCH_SIZE+1
NUM_EPOCH = 1000

NUM_CLUSTER = len(np.unique(y)) 
DIM_LATENT = 16
DIM_ORIGIN = X.shape[1]

# Load data and preprocess.
 train_dataset = tf.data.Dataset.from_tensor_slices((X, X_normalized, Scale_factor))
train_dataset = train_dataset.shuffle(buffer_size=n).batch(BATCH_SIZE)

vae = VariationalAutoEncoder(NUM_CLUSTER, DIM_ORIGIN, [128,32], DIM_LATENT,
                             data_type='non-UMI')


# ---------------------------------------------------------------------------- #
# Pretrain for optimizing encoder and decoder networks.
# ---------------------------------------------------------------------------- #
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss_metric = tf.keras.metrics.Mean()
mae_loss_fn = tf.keras.losses.MeanAbsoluteError()
early_stopping = Early_Stopping(patience=10, tolerance=1e-3)
for epoch in range(NUM_EPOCH_PRE):
    progbar = Progbar(NUM_STEP_PER_EPOCH)
    
    print('Pretrain - Start of epoch %d' % (epoch,))

    # Iterate over the batches of the dataset.
    for step, (x_batch, x_norm_batch, s) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            reconstructed = vae(x_norm_batch, x_batch, s, pre_train=True)
            # Compute reconstruction loss
            loss = tf.reduce_sum(vae.losses[0]) # mae_loss_fn(x_batch, reconstructed)
            
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
vae.save_weights('/content/drive/My Drive/Data/pre_train.checkpoint')


# ---------------------------------------------------------------------------- #
# Initialize paramters of GMM
# ---------------------------------------------------------------------------- #
_, _, z = vae.encoder(X_normalized)
gmm = GaussianMixture(n_components=NUM_CLUSTER, covariance_type='diag')
gmm.fit(z)
# acc_0 = np.mean(gmm.predict(z).ravel() == y.ravel())
means_0 = gmm.means_
covs_0 = gmm.covariances_


pred_c = gmm.predict(z)
n_states = int((NUM_CLUSTER+1)*NUM_CLUSTER/2)
pi = np.zeros((1,n_states))
cluster_center = [int((NUM_CLUSTER+(1-i)/2)*i) for i in range(NUM_CLUSTER)]
for i in range(NUM_CLUSTER):
    pi[0,cluster_center[i]] = np.sum(pred_c==i)
pi = pi / np.sum(pi)

vae.GMM.initialize(means_0.T, covs_0, pi)
# print(acc_0)

pi,mu,c,w,var_w,z,proj_z = vae(X_normalized, inference=True)

fit = umap.UMAP()
uz = fit.fit_transform(z)
u = fit.transform(mu.numpy().T)
plt.figure(figsize=(20,10))
ax = plt.subplot(111)
scatter = plt.scatter(uz[:,0], uz[:,1], c=c, s=0.1)
ax.set_title('Prediction')        
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
ax.add_artist(legend1)
cluster_center = [(NUM_CLUSTER+(1-i)/2)*i for i in range(NUM_CLUSTER)]  
plt.scatter(u[:,0], u[:,1], c=cluster_center, s=200, marker='s')
plt.show()


# ---------------------------------------------------------------------------- #
# Training together.
# ---------------------------------------------------------------------------- #
optimizer = tf.keras.optimizers.Adam(1e-4)
loss_total = tf.keras.metrics.Mean()
loss_neg_E_nb = tf.keras.metrics.Mean()
loss_neg_E_pz = tf.keras.metrics.Mean()
loss_E_qzx = tf.keras.metrics.Mean()
early_stopping = Early_Stopping(patience=10, tolerance=1e-4)
for epoch in range(NUM_EPOCH):
    print('Start of epoch %d' % (epoch,))
    progbar = Progbar(NUM_STEP_PER_EPOCH)
    
    # Iterate over the batches of the dataset.
    for step, (x_batch, x_norm_batch, x_scale_factor) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            reconstructed = vae(x_norm_batch, x_batch, x_scale_factor)
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
        pi,mu,c,w,var_w,z,proj_z = vae(X_normalized, inference=True)

        fit = umap.UMAP()
        uz = fit.fit_transform(z)
        u = fit.transform(mu.numpy().T)

        plt.figure(figsize=(8,4))
        ax = plt.subplot(121)
        plt.scatter(uz[:,0], uz[:,1], c=y, s=2)
        ax.set_title('Ground Truth')

        ax = plt.subplot(122)
        scatter = plt.scatter(uz[:,0], uz[:,1], c=c, s=2)
        ax.set_title('Prediction')        
        # legend1 = ax.legend(*scatter.legend_elements(),
        #                     loc="lower left", title="Classes")
        # ax.add_artist(legend1)
        cluster_center = [(NUM_CLUSTER+(1-i)/2)*i for i in range(NUM_CLUSTER)]        
        plt.scatter(u[:,0], u[:,1], c=cluster_center, s=60, marker='s')
        plt.savefig('/content/drive/My Drive/Data/%d.png'%epoch, dpi=300)
        plt.show()
        vae.save_weights('/content/drive/My Drive/Data/train.checkpoint')
        
vae.save_weights('/content/drive/My Drive/Data/train.checkpoint')
