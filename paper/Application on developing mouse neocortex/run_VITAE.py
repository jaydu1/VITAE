# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
import h5py
import random
import os
import scanpy
from VITAE import VITAE, get_igraph, leidenalg_igraph, load_data
import random
import tensorflow as tf
import seaborn
seaborn.set(rc={'figure.figsize':(15,12)},style = "white")

file_name = 'mouse_brain_merged'
dd = load_data(path='data/',
                 file_name=file_name)
dd.obs.columns = ['grouping', 'S_Score', 'G2M_Score', 'Source']

sc.pp.normalize_total(dd, target_sum=1e4)
sc.pp.log1p(dd)
sc.pp.highly_variable_genes(dd, min_mean=0.0125, max_mean=3, min_disp=0.5)

sc.pp.scale(dd, max_value=10)

sc.tl.pca(dd, svd_solver='arpack')
sc.pp.neighbors(dd, n_neighbors=10, n_pcs=40)

## Hyper parameters
npc = 64
model_type = "Gaussian"
hidden_layers = [32,16]
latent_space_dim = 8
pi_prune_ratio = 0
n_posterior_samples = 10
beta = 1


seed = 400
tf.keras.backend.clear_session()
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

model = VITAE.VITAE(adata = dd,
                 npc = npc, model_type = 'Gaussian',
                 hidden_layers = hidden_layers, latent_space_dim = latent_space_dim,
                 covariates = ['Source', 'S_Score', 'G2M_Score'])

# save model weight with parameters in the latent space and inference results (embedding, posterior estimations)
# model.save_model(path_to_file='../weight/tutorial_mouse_brain_merged/mouse.checkpoint')

# if skip the training process, one can directly load weights and go straight to model.infer_backbone.
# load model weight
# model.load_model(path_to_file='../weight/tutorial_mouse_brain_merged/mouse.checkpoint', load_labels=True)

model.pre_train(learning_rate=0.001,early_stopping_tolerance = 0.01, early_stopping_relative = True)

model.init_latent_space(cluster_label= 'grouping', ratio_prune = pi_prune_ratio)

model.visualize_latent(color=['grouping', 'Source', 'S_Score', 'G2M_Score', 'vitae_init_clustering', 'Eomes'], method = "UMAP",legend_loc = 'on data')
model.train(beta = 1, learning_rate = 1e-3, early_stopping_tolerance = 0.01, early_stopping_relative = True)

model.posterior_estimation(batch_size=32, L=n_posterior_samples)
model.visualize_latent(color=['grouping', 'Source',  'S_Score', 'G2M_Score', 'vitae_init_clustering', 'vitae_new_clustering', 'Eomes'], method = "UMAP", legend_loc = 'on data')
model.visualize_latent(color=['vitae_init_clustering', 'vitae_new_clustering'], method = "UMAP", legend_loc = 'on data')


model.infer_backbone(cutoff = 0, no_loop = True, visualize = True,method = "raw_map")
## remain 8 edges in the graph.
a = list(model.backbone.edges(data = True))
a = [x[2]["weight"] for x in a]
a = np.sort(a)[-8] - 1e-10
model.infer_backbone(cutoff = a, no_loop = True, visualize = True,method = "raw_map")

model.infer_trajectory(root="NEC")

