
import pandas as pd
import scanpy as sc
import numpy as np
import sys
sys.path.append(r"C:\Users\10270\Desktop\硕一上\Wang\Trajectory\Dev\GitVersion\VITAE")
import os as os

import VITAE
from VITAE.utils import load_data

import pandas as pd
import scanpy as sc

dd = load_data(path = "D:/BioData", file_name = "mouse_brain_merged")

sc.pp.normalize_total(dd, target_sum=1e4)
sc.pp.log1p(dd)
sc.pp.highly_variable_genes(dd, min_mean=0.0125, max_mean=3, min_disp=0.5)


sc.pp.scale(dd, max_value=10)

sc.tl.pca(dd, svd_solver='arpack')
sc.pp.neighbors(dd, n_neighbors=10, n_pcs=40)

import random
import tensorflow as tf

tf.random.set_seed(
    seed = 1)
## Hyper parameters
npc = 64
model_type = "Gaussian"
hidden_layers = [32,16]
latent_space_dim = 8

clustering_res = 0.6
pi_prune_ratio = 0

n_posterior_samples = 10
beta = 1

dd.obs.columns = ['grouping', 'S_Score', 'G2M_Score', 'Source']

seed = 400
tf.keras.backend.clear_session()
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

model = VITAE.VITAE(adata = dd,
                 npc = npc, model_type = 'Gaussian',
                 hidden_layers = hidden_layers, latent_space_dim = latent_space_dim,
                 covariates = ['Source', 'S_Score', 'G2M_Score'])

model.pre_train(learning_rate=0.001,early_stopping_tolerance = 0.01, early_stopping_relative = True)

model.init_latent_space(cluster_label= 'grouping', ratio_prune = pi_prune_ratio)

model.visualize_latent(color=['grouping', 'Source', 'S_Score', 'G2M_Score', 'vitae_init_clustering', 'Eomes'], method = "UMAP",legend_loc = 'on data')
model.train(beta = 1, learning_rate = 1e-3, early_stopping_tolerance = 0.01, early_stopping_relative = True)
model.posterior_estimation(batch_size=32, L=n_posterior_samples)
model.visualize_latent(color=['grouping', 'Source',  'S_Score', 'G2M_Score', 'vitae_init_clustering', 'vitae_new_clustering', 'Eomes'], method = "UMAP", legend_loc = 'on data')
model.visualize_latent(color=['vitae_init_clustering', 'vitae_new_clustering'], method = "UMAP", legend_loc = 'on data')
model.infer_backbone(cutoff = 0.0, no_loop = True, visualize = True,method = "modified_map")
a = list(model.backbone.edges(data = True))
a = [x[2]["weight"] for x in a]
a = np.sort(a)[-7] - 1e-10
model.infer_backbone(cutoff = a, no_loop = True, visualize = True,method = "modified_map")

import seaborn
seaborn.set(rc={'figure.figsize':(15,12)},style = "white")
model.infer_backbone(cutoff = 0, no_loop = True, visualize = True,method = "raw_map")

a = list(model.backbone.edges(data = True))
a = [x[2]["weight"] for x in a]
a = np.sort(a)[-8] - 1e-10
model.infer_backbone(cutoff = a, no_loop = True, visualize = True,method = "raw_map")

