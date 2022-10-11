import scanpy as sc
import numpy as np
import pandas as pd
import os
from scipy import sparse

import sys

import importlib

## should be the latest version
import VITAE

dd=sc.read("/project/jingshuw/trajectory_analysis/data/cortex_mouse_brain_merge/inner_tidy_merge.h5ad")
dd.var["highly_variable"] = dd.var["highly_variable-0"].astype(bool) | dd.var["highly_variable-1"].astype(bool)

## tidy cluster IPC should be merged into in
a = dd.obs["tidy_clusters"].values.copy().astype(str)
a[a == "Intermediate progenitors"] = "IPC"
dd.obs["tidy_clusters1"] = a
dd.obs["tidy_clusters1"] = dd.obs["tidy_clusters1"].astype("category")

sc.tl.pca(dd, n_comps = 64)
sc.pp.neighbors(dd, n_neighbors = 20)
sc.tl.umap(dd)
dd.obs["Source"] = dd.obs["Source"].astype("category")

sc.set_figure_params(figsize=(10, 6), dpi = 80)
sc.pl.umap(dd, color=['Source'],  size = 5)
sc.pl.umap(dd, color=['tidy_clusters'],  size = 5)
sc.pl.umap(dd, color=['tidy_clusters1'],  size = 5)
sc.pl.umap(dd, color=['tidy_days'],  size = 5)

a = np.zeros(dd.shape[0])
a[dd.obs["Source"] == 0] = 1
dd.obs["cov1"] = a

a = np.zeros(dd.shape[0])
a[dd.obs["Source"] == 1] = 1
dd.obs["cov2"] = a

a = np.zeros(dd.shape[0])
a[dd.obs["Source"] == 2] = 1
dd.obs["cov3"] = a

a = np.array([np.nan] * dd.shape[0])
a[dd.obs["tidy_days"] == "E18"] = 1
a[dd.obs["tidy_days"] == "E18_S1"] = 2
a[dd.obs["tidy_days"] == "E18_S3"] = 3
dd.obs["merge_18"] = a

a = np.array([np.nan] * dd.shape[0])
a[dd.obs["tidy_days"] == "P1"] = 1
a[dd.obs["tidy_days"] == "P1_S1"] = 2
dd.obs["merge_P1"] = a

dd.obs["merge_18"] = dd.obs["merge_18"].astype("category")
dd.obs["merge_P1"] = dd.obs["merge_P1"].astype("category")

a = np.zeros(dd.shape[0])
a[dd.obs["tidy_days"].isin(['E10', 'E11', 'E12', 'E13', 'E14', 'E15', 'E16'])] = 1
dd.obs["day_cut"] = a

a = dd.obs["merge_18"].values.copy().astype(float)
a[a==0] = np.nan
dd.obs["merge_18_color"] = a
dd.obs["merge_18_color"] = dd.obs["merge_18_color"].astype("category")

a = dd.obs["merge_P1"].values.copy().astype(float)
a[a==0] = np.nan
dd.obs["merge_P1_color"] = a
dd.obs["merge_P1_color"] = dd.obs["merge_P1_color"].astype("category")

sc.pl.umap(dd, color=['merge_18_color',"merge_P1_color"],  size = 20)

import random
import tensorflow as tf

seed = 400
tf.keras.backend.clear_session()
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

model = VITAE.VITAE(adata = dd,
    #        adata_layer_counts = 'counts',
            hidden_layers = [32, 16],
            latent_space_dim = 16,
            model_type = 'Gaussian',covariates = ['S_Score', 'G2M_Score',"cov1","cov2"],npc = 96,conditions=["merge_18"])

model.pre_train(learning_rate = 1e-3,early_stopping_tolerance = 0.01, early_stopping_relative = True,L=1,phi=1,verbose=False,gamma = 1)

model.init_latent_space(cluster_label= 'tidy_clusters1',ratio_prune=0.4)

model.visualize_latent(color=['tidy_clusters1', 'tidy_days'],legend_loc = 'on data', size = 10)
import seaborn
seaborn.set(rc={'figure.figsize':(15,12)},style = "white")
model.visualize_latent(color=['S_Score', 'G2M_Score',
    "Eomes", "Satb2", "Cux2", "Bcl11b", "Tle4", "Apoe", "Dlx2", "Pdgfra"], method = "UMAP",size = 8)

model.visualize_latent(color=["merge_18_color","merge_P1_color"], palette = "tab20b",size= 40)

model.train(beta = 1, learning_rate = 1e-3, early_stopping_tolerance = 0.01, early_stopping_relative = True,gamma=1,phi=1, verbose=False)

model.visualize_latent(color=['tidy_clusters1', 'tidy_days'],legend_loc = 'on data', size = 10)
model.visualize_latent(color=['S_Score', 'G2M_Score',
    "Eomes", "Satb2", "Cux2", "Bcl11b", "Tle4", "Apoe", "Dlx2", "Pdgfra"], method = "UMAP",size = 8)

model.visualize_latent(color=["merge_18_color","merge_P1_color"], palette = "tab20b",size= 40)

model.posterior_estimation(batch_size=32, L=10)
model.visualize_latent(color=['vitae_new_clustering'], method = "UMAP",size = 8)

model.infer_backbone(cutoff = 0, no_loop = True, visualize = True,method = "raw_map")
a = list(model.backbone.edges(data = True))
a = [x[2]["weight"] for x in a]
a = np.sort(a)[-15] - 1e-10
model.infer_backbone(cutoff = a, no_loop = True, visualize = True,method = "raw_map")

model.infer_trajectory("Apical progenitors")


