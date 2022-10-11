import scanpy as sc
import numpy as np
import pandas as pd
import os
from scipy import sparse

import sys

import importlib

import VITAE

dd=sc.read("/project/jingshuw/trajectory_analysis/data/Nature_Cortex/transform.h5ad")

data_loc="/project/jingshuw/trajectory_analysis/data/Nature_Cortex/"
metadata = pd.read_csv(data_loc + 'metaData_scDevSC.txt', delimiter='\t', index_col = 0)

dd.obs['Day'] = metadata['orig_ident'][1:metadata.shape[0]]
dd.obs['Clusters'] = metadata['New_cellType'][1:metadata.shape[0]]
dd.obs['Clusters'] = pd.Categorical(dd.obs['Clusters'], categories = [
    'Apical progenitors', 'Intermediate progenitors', 'Migrating neurons',
    'Immature neurons', 'Cajal Retzius cells', 'CThPN', 'SCPN',
    'NP', 'Layer 6b', 'Layer 4', 'DL CPN', 'DL_CPN_1', 'DL_CPN_2', 'UL CPN',
    'Interneurons', 'Astrocytes', 'Oligodendrocytes', 'Microglia',
    'Cycling glial cells', 'Ependymocytes', 'Endothelial cells',
    'VLMC', 'Pericytes','Red blood cells', 'Doublet', 'Low quality cells'
    ], ordered = True)


dd.obs['S_Score'] = pd.to_numeric(metadata['S_Score'][1:metadata.shape[0]])
dd.obs['G2M_Score'] = pd.to_numeric(metadata['G2M_Score'][1:metadata.shape[0]])

dd = dd[dd.obs['Clusters'].isin(['Doublet', 'Low quality cells']) == False]

dd.obs.index=dd.obs.index.tolist()
dd.obs['Day']=dd.obs['Day'].tolist()
dd.obs['Clusters']=dd.obs['Clusters'].tolist()
dd.obs['S_Score']=dd.obs['S_Score'].tolist()
dd.obs['G2M_Score']=dd.obs['G2M_Score'].tolist()

sc.pp.highly_variable_genes(dd, flavor = "seurat")
sc.pp.scale(dd, max_value=10)

sc.tl.pca(dd, n_comps = 64)
#dd.obsm['X_pca'][:, 1] *= -1  # to match Seurat
#sc.pl.pca_scatter(dd, color = ['Clusters', 'Day'])
sc.pp.neighbors(dd, n_neighbors = 20)

sc.set_figure_params(figsize=(10, 6), dpi = 80)

merge = np.array([np.nan] * dd.shape[0])
merge[(dd.obs["Day"] == "E18_S1").values] = 1
merge[(dd.obs["Day"] == "E18_S3").values] = 2
dd.obs["merge_day_18"] = merge

merge = np.array([np.nan] * dd.shape[0])
merge[dd.obs["Day"] == "P1"] = 1
merge[dd.obs["Day"] == "P1_S1"] = 2
dd.obs["merge_P1"] = merge

dd.obs["merge_day_18"] = dd.obs["merge_day_18"].astype("category")
dd.obs["merge_P1"] = dd.obs["merge_P1"].astype("category")

sc.tl.umap(dd)
sc.pl.umap(dd, color=['Day'], palette = "Set2", size = 5)
sc.pl.umap(dd, color=['merge_day_18'], palette = "tab20", size = 5)
sc.pl.umap(dd, color=['merge_P1'], palette = "tab20", size = 5)

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
            model_type = 'Gaussian',covariates = ['S_Score', 'G2M_Score'],conditions=["merge_day_18","merge_P1"],npc = 96)

model.pre_train(learning_rate = 1e-3,early_stopping_tolerance = 0.01, early_stopping_relative = True,L=1,verbose=True,gamma = 1)

model.init_latent_space(cluster_label= 'Clusters', res = 0.6, ratio_prune = 0.5)

model.visualize_latent(color=['Clusters', 'Day'],legend_loc = 'on data', palette = "Set2",size = 10)
model.visualize_latent(color=['merge_day_18','merge_P1'], palette = "tab20",size= 8)
model.visualize_latent(color=['S_Score', 'G2M_Score',
    "Eomes", "Satb2", "Cux2", "Bcl11b", "Tle4", "Apoe", "Dlx2", "Pdgfra"], method = "UMAP",size = 8)

model.train(beta = 1, learning_rate = 1e-3, early_stopping_tolerance = 0.01, early_stopping_relative = True,gamma=1, verbose=True)

model.posterior_estimation(batch_size=32, L=10)
model.visualize_latent(color=['Clusters', 'Day'],legend_loc = 'on data', palette = "Set2",size = 10)
model.visualize_latent(color=['merge_day_18','merge_P1'], palette = "tab20",size= 8)
model.visualize_latent(color=['S_Score', 'G2M_Score',
    "Eomes", "Satb2", "Cux2", "Bcl11b", "Tle4", "Apoe", "Dlx2", "Pdgfra"], method = "UMAP",size = 8)

model.infer_backbone(cutoff = 0, no_loop = True, visualize = True,method = "raw_map")
a = list(model.backbone.edges(data = True))
a = [x[2]["weight"] for x in a]
a = np.sort(a)[-12] - 1e-10
model.infer_backbone(cutoff = a, no_loop = True, visualize = True,method = "raw_map")


