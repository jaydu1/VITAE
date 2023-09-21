# -*- coding: utf-8 -*-
import pandas as pd
import scanpy as sc
import numpy as np
import os
import sys#; sys.path.insert(0, '../..')
sys.path.append('/home/jinhongd/jingshu/VITAE-mm-pi-mm_added_pi')
print(os.getcwd())
import VITAE
from VITAE.utils import load_data, reset_random_seeds
import tensorflow as tf
import random
import h5py
import matplotlib.pyplot as plt

path = 'paper/An Application on scRNA and scATAC datasets/'
os.makedirs(path, exist_ok=True)

# Load data 
adata_atac = load_data('data', 'human_hematopoiesis_scATAC')
adata_rna = load_data('data', 'human_hematopoiesis_scRNA')

celltype_exclude = ['CD4.M', 'CD4.N', 'CD8.CM', 'CD8.EM', 'CD8.N', 'NK', 'Plasma', 'cDC', 'CD16.Mono']
adata_atac = adata_atac[~np.isin(adata_atac.obs['grouping'], celltype_exclude),:]
adata_rna = adata_rna[~np.isin(adata_rna.obs['grouping'], celltype_exclude),:]

# preprocess
hvg = []
for adata in [adata_atac, adata_rna]:  
    dd = adata.copy()    
    sc.pp.normalize_total(dd, target_sum=1e4)
    sc.pp.log1p(dd)
    hvg.append(
        sc.pp.highly_variable_genes(dd, inplace=False))
id_bool_genes = (hvg[0]['highly_variable']|hvg[1]['highly_variable']).values
adata_atac = adata_atac[:,id_bool_genes]
adata_rna = adata_rna[:,id_bool_genes]

adata = adata_rna.concatenate(adata_atac, index_unique=None)
adata.obs['id_dataset'] = adata.obs['batch'].cat.rename_categories({'0': 'scRNA', '1': 'scATAC'})
adata.obs['location'] = adata.obs['covariate_0'].str.split('_', expand=True).iloc[:,0]
adata.obs['location'] = adata.obs['location'].astype('category')
adata.obs['tissue'] = adata.obs['covariate_0'].str.split('_', expand=True).iloc[:,1].str.split('T', expand=True).iloc[:,1]
adata.obs['tissue'] = adata.obs['tissue'].astype('category')
adata.obs['day'] = adata.obs['covariate_0'].str.split('_', expand=True).iloc[:,1].str.split('T', expand=True).iloc[:,0]
adata.obs['day'] = adata.obs['day'].astype('category')

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.scale(adata, max_value=10)

# merge small celltypes
dict_merge = {
    'Baso.Eryth':['Early.Baso','Early.Eryth', 'Late.Eryth'],
    'GMP':['GMP', 'GMP.Neut']
             }
merged_groupings = adata.obs['grouping'].astype(str).values
for key in dict_merge.keys():
    merged_groupings[
        np.isin(merged_groupings, dict_merge[key])] = key
adata.obs["grouping"] = merged_groupings
adata.obs["grouping"] = adata.obs["grouping"].astype("category")

cond_group = np.unique(merged_groupings).astype(str)
for group in cond_group:
    col_name = 'cond_'+group    
    adata.obs[col_name] = np.where(merged_groupings==group, adata.obs['id_dataset'].values, np.nan)
cond = np.char.add('cond_', cond_group)
adata.obs[cond] = adata.obs[cond].astype("category")


# run the model
reset_random_seeds(400)
tf.keras.backend.clear_session() 
model = VITAE.VITAE(adata = adata, covariates=['id_dataset'], conditions=cond,
                    model_type = 'Gaussian', 
                    npc=128, hidden_layers = [32,16], latent_space_dim=8)


model.pre_train(gamma=0.6, phi=0.6, early_stopping_tolerance = 0.01, early_stopping_relative=True) 
model.visualize_latent(color = ['id_dataset','grouping','location','tissue','day'], method = "UMAP")
plt.savefig(path+"fig_pretrain.png", bbox_inches="tight")

model.init_latent_space(cluster_label='grouping', ratio_prune=0.5)

model.train(gamma=1., phi=1., early_stopping_tolerance = 0.01, early_stopping_relative=True)
model.posterior_estimation()
model.visualize_latent(color = ['vitae_new_clustering','grouping','id_dataset','location','tissue','day'], method = "UMAP")
plt.savefig(path+"fig_train.png", bbox_inches="tight")
model.infer_backbone(method = "modified_map")
model.plot_backbone(color='grouping')
plt.savefig(path+"fig_traj_modified_map.png", bbox_inches="tight")

model.infer_backbone(method = "raw_map")
model.plot_backbone(color='grouping')
plt.savefig(path+"fig_traj_raw_map.png", bbox_inches="tight")

model.save_model(
    path_to_file=path+'weight/model_inference.checkpoint',
    save_adata=True
)
