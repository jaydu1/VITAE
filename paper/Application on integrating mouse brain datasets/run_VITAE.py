import scanpy as sc
import numpy as np
import pandas as pd
import os
from scipy import sparse
import sys
import importlib
import VITAE
import seaborn

import random
import tensorflow as tf

## Preprocess for dataset from "Molecular logic of cellular diversification in the mouse cerebral cortex"
## One can download the dataset and meta data through the url:
## https://singlecell.broadinstitute.org/single_cell/study/SCP1290/molecular-logic-of-cellular-diversification-in-the-mammalian-cerebral-cortex
## transoform.h5ad file we used means data has been normalized to 10000 and done log(x+1) tranformation.


###################################### Preprocess Di Bella's dataset #################################################
dd=sc.read("transform.h5ad")
metadata = pd.read_csv('metaData_scDevSC.txt', delimiter='\t', index_col = 0)
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

day18 = np.array([np.nan] * dd.shape[0]).astype(object)
day18[(dd.obs["Day"] == "E18").values] = "E18"
day18[(dd.obs["Day"] == "E18_S1").values] = "E18_S1"
day18[(dd.obs["Day"] == "E18_S3").values] = "E18_S3"
dd.obs["merge_day_18"] = day18

dayP1 = np.array([np.nan] * dd.shape[0]).astype(object)
dayP1[dd.obs["Day"] == "P1"] = "P1"
dayP1[dd.obs["Day"] == "P1_S1"] = "P1_S1"
dd.obs["merge_P1"] = dayP1

dd.obs["merge_day_18"] = dd.obs["merge_day_18"].astype("category")
dd.obs["merge_P1"] = dd.obs["merge_P1"].astype("category")

dd.obs["Source"] = "Di Bella"



###################################### Preprocess Di Bella's dataset #################################################
mouse = load_data(path='data/',file_name="mouse_brain_merged")
sc.pp.normalize_total(mouse, target_sum=1e4)
sc.pp.log1p(mouse)
sc.pp.highly_variable_genes(mouse, min_mean=0.0125, max_mean=3, min_disp=0.5)
sc.pp.scale(mouse, max_value=10)
sc.tl.pca(mouse, n_comps = 64)
sc.pp.neighbors(mouse, n_neighbors = 20)
sc.tl.umap(mouse)

mouse_source = np.repeat("Yuzwa",mouse.shape[0])
mouse_source[mouse.obs["covariate_2"]==0] = "Ruan"
mouse.obs["Source"] = mouse_source
sc.pl.umap(mouse,color="Source",save = "_mouse_Source.pdf")

mouse_day = mouse.obs.index.values.copy()
mouse_day = [x[:3] for x in mouse_day]
mouse.obs["Day"] = mouse_day
mouse.obs.rename(columns={"grouping": "Clusters", "covariate_0": "S_Score",
                     "covariate_1":"G2M_Score"},inplace = True)



###################################### Merge two detasets  #################################################

dd = dd.concatenate(mouse,join="inner")
## define highly_variable in merged dataset
dd.var["highly_variable"] = dd.var["highly_variable-0"].astype(bool) | dd.var["highly_variable-1"].astype(bool)

## union different name of cell types in two datasets
group_dict = {"Immature Neuron" : "Immature neurons",
             "NEC":"Apical progenitors",
             "RGC":"Apical progenitors",
              "Layer I":"Cajal Retzius cells",
             "OPC":"Oligodendrocytes",
             "Interneurons":"Interneurons",
             "Endothelial Cell":"Endothelial cells",
             "Microglia":"Microglia",
             "Pericyte":"Pericytes",
             "Intermediate progenitors":"IPC"}

c = dd.obs["Clusters"].values.copy()
c = [x if group_dict.get(x) == None else group_dict.get(x) for x in c]
dd.obs["tidy_clusters"] = c.copy()

## Add covatiates for different source
a = np.zeros(dd.shape[0])
a[dd.obs["Source"] == "Ruan"] = 1
dd.obs["cov1"] = a

a = np.zeros(dd.shape[0])
a[dd.obs["Source"] == "Yuzwa"] = 1
dd.obs["cov2"] = a

a = np.zeros(dd.shape[0])
a[dd.obs["Source"] == "Di Bella"] = 1
dd.obs["cov3"] = a

a = np.array([np.nan] * dd.shape[0])
a[dd.obs["Day"] == "E18"] = 1
a[dd.obs["Day"] == "E18_S1"] = 2
a[dd.obs["Day"] == "E18_S3"] = 3
dd.obs["merge_18"] = a


a = np.array([np.nan] * dd.shape[0])
a[dd.obs["Day"] == "P1"] = 1
a[dd.obs["Day"] == "P1_S1"] = 2
dd.obs["merge_P1"] = a

dd.obs["merge_18"] = dd.obs["merge_18"].astype("category")
dd.obs["merge_P1"] = dd.obs["merge_P1"].astype("category")

a = np.array([np.nan] * dd.shape[0]).astype(object)
a[dd.obs["merge_P1"] == 1] = "P1"
a[dd.obs["merge_P1"] == 2] = "P1_S1"
dd.obs["merge_day_P1"] = a

a = dd.obs["tidy_clusters"].values.copy().astype(str)
a[(dd.obs["Day"].isin(["E14","E15","E16"])) & (a == "SCPN")] = "SCPN1"
dd.obs["Cluster2"] = a.copy()

###################################### Naive Merge  #################################################
## See what happened if we directly merge them ##
dd.obs["Source"] = dd.obs["Source"].astype(pd.CategoricalDtype(
                  ['Ruan', 'Yuzwa','Di Bella'], ordered=True))

sc.tl.pca(dd, n_comps = 64)
sc.pp.neighbors(dd, n_neighbors = 20)
sc.tl.umap(dd)
seaborn.set(rc={'figure.figsize':(6,6)},style = "white")
sc.pl.umap(dd,color="Source",save="_naive_Source.pdf",size = 5,palette= ["#F8766D","#00BFC4","#E5E8E8"],alpha = 0.5,na_color="#E5E8E8")

color_ggplot=["#F8766D","#E38900","#C49A00","#99A800","#53B400","#00BC56","#00C094","#00BFC4","#00B6EB","#06A4FF","#A58AFF","#DF70F8","#FB61D7","#FF66A8"]
seaborn.set(rc={'figure.figsize':(5,5)},style = "white")
sc.pl.umap(dd,color="Day",size = 2, alpha=0.4,palette=color_ggplot,save="_naive_merge_day.pdf")


###################################### Train VITAE  #################################################


seed = 400
tf.keras.backend.clear_session()
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

model = VITAE.VITAE(adata = dd,
    #        adata_layer_counts = 'counts',
            hidden_layers = [48, 24],
            latent_space_dim = 16,
            model_type = 'Gaussian',covariates = ['S_Score', 'G2M_Score',"cov1","cov2"],npc = 96)


model.pre_train(learning_rate = 1e-3,early_stopping_tolerance = 0.01, early_stopping_relative = True,L=1,phi=1,verbose=False,gamma = 1)
model.init_latent_space(cluster_label= 'Cluster2',ratio_prune=0.45)
model.train(beta = 1, learning_rate = 1e-3, early_stopping_tolerance = 0.01, early_stopping_relative = True,gamma=0,phi=1, verbose=False)
model.posterior_estimation(batch_size=32, L=10)
model.save_model(path_to_file="model.checkpoint")
