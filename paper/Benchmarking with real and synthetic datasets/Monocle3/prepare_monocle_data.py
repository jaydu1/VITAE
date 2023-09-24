import scanpy as sc
import numpy as np
import pandas as pd
import os
from scipy import sparse

import sys
sys.path.append("../../VITAE")

import importlib

import VITAE

# Nature Cortex
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

# Mouse

from VITAE.utils import load_data
mouse = load_data(path = "../data", file_name = "mouse_brain_merged")

sc.pp.normalize_total(mouse, target_sum=1e4)
sc.pp.log1p(mouse)
sc.pp.highly_variable_genes(mouse, min_mean=0.0125, max_mean=3, min_disp=0.5)

sc.pp.scale(mouse, max_value=10)

dd.obs["Source"] = 2

# merge

temp_day = mouse.obs.index.values.copy()
temp_day = [x[:3] for x in temp_day]
mouse.obs["Day"] = temp_day

mouse.obs.columns = ["Clusters","S_Score","G2M_Score","Source","Day"]

dd = dd.concatenate(mouse,join="inner")

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

dd.obs["tidy_clusters"] = c.copy()

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
#here is where I change
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

a = dd.obs["tidy_clusters"].values.copy().astype(str)
a[(dd.obs["Day"].isin(["E14","E15","E16"])) & (a == "SCPN")] = "SCPN1"
dd.obs["Cluster2"] = a.copy()

dd.X = dd.X.astype(np.float16)
dd.obs["S_Score"] = dd.obs["S_Score"].astype(np.float16)
dd.obs["G2M_Score"] = dd.obs["G2M_Score"].astype(np.float16)

dd.var["highly_variable"] = dd.var["highly_variable-0"] | dd.var["highly_variable-1"]
dd.var = dd.var.drop(['highly_variable-0', 'means-0', 'dispersions-0', 'dispersions_norm-0', 'mean-0', 'std-0', 'highly_variable-1', 'means-1', 'dispersions-1', 'dispersions_norm-1', 'mean-1', 'std-1'],axis=1)
dd.obs = dd.obs.drop(['merge_day_18', 'merge_P1', 'batch', 'tidy_clusters', 'cov1', 'cov2', 'cov3', 'merge_18'],axis=1)

dd = dd[:,dd.var["highly_variable"]].copy()

## transpose because monocle need transpose here.
dd.T.write_h5ad("monocle_adata_forR_trans_highly.h5ad")




