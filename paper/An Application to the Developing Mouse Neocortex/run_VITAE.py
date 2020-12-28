# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
import h5py
import umap
import matplotlib.pyplot as plt
import matplotlib
from VITAE import VITAE, get_igraph, louvain_igraph, plot_clusters, load_data

file_name = 'mouse_brain_merged'
data = load_data(path='data/',
                 file_name=file_name)

tf.keras.backend.clear_session() 
model = VITAE()
model.get_data(data['count'],                 # count or expression matrix, (dense or sparse) numpy array 
               labels = data['grouping'],       # (optional) labels, which will be converted to string
               covariate = data['covariates'],#None,#,  # (optional) covariates
               gene_names = data['gene_names'], # (optional) gene names, which will be converted to string
               cell_names = data['cell_ids']                # (optional) cell names, which will be converted to string
              )
model.preprocess_data(K = 1e4,                # (optional) denominator of the scale factor for log-normalization (the default is 1e4)
                      gene_num = 2000,        # (optional) maximum number of influential genes to keep (the default is 2000)
                      data_type = 'Gaussian',      # (optional) data_type can be 'UMI', 'non-UMI' or 'Gaussian' (the default is 'UMI')
                      npc = 64                # (optional) number of PCs to keep if data_type='Gaussian' (the default is 64)
                     )

model.pre_train(learning_rate = 1e-3,        # (Optional) the initial learning rate for the Adam optimizer (the default is 1e-3).
                batch_size=256,              # (Optional) the batch size for pre-training (the default is 32). 
                L=1,                         # (Optional) the number of MC samples (the default is 1).
                num_epoch = 300,             # (Optional) the maximum number of epoches (the default is 300).                
                num_step_per_epoch = None,   # (Optional) the number of step per epoch, it will be inferred from number of cells and batch size if it is None (the default is None).
                early_stopping_tolerance=1,  # (Optional) the minimum change of loss to be considered as an improvement (the default is 1e-3).
                early_stopping_patience=5,   # (Optional) the maximum number of epoches if there is no improvement (the default is 10).
                early_stopping_warmup=0,     # (Optional) the number of warmup epoches (the default is 0).
                path_to_weights=None,         # (Optional) the path of weight file to be saved; not saving weight if None (the default is None).
                alpha=0.1,
                stratify=False
                ) 
# Get latent representations of X after pre-training
z = model.get_latent_z()
g = get_igraph(z)
labels = louvain_igraph(g, 0.6, random_state=0) # You can choose different resolution parameter to get satisfied clustering results.
plot_clusters(embed_z, labels)
print(np.unique(labels)) 

NUM_CLUSTER = len(np.unique(labels))
n_states = int((NUM_CLUSTER+1)*NUM_CLUSTER/2)
cluster_center = [int((NUM_CLUSTER+(1-i)/2)*i) for i in range(NUM_CLUSTER)]
mu = np.zeros((z.shape[1],NUM_CLUSTER))
for i,l in enumerate(np.unique(labels)):
    mu[:,i] = np.mean(z[labels==l], axis=0)
model.init_latent_space(
                NUM_CLUSTER,                     # numebr of clusters
               cluster_labels=labels,           # (optional) names of the clustering labels for plotting
               mu=mu,                           # (optional) initial mean
               log_pi=None                    # (optional) initial pi
               )                   
model.train(learning_rate = 1e-3,        # (Optional) the initial learning rate for the Adam optimizer (the default is 1e-3).
            batch_size=256,              # (Optional) the batch size for pre-training (the default is 32). 
            L=1,                         # (Optional) the number of MC samples (the default is 1).
            alpha=0.1,
            beta=2,
            num_epoch = 300,             # (Optional) the maximum number of epoches (the default is 300).                
            num_step_per_epoch = None,   # (Optional) the number of step per epoch, it will be inferred from number of cells and batch size if it is None (the default is None).
            early_stopping_tolerance=1,  # (Optional) the minimum change of loss to be considered as an improvement (the default is 1e-3).
            early_stopping_patience=5,   # (Optional) the maximum number of epoches if there is no improvement (the default is 10).
            early_stopping_warmup=5,     # (Optional) the number of warmup epoches (the default is 0).
            plot_every_num_epoch=None,
            path_to_weights=None,         # (Optional) the path of weight file to be saved; not saving weight if None (the default is None). 
            stratify=False
            )

# model.save_model(path_to_file='../../weight/mouse_brain_merged/mouse_brain_merged_Gaussian_train.checkpoint')
model.load_model(path_to_file='../../weight/mouse_brain_merged/mouse_brain_merged_Gaussian_train.checkpoint', load_labels=True)

import networkx as nx
G = model.comp_inference_score(method='modified_map',  # 'mean', 'modified_mean', 'map', and 'modified_map'
                               thres=0.5,              # (Optional) threshold for compute the conditional probablity, only applies to 'mean' and 'modified_mean'
                               no_loop=True            # if no_loop=True, then find the maximum spanning tree
                               )           
begin_node_pred = model.select_root(days, 'sum')
modified_G, modified_w,pseudotime = model.infer_trajectory(init_node=begin_node_pred,  # initial node for computing pseudotime.
                       cutoff=0.16              # (Optional) cutoff score for edges (the default is 0.01).
                       )  



id_branches = [((modified_w[:,5] > 0.0)&(modified_w[:,7] > 0.0)) | \
    ((modified_w[:,7] > 0.0)&(modified_w[:,0] > 0.0)) | \
    ((modified_w[:,0] > 0.0)&(modified_w[:,11] > 0.0)) | \
    (modified_w[:,5] > 0.99) | \
    (modified_w[:,7] > 0.99) | \
    (modified_w[:,0] > 0.99) | \
    (modified_w[:,11] > 0.99),
((modified_w[:,7] > 0.0)&(modified_w[:,4] > 0.0)) | \
    ((modified_w[:,4] > 0.0)&(modified_w[:,6] > 0.0)) | \
    ((modified_w[:,6] > 0.0)&(modified_w[:,1] > 0.0)) | \
    (modified_w[:,7] > 0.99) | \
    (modified_w[:,4] > 0.99) | \
    (modified_w[:,6] > 0.99) | \
    (modified_w[:,1] > 0.99)   ] 
branch_names = ['branch 5-7-0-11', 'bracnh 7-4-6-1']
for i in range(2):
    id_branch = id_branches[i]
    branch_name = branch_names[i]
    model.set_cell_subset(
        model.cell_names[id_branch]
        )    
    with h5py.File('result/%s/expression.h5'%branch_name, 'w') as f:
        f.create_dataset('expression', 
                        data=model.expression[model.selected_cell_subset_id,:], compression="gzip", compression_opts=9
                        )
        f.create_dataset('gene_names', 
                        data=model.gene_names.astype('bytes'), compression="gzip", compression_opts=9
                        )
        f.create_dataset('cell_ids', 
                        data=model.selected_cell_subset.astype('bytes'), compression="gzip", compression_opts=9
                        )                        
    pd.DataFrame(pseudotime[id_branch], 
                index=model.selected_cell_subset,
                columns=['pseudotime']
                ).to_csv('result/%s/pseudotime.csv'%branch_name)
    pd.DataFrame(data['covariates'][id_branch,:], 
                index=model.selected_cell_subset,
                columns=['S_score','G2M_score','id_data']
                ).to_csv('result/%s/covariate.csv'%branch_name)
    pd.DataFrame(np.array([i[:3] for i in data['cell_ids']])[id_branch], 
                index=model.selected_cell_subset,
                columns=['cell_day']
                ).to_csv('result/%s/cell_day.csv'%branch_name)                       