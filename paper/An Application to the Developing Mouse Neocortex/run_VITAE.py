# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
import h5py
import random
import os
from VITAE import VITAE, get_igraph, leidenalg_igraph, load_data

file_name = 'mouse_brain_merged'
data = load_data(path='data/',
                 file_name=file_name)

seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

model = VITAE()
model.get_data(data['count'],                 # count or expression matrix, (dense or sparse) numpy array 
               labels = data['grouping'],       # (optional) labels, which will be converted to string
               covariate = data['covariates'],#None,#,  # (optional) covariates
               gene_names = data['gene_names'], # (optional) gene names, which will be converted to string
               cell_names = data['cell_ids']                # (optional) cell names, which will be converted to string
              )
model.preprocess_data(gene_num = 2000,        # (optional) maximum number of influential genes to keep (the default is 2000)
                      data_type = 'Gaussian',      # (optional) data_type can be 'UMI', 'non-UMI' or 'Gaussian' (the default is 'UMI')
                      npc = 64                # (optional) number of PCs to keep if data_type='Gaussian' (the default is 64)
                     )

model.build_model(dim_latent = 8,         # The size of the latent dimension
                  dimensions=[32]      # The size of each layer in the encoder between the input layer and the 
                                          # latent layer. The size of each layer in the decoder is the reverse.
                  )

model.pre_train(test_size = 0.1,             # (Optional) the proportion or size of the test set.
                random_state = seed,         # (Optional) the random state of data splitting.
                batch_size=256,              # (Optional) the batch size for pre-training (the default is 32). 
                alpha=0.10,                  # (Optional) the value of alpha in [0,1] to encourage covariate adjustment. Not used if there is no covariates.                
                num_epoch = 300,             # (Optional) the maximum number of epoches (the default is 300).                                
                )

# Get latent representations of X after pre-training
z = model.get_latent_z()
g = get_igraph(z, random_state=seed)
labels = leidenalg_igraph(g, 0.65, random_state=seed) 

NUM_CLUSTER = len(np.unique(labels))
model.init_latent_space(
    NUM_CLUSTER,                     # numebr of clusters
    cluster_labels=labels,           # (optional) clustering labels or their names for plotting
    )

model.train(test_size = 0.1,             # (Optional) the proportion or size of the test set.
            random_state = seed,         # (Optional) the random state of data splitting.            
            batch_size=256,              # (Optional) the batch size for pre-training (the default is 32). 
            alpha=0.10,                  # (Optional) the value of alpha in [0,1] to encourage covariate adjustment. Not used if there is no covariates.
            beta=2,                      # (Optional) the value of beta in beta-VAE.
            num_epoch = 300,             # (Optional) the maximum number of epoches (the default is 300).                            
            early_stopping_warmup=10,    # (Optional) the number of warmup epoches (the default is 0).            
            #**kwargs                    # (Optional ) extra key-value arguments for calling the dimension reduction algorithms.   
            )

model.init_inference(batch_size=128, 
                     L=150,            # L is the number of MC samples
                     dimred='umap',    # dimension reduction methods
                     #**kwargs         # extra key-value arguments for dimension reduction algorithms.    
                     random_state=seed
                    ) 


# save model weight with parameters in the latent space and inference results (embedding, posterior estimations)
# model.save_model(path_to_file='../weight/tutorial_mouse_brain_merged/mouse_brain_inference.checkpoint')

# load model weight
# model.load_model(path_to_file='../weight/tutorial_mouse_brain_merged/mouse_brain_inference.checkpoint', load_labels=True)

import networkx as nx
G = model.comp_inference_score(method='modified_map',  # 'mean', 'modified_mean', 'map', and 'modified_map'                               
                               no_loop=True            # if no_loop=True, then find the maximum spanning tree
                               )           
days = np.array([i[1:3] for i in data['cell_ids']], dtype=np.float32)
begin_node_pred = model.select_root(days, 'sum')
modified_G, modified_w, pseudotime = model.infer_trajectory(
    init_node=begin_node_pred,  # initial node for computing pseudotime.
    cutoff=0.09                 # (Optional) cutoff score for edges (the default is 0.01).
    )



id_branches = [((modified_w[:,3] > 0.0)&(modified_w[:,0] > 0.0)) | \
               ((modified_w[:,0] > 0.0)&(modified_w[:,11] > 0.0)) | \
    (modified_w[:,3] > 0.99) | \
    (modified_w[:,0] > 0.99) | \
    (modified_w[:,11] > 0.99),
((modified_w[:,0] > 0.0)&(modified_w[:,5] > 0.0)) | \
    ((modified_w[:,5] > 0.0)&(modified_w[:,4] > 0.0)) | \
    ((modified_w[:,4] > 0.0)&(modified_w[:,1] > 0.0)) | \
    (modified_w[:,0] > 0.99) | \
    (modified_w[:,5] > 0.99) | \
    (modified_w[:,4] > 0.99) | \
    (modified_w[:,1] > 0.99)] 
branch_names = ['branch 3-0-11', 'branch 0-5-4-1']
for i in range(2):
    id_branch = id_branches[i]
    branch_name = branch_names[i]
    model.set_cell_subset(
        model.cell_names[id_branch]
        )    
    os.makedirs('result/%s'%branch_name, exist_ok=True)
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