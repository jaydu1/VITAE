import tensorflow as tf
import numpy as np
import pandas as pd
import math
import umap
import os 
import sys
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx

from VITAE import VITAE, get_igraph, louvain_igraph, plot_clusters, load_data, get_embedding
type_dict = {
    # dyno
    'dentate':'UMI', 
    'immune':'UMI', 
    'neonatal':'UMI', 
    'planaria_muscle':'UMI',
    'planaria_full':'UMI',
    'aging':'non-UMI', 
    'cell_cycle':'non-UMI',
    'fibroblast':'non-UMI', 
    'germline':'non-UMI',    
    'human':'non-UMI', 
    'mesoderm':'non-UMI',
    
    # dyngen
    'bifurcating_2':'non-UMI',
    "cycle_1":'non-UMI', 
    "cycle_2":'non-UMI', 
    "cycle_3":'non-UMI',
    "linear_1":'non-UMI', 
    "linear_2":'non-UMI', 
    "linear_3":'non-UMI', 
    "trifurcating_1":'non-UMI', 
    "trifurcating_2":'non-UMI', 
    "bifurcating_1":'non-UMI', 
    "bifurcating_3":'non-UMI', 
    "converging_1":'non-UMI',
    
    # our model
    'linear':'UMI',
    'bifurcation':'UMI',
    'multifurcating':'UMI',
    'tree':'UMI',
}
source_dict = {
    'dentate':'dyno', 
    'immune':'dyno', 
    'neonatal':'dyno', 
    'planaria_muscle':'dyno',
    'planaria_full':'dyno',
    'aging':'dyno', 
    'cell_cycle':'dyno',
    'fibroblast':'dyno', 
    'germline':'dyno',    
    'human':'dyno', 
    'mesoderm':'dyno',
    
    'bifurcating_2':'dyngen',
    "cycle_1":'dyngen', 
    "cycle_2":'dyngen', 
    "cycle_3":'dyngen',
    "linear_1":'dyngen', 
    "linear_2":'dyngen', 
    "linear_3":'dyngen', 
    "trifurcating_1":'dyngen', 
    "trifurcating_2":'dyngen', 
    "bifurcating_1":'dyngen', 
    "bifurcating_3":'dyngen', 
    "converging_1":'dyngen',
    
    'linear':'our model',
    'bifurcation':'our model',
    'multifurcating':'our model',
    'tree':'our model',
}

df = pd.DataFrame()

for datatype in ['NB','Gaussian']:
    if datatype=='NB':
        data_type = data['type']
    else:
        data_type = datatype

    for file_name in type_dict.keys():
        no_loop = False if 'cycle' in file_name else True
        is_init = True
        data = load_data('../../data/',file_name)
        model = VITAE()
        model.get_data(
            data['count'].copy(), 
            labels = data['grouping'].copy(), 
            gene_names=data['gene_names'])
        model.preprocess_data(                      
            gene_num = 2000,            # (optional) maximum number of influential genes to keep (the default is 2000)
            data_type = data_type,   # (optional) data_type can be 'UMI', 'non-UMI' or 'Gaussian' (the default is 'UMI')
            npc = 64                    # (optional) number of PCs to keep if data_type='Gaussian' (the default is 64)
        )
            
        num_simulation = 20
        dim_latent = 8
        NUM_CLUSTER = len(np.unique(data['grouping']))
        for n in range(num_simulation):
            tf.keras.backend.clear_session()
            model.build_model(dim_latent = dim_latent, 
                            dimensions=[16])
            model.pre_train(learning_rate = 1e-3,    # (Optional) the initial learning rate for the Adam optimizer (the default is 1e-3).
                        batch_size=256,              # (Optional) the batch size for pre-training (the default is 32). 
                        L=1,                         # (Optional) the number of MC samples (the default is 1).
                        num_epoch = 300,             # (Optional) the maximum number of epoches (the default is 300).                
                        num_step_per_epoch = None,   # (Optional) the number of step per epoch, it will be inferred from number of cells and batch size if it is None (the default is None).
                        early_stopping_tolerance=1,  # (Optional) the minimum change of loss to be considered as an improvement (the default is 1e-3).
                        early_stopping_patience=5,   # (Optional) the maximum number of epoches if there is no improvement (the default is 10).
                        early_stopping_warmup=0,     # (Optional) the number of warmup epoches (the default is 0).
                        path_to_weights=None,        # (Optional) the path of weight file to be saved; not saving weight if None (the default is None).
                        alpha=0.1
                        ) 

            z = model.get_latent_z()
            labels = model.labels    
            mu = np.zeros((z.shape[1],NUM_CLUSTER))
            for i in np.arange(NUM_CLUSTER):
                mu[:,i] = np.mean(z[labels==i], axis=0)

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
                    path_to_weights=None,          # (Optional) the path of weight file to be saved; not saving weight if None (the default is None). 
                    )

            begin_node_true = model.le.transform([data['root_milestone_id']])[0]
            num_inference = 5
            L = 300
            for i in range(num_inference):
                if data['count'].shape[0]>15000:
                    batch_size = 16
                else:
                    batch_size = 64
                model.init_inference(batch_size=batch_size, L=L, refit_dimred=False)
                
                for method in ['mean','modified_mean','map','modified_map']:
                    _df = pd.DataFrame()            
                    res = model.evaluate(data['milestone_network'].copy(),
                                        begin_node_true, 
                                        grouping=data['grouping'].copy(), 
                                        method=method,
                                        no_loop=no_loop,
                                        cutoff=None,

                                        )
                    _df = _df.append(pd.DataFrame(res, index=[0]),ignore_index=True)
                    _df['method'] = method
                    _df['type'] = data['type']
                    _df['data'] = file_name
                    _df['source'] = source_dict[file_name]
                    df = df.append(_df,ignore_index=True)
    df = df.groupby('method').mean().sort_values(['data','method']).reset_index(drop=True)
    df.to_csv('result/result_VITAE_%s.csv'%(file_name,datatype))