import os
import sys
sys.path.append("../")

import tensorflow as tf
import numpy as np
import pandas as pd
import umap

import matplotlib.pyplot as plt
import matplotlib

import pickle as pk
import h5py
import random

from sklearn.decomposition import PCA
import networkx as nx
import scanpy as sc

import VITAE
from VITAE.utils import load_data, plot_clusters

type_dict = {
    # dyno
    'dentate': 'UMI',
    'immune': 'UMI',
    'planaria_muscle': 'UMI',
    'planaria_full': 'UMI',
    'aging': 'non-UMI',
    'cell_cycle': 'non-UMI',
    'fibroblast': 'non-UMI',
    'germline': 'non-UMI',
    'human_embryos': 'non-UMI',
    'mesoderm': 'non-UMI',

    # dyngen
    "cycle_1": 'non-UMI',
    "cycle_2": 'non-UMI',
    "cycle_3": 'non-UMI',
    "linear_1": 'non-UMI',
    "linear_2": 'non-UMI',
    "linear_3": 'non-UMI',
    "trifurcating_1": 'non-UMI',
    "trifurcating_2": 'non-UMI',
    "bifurcating_1": 'non-UMI',
    'bifurcating_2': 'non-UMI',
    "bifurcating_3": 'non-UMI',
    "converging_1": 'non-UMI',

    # our model
    'linear': 'UMI',
    'bifurcation': 'UMI',
    'multifurcating': 'UMI',
    'tree': 'UMI',
}


no_loop = False if 'cycle' in file_name else True
is_init = True
data,dd = load_data(path='../data/',
                 file_name=file_name,return_dict = True)

all_hvg_data = ["linear","linear_1","linear_2","linear_3",
                "bifurcation","bifurcating_1","bifurcating_2",
                "converging_1","cycle_1","cycle_2","multifurcating","tree"]

sc.pp.normalize_total(dd)
sc.pp.log1p(dd)
sc.pp.highly_variable_genes(dd)
sc.pp.scale(dd)

if file_name in all_hvg_data:
    dd.var["highly_variable"] = True

model = VITAE.VITAE(adata=dd,
                    hidden_layers=[32, 16],
                    latent_space_dim=16,
                    model_type="Gaussian",npc = 64)

NUM_CLUSTER = len(np.unique(data['grouping']))

df = pd.DataFrame()
n = int(sys.argv[1])
num_inference = 5
id_simulation = int(sys.argv[1])
PATH = 'result_Gaussian/%s/%s/'%(file_name,'weight')

random.seed(n)
np.random.seed(n)
tf.random.set_seed(n)

path = os.path.join(PATH, '%d' % n)
os.makedirs(path, exist_ok=True)

tf.keras.backend.clear_session()
tf.keras.backend.set_floatx('float32')
if os.path.exists(os.path.join(path, 'train.checkpoint.index')):
    model.load_model(os.path.join(path, 'train.checkpoint'), load_labels=True)
else:
    model.pre_train(learning_rate=1e-3, early_stopping_tolerance=0.01,
                    early_stopping_relative=True, L=1, verbose=True, num_epoch=400,
                    path_to_weights=os.path.join(path, 'pre_train.checkpoint'))
    model.init_latent_space(cluster_label='grouping', ratio_prune=0.5,dist_thres=0)
    z = model.get_latent_z()

    embed_z = umap.UMAP().fit_transform(z)
    plot_clusters(embed_z, model.labels, path=os.path.join(path, 'cluster_umap.png'))

    embed_z = PCA(n_components=2).fit_transform(z)
    plot_clusters(embed_z, model.labels, path=os.path.join(path, 'cluster_pca.png'))

    model.train(beta=1, learning_rate=1e-3, early_stopping_tolerance=0.01,
                early_stopping_relative=True, verbose=False, num_epoch=400,
                path_to_weights=os.path.join(path, 'train.checkpoint'))

L = 300
for i in range(num_inference):
    if data['count'].shape[0] > 15000:
        batch_size = 32
    else:
        batch_size = 128
    model.posterior_estimation(batch_size=batch_size, L=L)
    model.save_model(path_to_file=os.path.join(path, 'inference.checkpoint'))

    for method in ['mean', 'modified_mean', 'map', 'modified_map', "w_base","raw_map"]:
        if i == 0:
            ## simulation dataset default cutoff = 0.01
            bb_path = os.path.join(path, f'backbone_{method}.png')
            model.infer_backbone(cutoff=0.0001, no_loop=True, visualize=False, method=method, path_to_fig=bb_path)
            G = model.backbone

            traj_path = os.path.join(path, f'trajectory_{method}.png')
            model.infer_trajectory(data["root_milestone_id"], visualize=True, path_to_fig=traj_path)
            print(nx.to_numpy_array(G))

        _df = pd.DataFrame()

        # if cutoff = None then default to 0.01
        res = model.evaluate(data['milestone_network'].copy(),
                             data["root_milestone_id"],
                             grouping=data['grouping'].copy(),
                             method=method,
                             no_loop=no_loop,
                             cutoff=0.0001,
                             )
        _df = _df.append(pd.DataFrame(res, index=[0]), ignore_index=True)
        _df['method'] = method
        df = df.append(_df, ignore_index=True)
    df.to_csv(os.path.join(PATH, 'result_%s.csv' % (file_name)),mode='a', header=False, index=None)
plt.close('all')