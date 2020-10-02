# -*- coding: utf-8 -*-

class Early_Stopping():
    def __init__(self, warmup=0, patience=10, tolerance=1e-3, is_minimize=True):
        self.warmup = warmup
        self.patience = patience
        self.tolerance = tolerance
        self.is_minimize = is_minimize

        self.step = 0
        self.best_step = 0
        self.best_metric = 0

        if not self.is_minimize:
            self.factor = -1.0
        else:
            self.factor = 1.0

    def __call__(self, metric):
        if self.step == 0:
            self.best_step = 1
            self.best_metric = metric
            self.step += 1
            return False

        self.step += 1

        if self.factor*metric<self.factor*self.best_metric-self.tolerance:
            self.best_metric = metric
            self.best_step = self.step
        elif self.step - self.best_step>self.patience:
            if self.step < self.warmup:
                return False
            else:
                print('Best Epoch: %d. Best Metric: %f.'%(self.best_step, self.best_metric))
                return True


from umap.umap_ import nearest_neighbors
from sklearn.utils import check_random_state
from umap.umap_ import fuzzy_simplicial_set
from scipy.sparse import coo_matrix
import igraph as ig
import louvain
import numpy as np
import matplotlib.pyplot as plt


def get_igraph(z):
    # Find knn
    n_neighbors = 15
    random_state = check_random_state(None)
    knn_indices, knn_dists, forest = nearest_neighbors(
        z, n_neighbors, random_state=random_state,
        metric='euclidean', metric_kwds={},
        angular=False, verbose=False,
    )

    # Build graph
    n_obs = z.shape[0]
    X = coo_matrix(([], ([], [])), shape=(n_obs, 1))
    connectivities = fuzzy_simplicial_set(
        X,
        n_neighbors,
        None,
        None,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
    )[0].tocsr()

    # Get igraph graph from adjacency matrix
    sources, targets = connectivities.nonzero()
    weights = connectivities[sources, targets].A1
    g = ig.Graph(directed=None)
    g.add_vertices(connectivities.shape[0])
    g.add_edges(list(zip(sources, targets)))
    g.es['weight'] = weights
    return g


def louvain_igraph(g, res):
    '''
    Params:
        g      - igraph object
        res    - resolution parameter
    Returns:
        labels - clustered labels
    '''
    # Louvain
    partition_kwargs = {}
    partition_type = louvain.RBConfigurationVertexPartition
    partition_kwargs["resolution_parameter"] = res
    partition_kwargs["seed"] = 0
    part = louvain.find_partition(
                    g, partition_type,
                    **partition_kwargs,
                )
    labels = np.array(part.membership)
    return labels
    

def plot_clusters(embed_z, labels, path=None):
    n_labels = len(np.unique(labels))
    colors = [plt.cm.jet(float(i)/n_labels) for i in range(n_labels)]
    
    fig, ax = plt.subplots(1,1, figsize=(10, 5))
    for i,l in enumerate(np.unique(labels)):
        ax.scatter(*embed_z[labels==l].T,
                    c=[colors[i]], label=str(l),
                    s=1, alpha=0.6)
    plt.setp(ax, xticks=[], yticks=[])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                        box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
        fancybox=True, shadow=True, markerscale=5, ncol=5)
    ax.set_title('Clustering')
    if path is not None:
        plt.savefig(path, dpi=300)
    plt.plot()

import os
import numpy as np
import pandas as pd
import h5py
import pickle as pk

type_dict = {
    # dyno
    'dentate':'UMI',
    'immune':'UMI',
    'neonatal':'UMI',
    'mouse_brain':'UMI',
    'planaria_full':'UMI',
    'planaria_muscle':'UMI',
    'aging':'non-UMI',
    'cell_cycle':'non-UMI',
    'fibroblast':'non-UMI',
    'germline':'non-UMI',
    'human':'non-UMI',
    'mesoderm':'non-UMI',
    
    # dyngen
    'bifurcating_1000_2000_2':'non-UMI',
    
    # our model
    'linear':'UMI',
    'bifurcation':'UMI',
    'multifurcating':'UMI',
    'tree':'UMI',
}

def get_data(path, file_name):
    data = {}
    
    # file_name = 'bifurcating_1000_2000_2'
    with h5py.File(os.path.join(path,file_name+'.h5'), 'r') as f:
        data['count'] = np.array(f['count'], dtype=np.float32)
        data['grouping'] = np.array(f['grouping']).astype('U')
        if 'gene_names' in f:
            data['gene_names'] = np.array(f['gene_names']) .astype('U')
        else:
            data['gene_names'] = None
        if 'cell_ids' in f:
            data['cell_ids'] = np.array(f['cell_ids']) .astype('U')
        else:
            data['cell_ids'] = None
        if 'milestone_network' in f:
            if file_name in ['linear','bifurcation','multifurcating','tree']:
                data['milestone_network'] = pd.DataFrame(
                    np.array(np.array(list(f['milestone_network'])).tolist(), dtype='U'),
                    columns=['from','to','w']
                ).astype({'w':np.float32})
            else:
                data['milestone_network'] = pd.DataFrame(
                    np.array(np.array(list(f['milestone_network'])).tolist(), dtype='U'),
                    columns=['from','to']
                )
            data['root_milestone_id'] = np.array(f['root_milestone_id']).astype('U')[0]
        else:
            data['milestone_net'] = None
            data['root_milestone_id'] = None
    
    data['type'] = type_dict[file_name]
    if data['type']=='non-UMI':
        scale_factor = np.sum(data['x'],axis=1, keepdims=True)/1e6
        data['x'] = data['x']/scale_factor
    
    return data['x']    
