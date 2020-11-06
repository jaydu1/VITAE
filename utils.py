# -*- coding: utf-8 -*-
from umap.umap_ import nearest_neighbors
from sklearn.utils import check_random_state
from umap.umap_ import fuzzy_simplicial_set
from scipy.sparse import coo_matrix
import igraph as ig
import louvain
import matplotlib.pyplot as plt
import matplotlib
import os     
import numpy as np
from numba import jit, float32, int32
import pandas as pd
import h5py


#------------------------------------------------------------------------------
# Early stopping
#------------------------------------------------------------------------------

class Early_Stopping():
    def __init__(self, warmup=0, patience=10, tolerance=1e-3, is_minimize=True):
        self.warmup = warmup
        self.patience = patience
        self.tolerance = tolerance
        self.is_minimize = is_minimize

        self.step = -1
        self.best_step = -1
        self.best_metric = np.inf

        if not self.is_minimize:
            self.factor = -1.0
        else:
            self.factor = 1.0

    def __call__(self, metric):
        self.step += 1
        
        if self.step < self.warmup:
            return False
        elif self.factor*metric<self.factor*self.best_metric-self.tolerance:
            self.best_metric = metric
            self.best_step = self.step
            return False
        elif self.step - self.best_step>self.patience:
            print('Best Epoch: %d. Best Metric: %f.'%(self.best_step, self.best_metric))
            return True
        else:
            return False
            
            
#------------------------------------------------------------------------------
# Utils functions
#------------------------------------------------------------------------------

@jit((float32[:,:],), nopython=True, nogil=True)
def _check_expression(A):
    n_rows = A.shape[0]
    out = np.ones(n_rows, dtype=int32)
    for i in range(n_rows):        
        for j in A[i,:]:
            if j>0.0:
                break
        else:
            out[i] = 0
    return out

@jit((float32[:,:],), nopython=True, nogil=True)
def _check_variability(A):
    n_cols = A.shape[1]
    out = np.ones(n_cols, dtype=int32)
    for i in range(n_cols):
        init = A[0, i]
        for j in A[1:, i]:
            if j != init:
                break
        else:
            out[i] = 0
    return out


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
    

def plot_clusters(embed_z, labels, plot_labels=False, path=None):
    n_labels = len(np.unique(labels))
    colors = [plt.cm.jet(float(i)/n_labels) for i in range(n_labels)]
    
    fig, ax = plt.subplots(1,1, figsize=(20, 10))
    for i,l in enumerate(np.unique(labels)):
        ax.scatter(*embed_z[labels==l].T,
                    c=[colors[i]], label=str(l),
                    s=8, alpha=0.4)
        if plot_labels:
            ax.text(np.mean(embed_z[labels==l,0]), np.mean(embed_z[labels==l,1]), str(l), fontsize=16)
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

    
def plot_marker_gene(expression, gene_name, embed_z, path=None):
    fig, ax = plt.subplots(1,1, figsize=(20, 10))
    cmap = matplotlib.cm.get_cmap('Reds')
    sc = ax.scatter(*embed_z.T, c='yellow', s=15)
    sc = ax.scatter(*embed_z.T, cmap=cmap, c=expression, s=10)
    sc.set_clim(0,1) 
    plt.colorbar(sc, ax=[ax], location='right')
    ax.set_title('Normalized expression of {}'.format(gene_name))
    if path is not None:
        plt.savefig(path, dpi=300)
    plt.show()
    return None
    
    
#------------------------------------------------------------------------------
# Data loader
#------------------------------------------------------------------------------

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
    'linear_1':'non-UMI',
    'bifurcating_1':'non-UMI',
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
    "converging_2":'non-UMI',
    
    # our model
    'linear':'UMI',
    'bifurcation':'UMI',
    'multifurcating':'UMI',
    'tree':'UMI',
}

def load_data(path, file_name):   
    data = {}
    
    with h5py.File(os.path.join(path, file_name+'.h5'), 'r') as f:
        data['count'] = np.array(f['count'], dtype=np.float32)
        data['grouping'] = np.array(f['grouping']).astype(str)
        if 'gene_names' in f:
            data['gene_names'] = np.array(f['gene_names']).astype(str)
        else:
            data['gene_names'] = None
        if 'cell_ids' in f:
            data['cell_ids'] = np.array(f['cell_ids']).astype(str)
        else:
            data['cell_ids'] = None
            
        if 'milestone_network' in f:
            if file_name in ['linear','bifurcation','multifurcating','tree',
                             'bifurcating_1000_2000_2', 'linear_1', 'bifurcating_1',
                            "cycle_1", "cycle_2", "cycle_3",
                            "linear_1", "linear_2", "linear_3", 
                            "trifurcating_1", "trifurcating_2", 
                            "bifurcating_1", "bifurcating_3", 
                            "converging_2"]:
                data['milestone_network'] = pd.DataFrame(
                    np.array(np.array(list(f['milestone_network'])).tolist(), dtype=str), 
                    columns=['from','to','w']
                ).astype({'w':np.float32})
            else:
                data['milestone_network'] = pd.DataFrame(
                    np.array(np.array(list(f['milestone_network'])).tolist(), dtype=str), 
                    columns=['from','to']
                )
            data['root_milestone_id'] = np.array(f['root_milestone_id']).astype(str)[0]            
        else:
            data['milestone_net'] = None
            data['root_milestone_id'] = None
            
        if file_name in ['mouse_brain']:
            data['grouping'] = np.array(['%02d'%int(i) for i in data['grouping']], dtype=str)
            data['root_milestone_id'] = '06'
            data['covariates'] = np.array(np.array(list(f['covariates'])).tolist(), dtype=np.float32)

    data['type'] = type_dict[file_name]
    if data['type']=='non-UMI':
        scale_factor = np.sum(data['count'],axis=1, keepdims=True)/1e6
        data['count'] = data['count']/scale_factor
    
    return data  