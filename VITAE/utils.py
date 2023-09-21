# -*- coding: utf-8 -*-
import sys
import os
import random
import numpy as np
import pandas as pd
from numba import jit, float32, int32
import scipy
from scipy import stats

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K

import h5py
import scanpy as sc
import anndata

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from umap.umap_ import nearest_neighbors, smooth_knn_dist
import umap
from sklearn.utils import check_random_state
from scipy.sparse import coo_matrix
import igraph as ig
import leidenalg

import matplotlib.pyplot as plt
import matplotlib


#------------------------------------------------------------------------------
# Early stopping
#------------------------------------------------------------------------------

class Early_Stopping():
    '''
    The early-stopping monitor.
    '''
    def __init__(self, warmup=0, patience=10, tolerance=1e-3, 
            relative=False, is_minimize=True):
        self.warmup = warmup
        self.patience = patience
        self.tolerance = tolerance
        self.is_minimize = is_minimize
        self.relative = relative

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
        elif (self.best_metric==np.inf) or \
                (self.relative and (self.best_metric-metric)/self.best_metric > self.tolerance) or \
                ((not self.relative) and self.factor*metric<self.factor*self.best_metric-self.tolerance):
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

def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _comp_dist(x, y, mu=None, S=None):
    uni_y = np.unique(y)
    n_uni_y = len(uni_y)
    d = x.shape[1]
    if mu is None:
        mu = np.zeros((n_uni_y, d))
        for i,l in enumerate(uni_y):
            mu[i, :] = np.mean(x[y==l], axis=0)
    if S is None:
        S = np.zeros((n_uni_y, d, d))
        for i,l in enumerate(uni_y):
            S[i, :, :] = np.cov(x[y==l], rowvar=False)
    dist = np.zeros((n_uni_y, n_uni_y))
    for i,li in enumerate(uni_y):
        for j,lj in enumerate(uni_y):            
            if i<j:
                dist[i,j] = (mu[i:i+1,:]-mu[j:j+1,:]) @ np.linalg.inv(S[i, :, :] + S[j, :, :]) @ (mu[i:i+1,:]-mu[j:j+1,:]).T
    dist = dist + dist.T
    return dist


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


def get_embedding(z, dimred='umap', **kwargs):
    '''Get low-dimensional embeddings for visualizations.

    Parameters
    ----------
    z : np.array
        \([N, d]\) The latent variables.
    dimred : str, optional
        'pca', 'tsne', or umap'.   
    **kwargs :  
        Extra key-value arguments for dimension reduction algorithms.  

    Returns:
    ----------
    embed : np.array
        \([N, 2]\) The latent variables after dimension reduction.
    '''
    if dimred=='umap':
        if 'random_state' in kwargs:
            kwargs['random_state'] = np.random.RandomState(kwargs['random_state'])
        # umap has some bugs that it may change the original matrix when doing transform 
        mapper = umap.UMAP(**kwargs).fit(z.copy())
        embed = mapper.embedding_
    elif dimred=='pca':
        kwargs['n_components'] = 2            
        embed = PCA(**kwargs).fit_transform(z)
    elif dimred=='tsne':
        embed = TSNE(**kwargs).fit_transform(z)
    else:
        raise ValueError("Dimension reduction method can only be 'umap', 'pca' or 'tsne'!")
    return embed


def _compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos, return_dists=False, bipartite=False):
    '''
    Overwrite the UMAP `compute_membership_strengths` function to allow computation with float64.
    '''
    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]

    rows = np.zeros(knn_indices.size, dtype=np.int32)
    cols = np.zeros(knn_indices.size, dtype=np.int32)
    vals = np.zeros(knn_indices.size, dtype=np.float64)
    if return_dists:
        dists = np.zeros(knn_indices.size, dtype=np.float64)
    else:
        dists = None

    for i in range(n_samples):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            # If applied to an adjacency matrix points shouldn't be similar to themselves.
            # If applied to an incidence matrix (or bipartite) then the row and column indices are different.
            if (bipartite == False) & (knn_indices[i, j] == i):
                val = 0.0
            elif knn_dists[i, j] - rhos[i] <= 0.0 or sigmas[i] == 0.0:
                val = 1.0
            else:
                val = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i])))

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val
            if return_dists:
                dists[i * n_neighbors + j] = knn_dists[i, j]

    return rows, cols, vals, dists


def _fuzzy_simplicial_set(X, n_neighbors, random_state,
    metric, metric_kwds={}, knn_indices=None, knn_dists=None, angular=False,
    set_op_mix_ratio=1.0, local_connectivity=1.0, apply_set_operations=True,
    verbose=False, return_dists=None):
    '''
    Overwrite the UMAP `fuzzy_simplicial_set` function to allow computation with float64.
    '''

    if knn_indices is None or knn_dists is None:
        knn_indices, knn_dists, _ = nearest_neighbors(
            X, n_neighbors, metric, metric_kwds, angular, random_state, verbose=verbose,
        )

    sigmas, rhos = smooth_knn_dist(
        knn_dists, float(n_neighbors), local_connectivity=float(local_connectivity),
    )

    rows, cols, vals, dists = _compute_membership_strengths(
        knn_indices, knn_dists, sigmas, rhos, return_dists
    )

    result = scipy.sparse.coo_matrix(
        (vals, (rows, cols)), shape=(X.shape[0], X.shape[0])
    )
    result.eliminate_zeros()

    if apply_set_operations:
        transpose = result.transpose()

        prod_matrix = result.multiply(transpose)

        result = (
            set_op_mix_ratio * (result + transpose - prod_matrix)
            + (1.0 - set_op_mix_ratio) * prod_matrix
        )

    result.eliminate_zeros()

    if return_dists is None:
        return result, sigmas, rhos
    else:
        if return_dists:
            dmat = scipy.sparse.coo_matrix(
                (dists, (rows, cols)), shape=(X.shape[0], X.shape[0])
            )

            dists = dmat.maximum(dmat.transpose()).todok()
        else:
            dists = None

        return result, sigmas, rhos, dists


def get_igraph(z, random_state=0):
    '''Get igraph for running Leidenalg clustering.

    Parameters
    ----------
    z : np.array
        \([N, d]\) The latent variables.
    random_state : int, optional
        The random state.
    Returns:
    ----------
    g : igraph
        The igraph object of connectivities.      
    '''    
    # Find knn
    n_neighbors = 15
    knn_indices, knn_dists, forest = nearest_neighbors(
        z, n_neighbors, 
        random_state=np.random.RandomState(random_state),
        metric='euclidean', metric_kwds={},
        angular=False, verbose=False,
    )

    # Build graph
    n_obs = z.shape[0]
    X = coo_matrix(([], ([], [])), shape=(n_obs, 1))
    connectivities = _fuzzy_simplicial_set(
        X,
        n_neighbors,
        random_state=np.random.RandomState(random_state),
        metric=None,
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


def leidenalg_igraph(g, res, random_state=0):
    '''Leidenalg clustering on an igraph object.

    Parameters
    ----------
    g : igraph
        The igraph object of connectivities.
    res : float
        The resolution parameter for Leidenalg clustering.
    random_state : int, optional
        The random state.      

    Returns
    ----------
    labels : np.array     
        \([N, ]\) The clustered labels.
    '''
    partition_kwargs = {}
    partition_type = leidenalg.RBConfigurationVertexPartition
    partition_kwargs["resolution_parameter"] = res
    partition_kwargs["seed"] = random_state
    part = leidenalg.find_partition(
                    g, partition_type,
                    **partition_kwargs,
                )
    labels = np.array(part.membership)
    return labels
    

def plot_clusters(embed_z, labels, plot_labels=False, path=None):
    '''Plot the clustering results.

    Parameters
    ----------
    embed_z : np.array
        \([N, 2]\) The latent variables after dimension reduction.
    labels : np.array     
        \([N, ]\) The clustered labels.
    plot_labels : boolean, optional
        Whether to plot text of labels or not.
    path : str, optional
        The path to save the figure.
    '''    
    n_labels = len(np.unique(labels))
    colors = [plt.cm.jet(float(i)/n_labels) for i in range(n_labels)]
    
    fig, ax = plt.subplots(1,1, figsize=(20, 10))
    for i,l in enumerate(np.unique(labels)):
        ax.scatter(*embed_z[labels==l].T,
                    c=[colors[i]], label=str(l),
                    s=16, alpha=0.4)
        if plot_labels:
            ax.text(np.mean(embed_z[labels==l,0]), np.mean(embed_z[labels==l,1]), str(l), fontsize=16)
    plt.setp(ax, xticks=[], yticks=[])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                        box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
        fancybox=True, shadow=True, markerscale=3, ncol=5)
    ax.set_title('Clustering')
    if path is not None:
        plt.savefig(path, dpi=300)
    plt.plot()

    
def plot_marker_gene(expression, gene_name: str, embed_z, path=None):
    '''Plot the marker gene.

    Parameters
    ----------
    expression : np.array
        \([N, ]\) The expression of the marker gene.
    gene_name : str
        The name of the marker gene.
    embed_z : np.array
        \([N, 2]\) The latent variables after dimension reduction.
    path : str, optional
        The path to save the figure.
    '''      
    fig, ax = plt.subplots(1,1, figsize=(20, 10))
    cmap = matplotlib.cm.get_cmap('Reds')
    sc = ax.scatter(*embed_z.T, c='yellow', s=15, alpha=0.1)
    sc = ax.scatter(*embed_z.T, cmap=cmap, c=expression, s=10, alpha=0.5)
    plt.colorbar(sc, ax=[ax], location='right')
    plt.setp(ax, xticks=[], yticks=[])
    ax.set_title('Normalized expression of {}'.format(gene_name))
    if path is not None:
        plt.savefig(path, dpi=300)
    plt.show()
    return None


def plot_uncertainty(uncertainty, embed_z, path=None):
    '''Plot the uncertainty for all selected cells.

    Parameters
    ----------
    uncertainty : np.array
        \([N, ]\) The uncertainty of the all cells.    
    embed_z : np.array
        \([N, 2]\) The latent variables after dimension reduction.
    path : str, optional
        The path to save the figure.
    '''          
    fig, ax = plt.subplots(1,1, figsize=(20, 10))
    cmap = matplotlib.cm.get_cmap('RdBu_r')
    sc = ax.scatter(*embed_z.T, cmap=cmap, c=uncertainty, s=10, alpha=1.0)
    plt.colorbar(sc, ax=[ax], location='right')
    plt.setp(ax, xticks=[], yticks=[])
    ax.set_title("Cells' Uncertainty")
    if path is not None:
        plt.savefig(path, dpi=300)
    plt.show()
    return None


def _polyfit_with_fixed_points(n, x, y, xf, yf):
    '''
    Fix a polynomial with degree n that goes through 
    fixed points (xf_j, yf_j).
    '''
    mat = np.empty((n + 1 + len(xf),) * 2)
    vec = np.empty((n + 1 + len(xf),))
    x_n = x**np.arange(2 * n + 1)[:, None]
    yx_n = np.sum(x_n[:n + 1] * y, axis=1)
    x_n = np.sum(x_n, axis=1)
    idx = np.arange(n + 1) + np.arange(n + 1)[:, None]
    mat[:n + 1, :n + 1] = np.take(x_n, idx)
    xf_n = xf**np.arange(n + 1)[:, None]
    mat[:n + 1, n + 1:] = xf_n / 2
    mat[n + 1:, :n + 1] = xf_n.T
    mat[n + 1:, n + 1:] = 0
    vec[:n + 1] = yx_n
    vec[n + 1:] = yf
    params = np.linalg.solve(mat, vec)

    return params[:n + 1]


def _get_smooth_curve(xy, xy_fixed, y_range):
    xy = np.r_[xy, xy_fixed]
    _, idx = np.unique(xy[:,0], return_index=True)
    xy = xy[idx,:]
    order = 3
    while order>0:
        params = _polyfit_with_fixed_points(
            order, 
            xy[:,0], xy[:,1], 
            xy_fixed[:,0], xy_fixed[:,1]
            )
        poly = np.polynomial.Polynomial(params)
        xx = np.linspace(xy_fixed[0,0], xy_fixed[-1,0], 100)
        yy = poly(xx)
        if np.max(yy)>y_range[1] or np.min(yy)<y_range[0] :
            order -= 1
        else:
            break
    return xx, yy


def _pinv_extended(x, rcond=1e-15):
    """
    Return the pinv of an array X as well as the singular values
    used in computation.
    Code adapted from numpy.
    """
    x = np.asarray(x)
    x = x.conjugate()
    u, s, vt = np.linalg.svd(x, False)
    s_orig = np.copy(s)
    m = u.shape[0]
    n = vt.shape[1]
    cutoff = rcond * np.maximum.reduce(s)
    for i in range(min(n, m)):
        if s[i] > cutoff:
            s[i] = 1./s[i]
        else:
            s[i] = 0.
    res = np.dot(np.transpose(vt), np.multiply(s[:, np.core.newaxis],
                                               np.transpose(u)))
    return res, s_orig


def _cov_hc3(h, pinv_wexog, resid):
    het_scale = (resid/(1-h))**2

    # sandwich with pinv(x) * diag(scale) * pinv(x).T
    # where pinv(x) = (X'X)^(-1) X and scale is (nobs,)
    cov_hc3_ = np.dot(pinv_wexog, het_scale[:,None]*pinv_wexog.T)
    return cov_hc3_


def _p_adjust_bh(p):
    """Benjamini-Hochberg p-value correction for multiple hypothesis testing."""
    n = len(p)
    nna = ~np.isnan(p)
    lp = np.sum(nna)

    p0 = np.empty_like(p)
    p0[~nna] = np.nan
    p = p[nna]
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(lp) / np.arange(lp, 0, -1)
    p0[nna] = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))[by_orig]
    return p0

def DE_test(Y, X, gene_names, i_test, alpha: float = 0.05):
    '''Differential gene expression test.

    Parameters
    ----------
    Y : numpy.array
        \(n,\) the expression matrix.
    X : numpy.array
        \(n,1+1+s\) the constant term, the pseudotime and the covariates.
    gene_names : numpy.array
        \(n,\) the names of all genes.
    i_test : numpy.array
        The indices of covariates to be tested.
    alpha : float, optional
        The cutoff of p-values.

    Returns
    ----------
    res_df : pandas.DataFrame
        The test results of expressed genes with two columns,
        the estimated coefficients and the adjusted p-values.
    '''
    pinv_wexog, singular_values = _pinv_extended(X)
    normalized_cov = np.dot(
            pinv_wexog, np.transpose(pinv_wexog))
    h = np.diag(np.dot(X,
                    np.dot(normalized_cov,X.T)))

    def _DE_test(wendog,pinv_wexog,h):
        if np.any(np.isnan(wendog)):
            return np.empty(2)*np.nan
        else:
            beta = np.dot(pinv_wexog, wendog)
            resid = wendog - X @ beta
            cov = _cov_hc3(h, pinv_wexog, resid)
            t = np.array([])
            for j in i_test:
                if np.diag(cov)[j] == 0:
                    _t = float("nan")
                else:
                    _t = beta[j]/(np.sqrt(np.diag(cov)[j])+1e-6)
                t = np.append(t, _t)
            return np.r_[beta[i_test], t]

    res = np.apply_along_axis(lambda y: _DE_test(wendog=y, pinv_wexog=pinv_wexog, h=h),
                            0,
                            Y).T

    res_df = pd.DataFrame()
    for i,j in enumerate(i_test):
        if 'median_abs_deviation' in dir(stats):
            sigma = stats.median_abs_deviation(res[:,len(i_test)+i], nan_policy='omit')
        else:
            sigma = stats.median_absolute_deviation(res[:,len(i_test)+i], nan_policy='omit')
        pdt_new_pval = np.array([stats.norm.sf(x)*2 for x in np.abs(res[:,len(i_test)+i]/sigma)])
        new_adj_pval = _p_adjust_bh(pdt_new_pval/len(i_test))
        _res_df = pd.DataFrame(np.c_[res[:,i], pdt_new_pval, new_adj_pval],
                        index=gene_names,
                        columns=['beta_{}'.format(j),
                                 'pvalue_{}'.format(j),
                                 'pvalue_adjusted_{}'.format(j)])
        res_df = pd.concat([res_df, _res_df], axis=1)
    res_df = res_df[
        (np.sum(
            res_df[
                res_df.columns[
                    np.char.startswith(
                        np.array(res_df.columns, dtype=str),
                        'pvalue_adjusted')]
            ] < alpha, axis=1
        )>0) & np.any(~np.isnan(Y), axis=0)]
#     res_df = res_df.iloc[np.argsort(res_df.pvalue_adjusted).tolist(), :]
    return res_df

#------------------------------------------------------------------------------
# Data loader
#------------------------------------------------------------------------------

type_dict = {
    # real data / dyno
    'dentate_withdays':'UMI',
    'dentate':'UMI', 
    'immune':'UMI', 
    'neonatal':'UMI', 
    'mouse_brain':'UMI', 
    'mouse_brain_miller':'UMI',
    'mouse_brain_merged':'UMI',
    'planaria_full':'UMI', 
    'planaria_muscle':'UMI',
    'aging':'non-UMI', 
    'cell_cycle':'non-UMI',
    'fibroblast':'non-UMI', 
    'germline':'non-UMI',    
    'human_embryos':'non-UMI', 
    'mesoderm':'non-UMI',
    'human_hematopoiesis_scATAC':'UMI',
    'human_hematopoiesis_scRNA':'UMI',
    
    # dyngen
    "linear_1":'non-UMI', 
    "linear_2":'non-UMI', 
    "linear_3":'non-UMI',
    'bifurcating_1':'non-UMI',
    'bifurcating_2':'non-UMI',
    "bifurcating_3":'non-UMI', 
    "cycle_1":'non-UMI', 
    "cycle_2":'non-UMI', 
    "cycle_3":'non-UMI',
    "trifurcating_1":'non-UMI', 
    "trifurcating_2":'non-UMI',         
    "converging_1":'non-UMI',
    
    # our model
    'linear':'UMI',
    'bifurcation':'UMI',
    'multifurcating':'UMI',
    'tree':'UMI',
}




def load_data(path, file_name,return_dict = False):
    '''Load h5df data.

    Parameters
    ----------
    path : str
        The path of the h5 files.
    file_name : str
        The dataset name.
    
    Returns:
    ----------
    data : dict
        The dict containing count, grouping, etc. of the dataset.
    '''     
    data = {}
    
    with h5py.File(os.path.join(path, file_name+'.h5'), 'r') as f:
        data['count'] = np.array(f['count'], dtype=np.float32)
        dd = anndata.AnnData(X=data["count"])
        dd.layers["count"] = data["count"].copy()

        data['grouping'] = np.array(f['grouping']).astype(str)
        dd.obs["grouping"] = data["grouping"]
        dd.obs["grouping"] = dd.obs["grouping"].astype("category")
        if 'gene_names' in f:
            data['gene_names'] = np.array(f['gene_names']).astype(str)
            dd.var.index = data["gene_names"]
        else:
            data['gene_names'] = None
        if 'cell_ids' in f:
            data['cell_ids'] = np.array(f['cell_ids']).astype(str)
            dd.obs.index = data["cell_ids"]
        else:
            data['cell_ids'] = None
        if 'days' in f:
            data['days'] = np.array(f['days']).astype(str)

        if 'milestone_network' in f:
            if file_name in ['linear','bifurcation','multifurcating','tree',
                               "cycle_1", "cycle_2", "cycle_3",
                            "linear_1", "linear_2", "linear_3", 
                            "trifurcating_1", "trifurcating_2", 
                            "bifurcating_1", 'bifurcating_2', "bifurcating_3", 
                            "converging_1"]:
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
            
        if file_name in ['mouse_brain', 'mouse_brain_miller']:
            data['grouping'] = np.array(['%02d'%int(i) for i in data['grouping']], dtype=object)
            data['root_milestone_id'] = dict(zip(['mouse_brain', 'mouse_brain_miller'], ['06', '05']))[file_name]
            data['covariates'] = np.array(np.array(list(f['covariates'])).tolist(), dtype=np.float32)
        if file_name in ['mouse_brain_merged']:
            data['grouping'] = np.array(data['grouping'], dtype=object)
            data['root_milestone_id'] = np.array(f['root_milestone_id']).astype(str)[0]
            data['covariates'] = np.array(np.array(list(f['covariates'])).tolist(), dtype=np.float32)
        if file_name == 'dentate_withdays':
            data['covariates'] = np.array([item.decode('utf-8').replace('*', '') for item in f['days']], dtype=object)
            data['covariates'] = data['covariates'].astype(float).reshape(-1, 1)
        if file_name.startswith('human_hematopoiesis'):
            data['covariates'] = np.array(np.array(list(f['covariates'])[0], dtype=str).tolist()).reshape((-1,1))
            
    data['type'] = type_dict[file_name]
    if data['type']=='non-UMI':
        scale_factor = np.sum(data['count'],axis=1, keepdims=True)/1e6
        data['count'] = data['count']/scale_factor

    if data.get("covariates") is not None:
        cov = data.get("covariates")
        cov_name = ["covariate_" + str(i) for i in range(cov.shape[1])]
        dd.obs[cov_name] = cov

    if return_dict:
        return data,dd
    else:
        return dd


# Below are some functions used in calculating MMD loss

def compute_kernel(x, y, kernel='rbf', **kwargs):
    """Computes RBF kernel between x and y.

    Parameters
    ----------
        x: Tensor
            Tensor with shape [batch_size, z_dim]
        y: Tensor
            Tensor with shape [batch_size, z_dim]

    Returns
    ----------
        The computed RBF kernel between x and y
    """
    scales = kwargs.get("scales", [])
    if kernel == "rbf":
        x_size = K.shape(x)[0]
        y_size = K.shape(y)[0]
        dim = K.shape(x)[1]
        tiled_x = K.tile(K.reshape(x, K.stack([x_size, 1, dim])), K.stack([1, y_size, 1]))
        tiled_y = K.tile(K.reshape(y, K.stack([1, y_size, dim])), K.stack([x_size, 1, 1]))
        return K.exp(-K.mean(K.square(tiled_x - tiled_y), axis=2) / K.cast(dim, tf.float32))
    elif kernel == 'raphy':
        scales = K.variable(value=np.asarray(scales))
        squared_dist = K.expand_dims(squared_distance(x, y), 0)
        scales = K.expand_dims(K.expand_dims(scales, -1), -1)
        weights = K.eval(K.shape(scales)[0])
        weights = K.variable(value=np.asarray(weights))
        weights = K.expand_dims(K.expand_dims(weights, -1), -1)
        return K.sum(weights * K.exp(-squared_dist / (K.pow(scales, 2))), 0)
    elif kernel == "multi-scale-rbf":
        sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]

        beta = 1. / (2. * (K.expand_dims(sigmas, 1)))
        distances = squared_distance(x, y)
        s = K.dot(beta, K.reshape(distances, (1, -1)))

        return K.reshape(tf.reduce_sum(input_tensor=tf.exp(-s), axis=0), K.shape(distances)) / len(sigmas)


def squared_distance(x, y):
    '''Compute the pairwise euclidean distance.

    Parameters
    ----------
    x: Tensor
        Tensor with shape [batch_size, z_dim]
    y: Tensor
        Tensor with shape [batch_size, z_dim]

    Returns
    ----------
    The pairwise euclidean distance between x and y.
    '''
    r = K.expand_dims(x, axis=1)
    return K.sum(K.square(r - y), axis=-1)


def compute_mmd(x, y, kernel, **kwargs):
    """Computes Maximum Mean Discrepancy(MMD) between x and y.
    
    Parameters
    ----------
    x: Tensor
        Tensor with shape [batch_size, z_dim]
    y: Tensor
        Tensor with shape [batch_size, z_dim]
    kernel: str
        The kernel type used in MMD. It can be 'rbf', 'multi-scale-rbf' or 'raphy'.
    **kwargs: dict
        The parameters used in kernel function.
    
    Returns
    ----------
    The computed MMD between x and y
    """
    x_kernel = compute_kernel(x, x, kernel=kernel, **kwargs)
    y_kernel = compute_kernel(y, y, kernel=kernel, **kwargs)
    xy_kernel = compute_kernel(x, y, kernel=kernel, **kwargs)
    return K.mean(x_kernel) + K.mean(y_kernel) - 2 * K.mean(xy_kernel)


def sample_z(args):
    """Samples from standard Normal distribution with shape [size, z_dim] and
    applies re-parametrization trick. It is actually sampling from latent
    space distributions with N(mu, var) computed in `_encoder` function.
    
    Parameters
    ----------
    args: list
        List of [mu, log_var] computed in `_encoder` function.
        
    Returns
    ----------
    The computed Tensor of samples with shape [size, z_dim].
    """
    mu, log_var = args
    batch_size = K.shape(mu)[0]
    z_dim = K.int_shape(mu)[1]
    eps = K.random_normal(shape=[batch_size, z_dim])
    return mu + K.exp(log_var / 2) * eps


def _nan2zero(x):
    return tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)


def _nan2inf(x):
    return tf.where(tf.math.is_nan(x), tf.zeros_like(x) + np.inf, x)

def _nelem(x):
    nelem = tf.reduce_sum(input_tensor=tf.cast(~tf.math.is_nan(x), tf.float32))
    return tf.cast(tf.compat.v1.where(tf.equal(nelem, 0.), 1., nelem), x.dtype)


def _reduce_mean(x):
    nelem = _nelem(x)
    x = _nan2zero(x)
    return tf.divide(tf.reduce_sum(input_tensor=x), nelem)


