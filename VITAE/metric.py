import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
from scipy.integrate import quad
from sklearn.metrics import pairwise_distances
import warnings

import numba
from numba import jit, float32


def topology(G_true, G_pred):
    '''Evaulate topology metrics.

    Parameters
    ----------
    G_true : nx.Graph
        The reference graph.
    G_pred : nx.Graph
        The estimated graph.
    
    Returns
    ----------
    res : dict
        a dict containing evaulation results.
    '''     
    res = {}
    
    # 1. Isomorphism with same initial node
    def comparison(N1, N2):
        if N1['is_init'] != N2['is_init']:
            return False
        else:
            return True
    score_isomorphism = int(nx.is_isomorphic(G_true, G_pred, node_match=comparison))
    res['ISO score'] = score_isomorphism
    
    # 2. GED (graph edit distance)
    if len(G_true)>10:
        warnings.warn("Didn't calculate graph edit distances for large graphs.")
        res['GED score'] = np.nan  
    else:
        max_num_oper = len(G_true)
        GED = nx.graph_edit_distance(G_pred, G_true, 
                                node_match=comparison,
                                upper_bound=max_num_oper)
        if GED is None:
            res['GED score'] = 0
        else:            
            score_GED = 1 - GED / max_num_oper
            res['GED score'] = score_GED
        
    # 3. Ipsen-Mikhailov distance
    if len(G_true)==len(G_pred):
        score_IM = 1 - IM_dist(G_true, G_pred)
        score_IM = np.maximum(0, score_IM)
    else:
        score_IM = 0
    res['IM score'] = score_IM
    return res


def IM_dist(G1, G2):
    '''The Ipsen-Mikailov distance is a global (spectral) metric, 
    corresponding to the square-root of the squared difference of the
    Laplacian spectrum for each graph.

    Implementation adapt from
    https://netrd.readthedocs.io/en/latest/_modules/netrd/distance/hamming_ipsen_mikhailov.html

    Parameters
    ----------
    G1 : nx.Graph
    G2 : nx.Graph

    Returns
    ----------
    IM(G1,G2) : float
        The IM distance between G1 and G2.
    '''
    adj1 = nx.to_numpy_array(G1)
    adj2 = nx.to_numpy_array(G2)
    hwhm = 0.08
    
    N = len(adj1)
    # get laplacian matrix
    L1 = laplacian(adj1, normed=False)
    L2 = laplacian(adj2, normed=False)

    # get the modes for the positive-semidefinite laplacian
    w1 = np.sqrt(np.abs(eigh(L1)[0][1:]))
    w2 = np.sqrt(np.abs(eigh(L2)[0][1:]))

    # we calculate the norm for both spectrum
    norm1 = (N - 1) * np.pi / 2 - np.sum(np.arctan(-w1 / hwhm))
    norm2 = (N - 1) * np.pi / 2 - np.sum(np.arctan(-w2 / hwhm))

    # define both spectral densities
    density1 = lambda w: np.sum(hwhm / ((w - w1) ** 2 + hwhm ** 2)) / norm1
    density2 = lambda w: np.sum(hwhm / ((w - w2) ** 2 + hwhm ** 2)) / norm2

    func = lambda w: (density1(w) - density2(w)) ** 2
    return np.sqrt(quad(func, 0, np.inf, limit=100)[0])


@jit((float32[:,:], float32[:,:]), nopython=True, nogil=True)
def _rand_index(true, pred):
    n = true.shape[0]
    m_true = true.shape[1]
    m_pred = pred.shape[1]
    RI = 0.0
    for i in range(1, n-1):
        for j in range(i, n):
            RI_ij = 0.0
            for k in range(m_true):
                RI_ij += true[i,k]*true[j,k]
            for k in range(m_pred):
                RI_ij -= pred[i,k]*pred[j,k]
            RI += 1-np.abs(RI_ij)
    return RI / (n*(n-1)/2.0)


def get_GRI(true, pred):
    '''Compute the GRI.

    Parameters
    ----------
    ture : np.array
        [n_samples, n_cluster_1] for proportions or [n_samples, ] for grouping
    pred : np.array
        [n_samples, n_cluster_2] for estimated proportions or [n_samples, ] for grouping

    Returns
    ----------
    GRI : float
        The GRI of two groups of proportions in the trajectories.
    '''
    if len(true)!=len(pred):
        raise ValueError('Inputs should have same lengths!')
        
    if len(true.shape)==1:
        true = pd.get_dummies(true).values
    if len(pred.shape)==1:
        pred = pd.get_dummies(pred).values
    
    true = np.sqrt(true).astype(np.float32)
    pred = np.sqrt(pred).astype(np.float32)

    return _rand_index(true, pred)
