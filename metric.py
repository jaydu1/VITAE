import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
from scipy.integrate import quad
from sklearn.metrics import pairwise_distances

def topology(G_true, G_pred):
    res = {}
    
    # 1. Isomorphism with same initial node
    def comparison(N1, N2):
        if N1['is_init'] != N2['is_init']:
            return False
        else:
            return True
    score_isomorphism = int(nx.is_isomorphic(G_true, G_pred, node_match=comparison))
    res['score_isomorphism'] = score_isomorphism
    
    # 2. GED (graph edit distance)
    max_num_oper = len(G_true)
    score_GED = 1 - np.min([nx.graph_edit_distance(G_true, G_pred, node_match=comparison),
                        max_num_oper]) / max_num_oper
    res['score_GED'] = score_GED
        
    # 3. Ipsen-Mikhailov distance
    if len(G_true)==len(G_pred):
        score_IM = 1 - IM_dist(G_true, G_pred)
    score_IM = np.maximum(0, score_IM)
    res['score_IM'] = score_IM
    return res


def IM_dist(G1, G2):
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



def get_RI_continuous(true, pred):
    '''
    Params:
        ture - [n_samples, n_cluster_1] for proportions or [n_samples, ] for grouping
        pred - [n_samples, n_cluster_2] for estimated proportions
    '''
    if len(true)!=len(pred):
        raise ValueError('Inputs should have same lengths!')
        
    if len(true.shape)==1:
        true = pd.get_dummies(true).values
    true = np.sqrt(true)
    pred = np.sqrt(pred)
    
    M_true = pairwise_distances(true, metric=lambda x,y:np.sum(x*y), n_jobs=-1)
    M_pred = pairwise_distances(pred, metric=lambda x,y:np.sum(x*y), n_jobs=-1)
    
    sum_triu = lambda A:(A.sum() - np.diag(A).sum())/2

    RI = sum_triu(1 - np.abs(M_true - M_pred)) / (len(true)*(len(true)-1)/2)
    return RI
