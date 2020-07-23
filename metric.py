import networkx as nx
import networkx.algorithms.isomorphism as iso
import netrd
import numpy as np
import pandas as pd

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
    score_GED = np.min([nx.graph_edit_distance(G_true, G_pred, node_match=comparison),
                        max_num_oper]) / max_num_oper
    res['score_GED'] = score_GED
        
    # 3. Hamming-Ipsen-Mikhailov distance
    if len(G_true)==len(G_pred):
        dist = netrd.distance.IpsenMikhailov()
        score_HIM = 1-dist.dist(G_true, G_pred)
    else:
        score_HIM = 0
    res['score_HIM'] = score_HIM
    return res
