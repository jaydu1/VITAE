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
        dist = netrd.distance.HammingIpsenMikhailov()
        score_HIM = 1-dist.dist(G_true, G_pred, combination_factor=0.1)
    else:
        score_HIM = 0
    res['score_HIM'] = score_HIM
    return res

def compute_F1_score(true, pred):
    '''
    Params:
        true - array of true labels
        pred - array of predicted labels (no need to correspond to true labels)
    Return:
        F1 score
    '''
    milestones_df = pd.DataFrame({'group_true':true,'group_pred':pred})
    intersect_df = milestones_df.groupby(['group_true','group_pred']).size().reset_index().rename(columns={0: "n_intersect"})
    n_group_true_df = milestones_df.groupby(['group_true']).size().to_frame().rename(columns={0: "n_group_true"})
    n_group_pred_df = milestones_df.groupby(['group_pred']).size().to_frame().rename(columns={0: "n_group_pred"})
    intersect_df = intersect_df.join(n_group_true_df, on='group_true').join(n_group_pred_df, on='group_pred')
    intersect_df['jaccards'] = intersect_df['n_intersect']/(
        intersect_df['n_group_true']+intersect_df['n_group_pred']-intersect_df['n_intersect'])

    recovery = intersect_df.groupby('group_true')['jaccards'].max().mean()
    relevance = intersect_df.groupby('group_pred')['jaccards'].max().mean()
    return 2/(1/(recovery+1e-12)+1/(relevance+1e-12))