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

def build_milestone_net(subgraph, init_node):
    '''
    Args:
        subgraph     - a connected component of the graph, csr_matrix
        init_node    - root node
    Returns:
        df_subgraph  - dataframe of milestone network
    '''

    if len(subgraph)==1:
        warnings.warn('Singular node.')
        return []
    else:
        # Dijkstra's Algorithm
        unvisited = {node: {'parent':None,
                            'score':np.inf,
                            'distance':np.inf} for node in subgraph.nodes}
        current = init_node
        currentScore = 0
        currentDistance = 0
        unvisited[current]['score'] = currentScore

        milestone_net = []
        while True:
            for neighbour in subgraph.neighbors(current):
                if neighbour not in unvisited: continue
                newScore = currentScore + subgraph[current][neighbour]['weight']
                if unvisited[neighbour]['score'] > newScore:
                    unvisited[neighbour]['score'] = newScore
                    unvisited[neighbour]['parent'] = current
                    unvisited[neighbour]['distance'] = currentDistance+1

            if len(unvisited)<len(subgraph):
                milestone_net.append([unvisited[current]['parent'],
                                      current,
                                      unvisited[current]['distance']])
            del unvisited[current]
            if not unvisited: break
            current, currentScore, currentDistance = \
                sorted([(i[0],i[1]['score'],i[1]['distance']) for i in unvisited.items()],
                        key = lambda x: x[1])[0]
        return np.array(milestone_net)

def comp_pseudotime(G, node, df):
    connected_comps = nx.node_connected_component(G, node)
    subG = G.subgraph(connected_comps)
    milestone_net = build_milestone_net(subG,node)

    # compute pseudotime
    pseudotime = - np.ones(len(df))
    for i in range(len(milestone_net)):
        _from, _to = milestone_net[i,:2]
        _from, _to = int(_from), int(_to)

        idc = (df['from']==_from)&(df['to']==_to)
        if np.sum(idc)>0:
            pseudotime[idc] = df['percentage'].values[idc] + milestone_net[i,-1] - 1
            
        idc = (df['from']==_to)&(df['to']==_from)
        if np.sum(idc)>0:
            pseudotime[idc] = 1-df['percentage'].values[idc] + milestone_net[i,-1] - 1        

        if np.any(df['from']==_from):
            idc = (df['from']==_from)&(df['to']==_from)
            pseudotime[idc] =  milestone_net[i,-1] - 1
    if len(milestone_net)>0 and np.any((df['from']==_to)&(df['to']==_to)):
        idc = (df['from']==_to)&(df['to']==_to)
        pseudotime[idc] =  milestone_net[i,-1]
    return pseudotime

def topology(G_true, G_pred, is_GED=True):   
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
    if len(G_true)>10 or len(G_pred)>10:
        warnings.warn("Didn't calculate graph edit distances for large graphs.")
        res['GED score'] = np.nan  
    else:
        max_num_oper = len(G_true)
        score_GED = 1 - np.min([nx.graph_edit_distance(G_pred, G_true, node_match=comparison),
                            max_num_oper]) / max_num_oper
        res['GED score'] = score_GED
        
    # 3. Hamming-Ipsen-Mikhailov distance
    if len(G_true)==len(G_pred):
        score_IM = 1-IM_dist(G_true, G_pred)
        score_IM = np.maximum(0, score_IM)
    else:
        score_IM = 0
    res['score_IM'] = score_IM
    return res

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
    '''
    Params:
        ture - [n_samples, n_cluster_1] for proportions or [n_samples, ] for grouping
        pred - [n_samples, n_cluster_2] for estimated proportions
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

import numpy as np
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
import sys#; sys.path.insert(0, '../..')
sys.path.append('../..')
from VITAE import load_data

type_dict = {
    # dyno
    'dentate':'UMI', 
    'immune':'UMI', 
    'planaria_full':'UMI', 
    'planaria_muscle':'UMI',
    'aging':'non-UMI', 
    'cell_cycle':'non-UMI',
    'fibroblast':'non-UMI', 
    'germline':'non-UMI',    
    'human_embryos':'non-UMI', 
    'mesoderm':'non-UMI',
    
    # dyngen    
    "cycle_1":'non-UMI', 
    "cycle_2":'non-UMI', 
    "cycle_3":'non-UMI',
    "linear_1":'non-UMI', 
    "linear_2":'non-UMI', 
    "linear_3":'non-UMI', 
    "trifurcating_1":'non-UMI', 
    "trifurcating_2":'non-UMI', 
    "bifurcating_1":'non-UMI', 
    'bifurcating_2':'non-UMI',
    "bifurcating_3":'non-UMI', 
    "converging_1":'non-UMI',
    
    # our model
    'linear':'UMI',
    'bifurcation':'UMI',
    'multifurcating':'UMI',
    'tree':'UMI',
}
source_dict = {
    'dentate':'real', 
    'immune':'real', 
    'planaria_muscle':'real',
    'planaria_full':'real',
    'aging':'real', 
    'cell_cycle':'real',
    'fibroblast':'real', 
    'germline':'real',    
    'human_embryos':'real', 
    'mesoderm':'real',
        
    "cycle_1":'dyngen', 
    "cycle_2":'dyngen', 
    "cycle_3":'dyngen',
    "linear_1":'dyngen', 
    "linear_2":'dyngen', 
    "linear_3":'dyngen', 
    "trifurcating_1":'dyngen', 
    "trifurcating_2":'dyngen', 
    "bifurcating_1":'dyngen', 
    'bifurcating_2':'dyngen',
    "bifurcating_3":'dyngen', 
    "converging_1":'dyngen',
    
    'linear':'our model',
    'bifurcation':'our model',
    'multifurcating':'our model',
    'tree':'our model',
}
df = pd.DataFrame()
for data_name in type_dict.keys():
    data = load_data('data/', data_name)
    milestone_net = data['milestone_network']    
    le = LabelEncoder()
    grouping = le.fit_transform(data['grouping'])
    begin_node_true = le.transform([data['root_milestone_id']])[0]
    if 'w' in milestone_net.columns:
        grouping = None
    if milestone_net is not None:
        milestone_net['from'] = le.transform(milestone_net['from'])
        milestone_net['to'] = le.transform(milestone_net['to'])  
    G_true = nx.Graph()
    G_true.add_nodes_from(np.unique(milestone_net[['from','to']].values.flatten()))
    if grouping is None:
        G_true.add_edges_from(list(
            milestone_net[~pd.isna(milestone_net['w'])].groupby(['from', 'to']).count().index))
    # otherwise, 'milestone_net' indicates edges
    else:
        if milestone_net is not None:             
            G_true.add_edges_from(list(
                milestone_net.groupby(['from', 'to']).count().index))
    G_true.remove_edges_from(nx.selfloop_edges(G_true))
    nx.set_node_attributes(G_true, False, 'is_init')
    G_true.nodes[begin_node_true]['is_init'] = True

        
    for method in ['monocle','paga','slingshot']:
                
        _df = pd.read_csv('result/other methods/%s/%s_progressions.csv'%(data_name,method), 
                          index_col=0).astype({'from':str,'to':str})   
        pred_milestone_net = pd.read_csv('result/other methods/%s/%s_milestone_network.csv'%(data_name,method), 
                                         index_col=0).sort_index().astype({'from':str,'to':str})
        if method=='paga':
            _df['from'] = _df['from'].apply(lambda x:int(x.split('-')[0]) if type(x)==str else x)    
            _df['to'] = _df['to'].apply(lambda x:int(x.split('-')[0]) if type(x)==str else x)  
            pred_milestone_net['from'] = pred_milestone_net['from'].apply(
                lambda x:int(x.split('-')[0]) if type(x)==str else x)  
            pred_milestone_net['to'] = pred_milestone_net['to'].apply(
                lambda x:int(x.split('-')[0]) if type(x)==str else x)              

        le = LabelEncoder()
        le.fit(np.unique(pred_milestone_net[['from','to']].values.flatten()))
        pred_milestone_net['from'] = le.transform(pred_milestone_net['from'])
        pred_milestone_net['to'] = le.transform(pred_milestone_net['to'])
        _df['from'] = le.transform(_df['from'])
        _df['to'] = le.transform(_df['to'])
        
        G_pred = nx.Graph()
        G_pred.add_nodes_from(np.unique(_df[['from', 'to']].values.flatten()))
        G_pred.add_edges_from(pred_milestone_net[['from', 'to']].values)
        G_pred.remove_edges_from(nx.selfloop_edges(G_pred))  
        nx.set_node_attributes(G_pred, False, 'is_init')
        G_pred.nodes[pred_milestone_net.iloc[0]['from']]['is_init'] = True
        
        pseudotime = comp_pseudotime(nx.from_numpy_array(nx.to_numpy_array(G_pred)), 
                                     pred_milestone_net.iloc[0]['from'], 
                                     _df.copy())
        
        # 1. topology
        res = topology(G_true, G_pred)

        # 2. Milestones assignment
        if grouping is None:
            milestones_true = milestone_net['from'].values.copy()
            milestones_true[(milestone_net['from']!=milestone_net['to'])
                           &(milestone_net['w']<0.5)] = milestone_net[(milestone_net['from']!=milestone_net['to'])
                                                                      &(milestone_net['w']<0.5)]['to'].values
        else:
            milestones_true = grouping
        milestones_true = milestones_true[pseudotime!=-1]
        milestones_pred = _df['from'].values.copy()
        milestones_pred[_df['percentage']>0.5] = _df[_df['percentage']>0.5]['to'].values.copy()
        milestones_pred = milestones_pred[pseudotime!=-1]
        res['ARI'] = (adjusted_rand_score(milestones_true, milestones_pred) + 1)/2

        n_samples = len(pseudotime)
        w = np.zeros((n_samples,n_samples))
        w[np.arange(n_samples), _df['to']] = _df['percentage']
        w[np.arange(n_samples), _df['from']] = 1 - _df['percentage']
        if grouping is None:
            n_samples = len(milestone_net)
            prop = np.zeros((n_samples,n_samples))
            prop[np.arange(n_samples), milestone_net['to']] = 1-milestone_net['w']
            prop[np.arange(n_samples), milestone_net['from']] = np.where(np.isnan(milestone_net['w']), 1, milestone_net['w'])
            res['GRI'] = get_GRI(prop, w)
        else:
            res['GRI'] = get_GRI(grouping, w)
        
        # 3. Correlation between geodesic distances / Pseudotime   
        if 'cycle' in data_name or 'immune' in data_name:
            res['PDT score'] = np.nan
        else:
            if grouping is None:
                pseudotime_true = milestone_net['from'].values + 1 - milestone_net['w'].values
                pseudotime_true[np.isnan(pseudotime_true)] = milestone_net[pd.isna(milestone_net['w'])]['from'].values            
            else:
                pseudotime_true = - np.ones(len(grouping))
                nx.set_edge_attributes(G_true, values = 1, name = 'weight')
                connected_comps = nx.node_connected_component(G_true, begin_node_true)
                subG = G_true.subgraph(connected_comps)
                milestone_net_true = build_milestone_net(subG, begin_node_true)
                if len(milestone_net_true)>0:
                    pseudotime_true[grouping==int(milestone_net_true[0,0])] = 0
                    for i in range(len(milestone_net_true)):
                        pseudotime_true[grouping==int(milestone_net_true[i,1])] = milestone_net_true[i,-1]
            pseudotime_true = pseudotime_true[pseudotime>-1]
            pseudotime_pred = pseudotime[pseudotime>-1]
            if np.std(pseudotime_true)==0 or np.std(pseudotime_pred)==0:
                res['PDT score'] = 0
            else:
                res['PDT score'] = (np.corrcoef(pseudotime_true,pseudotime_pred)[0,1]+1)/2
    
        res['method'] = method
        res['data'] = data_name            
        res['source'] = source_dict[data_name]       
        df = df.append(pd.DataFrame(res, index=[0]),ignore_index=True)        

df.to_csv('result/result_other_methods.csv')
