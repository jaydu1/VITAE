import pandas as pd
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, HoverTool
from bokeh.models.glyphs import MultiLine
from bokeh.plotting import figure
from bokeh.palettes import inferno

import networkx as nx
import umap

def get_edges_score(c):
    df_states = pd.value_counts(list(c))/len(c)
    df_edges = df_states[~df_states.index.isin(CLUSTER_CENTER)].to_frame()
    if len(df_edges)==0:
        return None
    else:
        # To Do:
        # add score functions
        def _score(row):
            i = row.name
            # To Do:
            # What if df_states[cluster_center[A[i]]]=0?
            score = row / (0.01+
                           np.min([df_states[CLUSTER_CENTER[A[i]]], 
                                   df_states[CLUSTER_CENTER[B[i]]]]))
            return score
        df_edges['score'] = df_edges.apply(_score, axis=1)

        return df_edges


def build_graph(df_edges, no_loop=False):
    edges = list(df_edges.index)
    graph = np.zeros((NUM_CLUSTER,NUM_CLUSTER), dtype=int)
    graph[A[edges], B[edges]] = np.array(df_edges['score'])
    G = nx.from_numpy_array(graph)
    
#     if names:        
#         mapping = {i:names[i] for i in cluster_center}
#         G = nx.relabel_nodes(G, mapping)

    if no_loop:
        G = nx.minimum_spanning_tree(G)
    
    return G


def get_umap(z, mu, proj_z_M):
    concate_z = np.concatenate((z, mu.T), axis=0)
    mapper = umap.UMAP().fit(concate_z)
    embed_z = mapper.embedding_[:-NUM_CLUSTER,:].copy()    
    embed_mu = mapper.embedding_[-NUM_CLUSTER:,:].copy()
    embed_edge = mapper.transform(proj_z_M)
    return embed_z, embed_mu, embed_edge


def smooth_line(ind_edges, embed_mu, embed_edges, proj_c):
    lines = {}
    for i in ind_edges:
        data = np.concatenate((embed_mu[A[i]:A[i]+1,:], embed_edges[proj_c==i,:], embed_mu[B[i]:B[i]+1,:]), axis=0)
        x_range = np.sort(embed_mu[[A[i],B[i]],0])
        y_range = np.sort(embed_mu[[A[i],B[i]],1])
        data[data[:,0]<x_range[0],0] = x_range[0]
        data[data[:,0]>x_range[1],0] = x_range[1]
        data[data[:,1]<y_range[0],1] = y_range[0]
        data[data[:,1]>y_range[1],1] = y_range[1]

        w = np.ones(len(data))*0.01
        w[0] = w[-1] = 1

        if data.shape[0]==2:
            lines[i] = data

        else:
            if np.sum(np.abs(embed_mu[A[i],:]-embed_mu[B[i],:])*[1,-1])<0:
                w = w[np.argsort(data[:,1])]
                data = data[np.argsort(data[:,1]), :]
                x,y = data[:,0], data[:,1]        
                bspl = splrep(y, x, w, s=5)        
                x = splev(y, bspl)
            else:
                w = w[np.argsort(data[:,0])]
                data = data[np.argsort(data[:,0]), :]
                x,y = data[:,0], data[:,1]        
                bspl = splrep(x, y, w, s=5)        
                y = splev(x,bspl)
            lines[i] = np.c_[x,y]
    return lines


def comp_trajectory(c, proj_c, proj_z_M, no_loop=False):
    # Score edges
    df_edges = get_edges_score(c)
    if df_edges is None:
        # only plot nodes
        return None

    
    # Build graph
    G = build_graph(df_edges, no_loop)
    ind_edges = np.array([C[e] for e in G.edges])
    df_edges = df_edges[df_edges.index.isin(ind_edges)]
    edges_score = np.array(df_edges['score'])
    
    # Umap
    embed_z, embed_mu, embed_edges = get_umap(z, mu, proj_z_M)

    # Smooth lines
    lines = smooth_line(ind_edges, embed_mu, embed_edges, proj_c)
    
    return embed_z, embed_mu, ind_edges, edges_score, lines
    
def plot_trajectory(embed_z, embed_mu, c, ind_edges, edges_score, lines, cutoff=None):
    if cutoff is None:
        select_edges = ind_edges[np.argsort(edges_score)[-NUM_CLUSTER+1:]]
    else:
        select_edges = ind_edges[edges_score>=cutoff]
    colors = [plt.cm.jet(float(i)/NUM_STATE) for i in range(NUM_STATE)]
    
    fig, ax = plt.subplots(1, figsize=(7, 5))
    plt.scatter(*embed_z.T, c=np.array([colors[i] for i in c]), s=1, alpha=0.1)
    for i in select_edges:
        plt.plot(*lines[i].T, color="black", alpha=0.5)
        
    for idx,i in enumerate(CLUSTER_CENTER):
        plt.scatter(*embed_mu[idx:idx+1,:].T, c=[colors[i]],
                    s=100, marker='*', label=str(idx))
    plt.setp(ax, xticks=[], yticks=[])
    plt.legend()
    plt.show()


    
with open('result.pkl', 'rb') as f:
    result = pk.load(f)

NUM_CLUSTER = 5 
NUM_STATE = int(NUM_CLUSTER*(NUM_CLUSTER+1)/2)
CLUSTER_CENTER = [int((NUM_CLUSTER+(1-i)/2)*i) for i in range(NUM_CLUSTER)]    
A, B = np.nonzero(np.triu(np.ones(NUM_CLUSTER)))
C = np.triu(np.ones(NUM_CLUSTER))
C[C>0] = np.arange(NUM_STATE)
C = C.astype(int)


   
c,proj_c,proj_z_M,pi,mu,c,w,var_w,wc,var_wc,z,proj_z = result
embed_z, embed_mu, ind_edges, edges_score, lines = comp_trajectory(c, proj_c, proj_z_M)
plot_trajectory(embed_z, embed_mu, c, ind_edges, edges_score, lines, cutoff=None)


# '''
# Pseudotime
# '''
# # -----------------------------------------------------------------------
# # Interactive plot
# # -----------------------------------------------------------------------
# graph = np.zeros((NUM_CLUSTER,NUM_CLUSTER), dtype=int)
# select_edges = edges[np.argsort(edges_score)[-4:]]
# graph[A[select_edges], B[select_edges]] = 1
# graph = csr_matrix(graph)

# names = np.arange(NUM_CLUSTER)

# n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
# print(labels)


# # -----------------------------------------------------------------------
# # Prune Graph
# # -----------------------------------------------------------------------
# Tcsr = minimum_spanning_tree(graph)
# mst_graph = Tcsr + Tcsr.T
# mst_graph = mst_graph.astype(int)


# # -----------------------------------------------------------------------
# # Build Milestone Network for Each Component
# # -----------------------------------------------------------------------
# def build_df_subgraph(subgraph, indexes, init_node):
#     '''
#     Args:
#         subgraph     - a connected component of the graph, csr_matrix
#         indexes      - indexes of each nodes in the original graphes
#         init_node    - root node
#     Returns:
#         df_subgraph  - dataframe of milestone network
#     '''

#     _n_nodes = len(indexes)
#     (idx, idy) = subgraph.nonzero()
    
#     if subgraph.getnnz(0)[0]==0:
#         raise Exception('Singular node.')
#     else:
#         # Dijkstra's Algorithm
#         unvisited = {node: {'parent':None,
#                            'distance':np.inf} for node in np.arange(_n_nodes)}
#         current = init_node
#         currentDistance = 0
#         unvisited[current]['distance'] = currentDistance

#         df_subgraph = pd.DataFrame(columns=['from', 'to', 'weight'])
#         while True:
#             for neighbour in idy[idx==current]:
#                 distance = subgraph[current, neighbour]

#                 if neighbour not in unvisited: continue
#                 newDistance = currentDistance + distance
#                 if unvisited[neighbour]['distance'] > newDistance:
#                     unvisited[neighbour]['distance'] = newDistance
#                     unvisited[neighbour]['parent'] = current

#             if len(unvisited)<_n_nodes:
#                 df_subgraph = df_subgraph.append({'from':indexes[unvisited[current]['parent']],
#                                             'to':indexes[current],
#                                             'weight':unvisited[current]['distance']}, ignore_index=True)
#             del unvisited[current]
#             if not unvisited: break
#             current, currentDistance = sorted([(i[0],i[1]['distance']) for i in unvisited.items()],
#                                               key = lambda x: x[1])[0]
#     return df_subgraph

# df_subgraph = build_df_subgraph(mst_graph, np.arange(NUM_CLUSTER), 3)


# # -----------------------------------------------------------------------
# # Sort Points
# # -----------------------------------------------------------------------
