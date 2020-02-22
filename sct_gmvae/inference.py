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

with open('result.pkl', 'rb') as f:
    result = pk.load(f)
    
c,proj_c,embed_z,embed_mu,embed_edges,edges,edges_score,w,w_var = result    

NUM_CLUSTER = 5
n_states = int(NUM_CLUSTER*(NUM_CLUSTER+1)/2)
A, B = np.nonzero(np.triu(np.ones(NUM_CLUSTER)))
cluster_center = [int((NUM_CLUSTER+(1-i)/2)*i) for i in range(NUM_CLUSTER)]


'''
Trajectory
'''
# -----------------------------------------------------------------------
# Scoring edges
# -----------------------------------------------------------------------
df = pd.value_counts(list(c))/len(c)
_df = df[~df.index.isin(cluster_center)].to_frame()
def _score(r):
    i = r.name
    return r / np.min([df[cluster_center[A[i]]], df[cluster_center[B[i]]]])
_df['score'] = _df.apply(_score, axis=1)
edges_score = np.array(_df['score'])

edges = np.array(edges)
edges_score = edges_score[np.argsort(edges)]
edges = np.sort(edges)

# -----------------------------------------------------------------------
# Smoothing lines
# -----------------------------------------------------------------------
lines = {}
for i in edges:
    data = np.concatenate((embed_mu[A[i]:A[i]+1,:], embed_edges[proj_c==i,:], embed_mu[B[i]:B[i]+1,:]), axis=0)
    x_range = np.sort([embed_mu[A[i],0], embed_mu[B[i],0]])
    y_range = np.sort([embed_mu[A[i],1], embed_mu[B[i],1]])
    data[data[:,0]<x_range[0],0] = x_range[0]
    data[data[:,0]>x_range[1],0] = x_range[1]
    data[data[:,1]<y_range[0],1] = y_range[0]
    data[data[:,1]>y_range[1],1] = y_range[1]
    
    w = np.ones(len(data))*0.01
    w[0] = w[-1] = 1
    
    if data.shape[0]==2:
        lines[i] = []        
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
        lines[i] = [x,y]

        

# -----------------------------------------------------------------------
# Interactive plot
# -----------------------------------------------------------------------
color_palette = inferno(n_states)
source = ColumnDataSource(dict(xs=[],ys=[],score=[]))
plot = figure(plot_height=800, plot_width=800, title="Embedding Latent Variables")
plot.circle(x=embed_z[:,0], y=embed_z[:,1], name='z', 
            size=2, color=[color_palette[i] for i in c], 
            line_color=None, fill_alpha=0.3)
plot.square(x=embed_mu[:,0], y=embed_mu[:,1], name='mu',
            size=15, color=[color_palette[i] for i in cluster_center], 
            line_color=None, fill_alpha=0.8)
glyph = MultiLine(xs="xs", ys="ys", name='edge',
                  line_color="red", line_width=2)
g1_r = plot.add_glyph(source, glyph)
hover = HoverTool(tooltips=[("score", "@score")], 
                  renderers=[g1_r],
                  mode='mouse')
plot.add_tools(hover)

def update_data(attrname, old, new):
    k = slider.value
    if k==0:
        source.data = dict(xs=[],ys=[],score=[])    
    else:
        ind = np.argsort(edges_score)[-k:]
        source.data = dict(xs=[lines[i][0] for i in edges[ind]],
                           ys=[lines[i][1] for i in edges[ind]],
                           score=edges_score[ind]) 
    return None

slider = Slider(start=0, end=len(edges), value=NUM_CLUSTER-1,
                step=1, title="Score")
slider.on_change('value', update_data)
update_data('value',NUM_CLUSTER-1,NUM_CLUSTER-1)

curdoc().add_root(row(slider, plot))
curdoc().title = "Sliders"



'''
Pseudotime
'''
# -----------------------------------------------------------------------
# Interactive plot
# -----------------------------------------------------------------------
graph = np.zeros((NUM_CLUSTER,NUM_CLUSTER), dtype=int)
select_edges = edges[np.argsort(edges_score)[-4:]]
graph[A[select_edges], B[select_edges]] = 1
graph = csr_matrix(graph)

names = np.arange(NUM_CLUSTER)

n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
print(labels)


# -----------------------------------------------------------------------
# Prune Graph
# -----------------------------------------------------------------------
Tcsr = minimum_spanning_tree(graph)
mst_graph = Tcsr + Tcsr.T
mst_graph = mst_graph.astype(int)


# -----------------------------------------------------------------------
# Build Milestone Network for Each Component
# -----------------------------------------------------------------------
def build_df_subgraph(subgraph, indexes, init_node):
    '''
    Args:
        subgraph     - a connected component of the graph, csr_matrix
        indexes      - indexes of each nodes in the original graphes
        init_node    - root node
    Returns:
        df_subgraph  - dataframe of milestone network
    '''

    _n_nodes = len(indexes)
    (idx, idy) = subgraph.nonzero()
    
    if subgraph.getnnz(0)[0]==0:
        raise Exception('Singular node.')
    else:
        # Dijkstra's Algorithm
        unvisited = {node: {'parent':None,
                           'distance':np.inf} for node in np.arange(_n_nodes)}
        current = init_node
        currentDistance = 0
        unvisited[current]['distance'] = currentDistance

        df_subgraph = pd.DataFrame(columns=['from', 'to', 'weight'])
        while True:
            for neighbour in idy[idx==current]:
                distance = subgraph[current, neighbour]

                if neighbour not in unvisited: continue
                newDistance = currentDistance + distance
                if unvisited[neighbour]['distance'] > newDistance:
                    unvisited[neighbour]['distance'] = newDistance
                    unvisited[neighbour]['parent'] = current

            if len(unvisited)<_n_nodes:
                df_subgraph = df_subgraph.append({'from':indexes[unvisited[current]['parent']],
                                            'to':indexes[current],
                                            'weight':unvisited[current]['distance']}, ignore_index=True)
            del unvisited[current]
            if not unvisited: break
            current, currentDistance = sorted([(i[0],i[1]['distance']) for i in unvisited.items()],
                                              key = lambda x: x[1])[0]
    return df_subgraph

df_subgraph = build_df_subgraph(mst_graph, np.arange(NUM_CLUSTER), 3)


# -----------------------------------------------------------------------
# Sort Points
# -----------------------------------------------------------------------
