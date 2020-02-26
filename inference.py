import pandas as pd
import numpy as np
import pickle as pk
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
from sklearn.metrics import pairwise_distances

import networkx as nx
import umap

class Inferer(object):
    def __init__(self, NUM_CLUSTER):
        self.NUM_CLUSTER = NUM_CLUSTER
        self.NUM_STATE = int(NUM_CLUSTER*(NUM_CLUSTER+1)/2)
        self.CLUSTER_CENTER = np.array([int((NUM_CLUSTER+(1-i)/2)*i) for i in range(NUM_CLUSTER)])
        self.A, self.B = np.nonzero(np.triu(np.ones(NUM_CLUSTER)))
        self.C = np.triu(np.ones(NUM_CLUSTER))
        self.C[self.C>0] = np.arange(self.NUM_STATE)
        self.C = self.C.astype(int)
        
    def get_edges_score(self, c):
        df_states = pd.value_counts(list(c))/len(c)
        df_edges = df_states[~df_states.index.isin(self.CLUSTER_CENTER)].to_frame()
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
                               np.min([df_states[self.CLUSTER_CENTER[self.A[i]]], 
                                       df_states[self.CLUSTER_CENTER[self.B[i]]]]))
                return score
            df_edges['score'] = df_edges.apply(_score, axis=1)

            return df_edges


    def build_graph(self, df_edges, no_loop=False):
        edges = list(df_edges.index)
        graph = np.zeros((self.NUM_CLUSTER,self.NUM_CLUSTER), dtype=int)
        graph[self.A[edges], self.B[edges]] = np.array(df_edges['score'])
        G = nx.from_numpy_array(graph)

    #     if names:        
    #         mapping = {i:names[i] for i in cluster_center}
    #         G = nx.relabel_nodes(G, mapping)

        if no_loop:
            G = nx.minimum_spanning_tree(G)

        return G


    def get_umap(self, z, mu, proj_z_M):
        concate_z = np.concatenate((z, mu.T), axis=0)
        mapper = umap.UMAP().fit(concate_z)
        embed_z = mapper.embedding_[:-self.NUM_CLUSTER,:].copy()    
        embed_mu = mapper.embedding_[-self.NUM_CLUSTER:,:].copy()
        embed_edge = mapper.transform(proj_z_M)
        return embed_z, embed_mu, embed_edge


    def smooth_line(self, ind_edges, embed_mu, embed_edges, proj_c):
        lines = {}
        for i in ind_edges:
            data = np.concatenate((embed_mu[self.A[i]:self.A[i]+1,:], 
                                   embed_edges[proj_c==i,:], 
                                   embed_mu[self.B[i]:self.B[i]+1,:]), axis=0)
            x_range = np.sort(embed_mu[[self.A[i],self.B[i]],0])
            y_range = np.sort(embed_mu[[self.A[i],self.B[i]],1])
            data[data[:,0]<x_range[0],0] = x_range[0]
            data[data[:,0]>x_range[1],0] = x_range[1]
            data[data[:,1]<y_range[0],1] = y_range[0]
            data[data[:,1]>y_range[1],1] = y_range[1]

            w = np.ones(len(data))*0.01
            w[0] = w[-1] = 1

            if data.shape[0]==2:
                lines[i] = data

            else:
                if np.sum(np.abs(embed_mu[self.A[i],:]-embed_mu[self.B[i],:])*[1,-1])<0:
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


    def comp_trajectory(self, c, w, proj_c, proj_z_M, no_loop=False):
        self.c = c
        self.w = w        
        
        # Score edges
        df_edges = self.get_edges_score(c)
        if df_edges is None:
            # only plot nodes
            return None


        # Build graph
        self.G = self.build_graph(df_edges, no_loop)
        self.ind_edges = np.array([self.C[e] for e in self.G.edges])
        df_edges = df_edges[df_edges.index.isin(self.ind_edges)]
        self.edges_score = np.array(df_edges['score'])

        # Umap
        self.embed_z, self.embed_mu, self.embed_edges = self.get_umap(z, mu, proj_z_M)

        # Smooth lines
        self.lines = self.smooth_line(self.ind_edges, self.embed_mu, self.embed_edges, proj_c)

        return None

    
    def plot_trajectory(self, cutoff=None):
        if cutoff is None:
            select_edges = self.ind_edges[np.argsort(self.edges_score)[-self.NUM_CLUSTER+1:]]
        else:
            select_edges = self.ind_edges[self.edges_score>=cutoff]
        colors = [plt.cm.jet(float(i)/self.NUM_STATE) for i in range(self.NUM_STATE)]

        fig, ax = plt.subplots(1, figsize=(7, 5))
        plt.scatter(*self.embed_z.T, c=np.array([colors[i] for i in self.c]), s=1, alpha=0.1)
        for i in select_edges:
            plt.plot(*self.lines[i].T, color="black", alpha=0.5)

        for idx,i in enumerate(self.CLUSTER_CENTER):
            plt.scatter(*self.embed_mu[idx:idx+1,:].T, c=[colors[i]],
                        s=100, marker='*', label=str(idx))
        plt.setp(ax, xticks=[], yticks=[])
        plt.legend()
        plt.show()  
        
    def build_df_subgraph(self, subgraph, subG_mu, init_node):
        '''
        Args:
            subgraph     - a connected component of the graph, csr_matrix
            init_node    - root node
        Returns:
            df_subgraph  - dataframe of milestone network
        '''

        if len(subgraph)==1:
            raise Exception('Singular node.')
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
                        unvisited[neighbour]['distance'] = subG_mu[current][neighbour]['weight']

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
    
    def plot_pseudotime(self, mu, w, z, node, no_loop=False):
        dist_mu = pairwise_distances(mu.T)
        
        connected_comps = nx.node_connected_component(self.G, node)
        subG = self.G.subgraph(connected_comps)
        subG_mu = nx.from_numpy_array(dist_mu).subgraph(connected_comps)

        milestone_net = self.build_df_subgraph(subG,subG_mu,node)

        # compute pseudotime
        pseudotime = - np.ones_like(c) 
        pseudotime_node = - np.ones_like(self.CLUSTER_CENTER) 
        for i in range(len(milestone_net)):
            _from, _to = milestone_net[i,:2]
            _from, _to = int(_from), int(_to)
            if i==0:
                pseudotime[c==self.CLUSTER_CENTER[_from]] = \
                    pseudotime_node[_from] = 0
            pseudotime[c==self.CLUSTER_CENTER[_to]] = \
                pseudotime_node[_to] = np.sum(milestone_net[:i+1,-1])

            flag = False
            if _from>_to:
                flag = True
                _from,_to = _to,_from
            state = self.C[_from,_to]

            # bug 
            if flag:
                pseudotime[c == state] = (1-w[c == state]) * milestone_net[i,-1] + np.sum(milestone_net[:i,-1])
            else:
                pseudotime[c == state] = w[c == state] * milestone_net[i,-1] + np.sum(milestone_net[:i,-1])


        fig, ax = plt.subplots(1, figsize=(7, 5))

        norm = matplotlib.colors.Normalize(vmin=np.min(pseudotime[pseudotime>-1]), vmax=np.max(pseudotime))      
        cmap = matplotlib.cm.get_cmap('viridis')

        if np.sum(pseudotime==-1)>0:
            plt.scatter(*embed_z[pseudotime==-1,:].T, 
                        c='gray', s=1, alpha=0.4)
        sc = plt.scatter(*embed_z[pseudotime>-1,:].T, 
                         norm=norm,
                         c=pseudotime[pseudotime>-1],
                         s=1, alpha=0.4)

        for idx,i in enumerate(self.CLUSTER_CENTER):
            plt.scatter(*embed_mu[idx:idx+1,:].T, c=[cmap(norm(pseudotime_node[idx]))],
                        norm=normalize,
                        s=200, marker='*', label=str(idx))
        plt.setp(ax, xticks=[], yticks=[])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                  fancybox=True, shadow=True, ncol=5)
        plt.title('Pseudotime')
        plt.colorbar(sc)
        plt.show()
    
with open('result.pkl', 'rb') as f:
    result = pk.load(f)

NUM_CLUSTER = 5 

inferer = Inferer(NUM_CLUSTER)
   
c,proj_c,proj_z_M,pi,mu,c,w,var_w,wc,var_wc,z,proj_z = result
inferer.comp_trajectory(c, w, proj_c, proj_z_M, no_loop=True)
inferer.plot_trajectory(cutoff=None)
inferer.plot_pseudotime(mu, w, z, 1, no_loop=False)