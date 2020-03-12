import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
from sklearn.metrics import pairwise_distances
import warnings
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
        # add null clusters
        null_clusters = pd.Series(0, index=[i for i in self.CLUSTER_CENTER
            if i not in df_states.index])  
        df_states = df_states.append(null_clusters)
        # compute score of edges
        df_edges = df_states[~df_states.index.isin(self.CLUSTER_CENTER)].to_frame()
        df_edges.index.names = ['edge']
        df_edges.columns = ['count']
        df_edges = df_edges.assign(**{'from': [self.A[i] for i in df_edges.index]})
        df_edges = df_edges.assign(**{'to': [self.B[i] for i in df_edges.index]})
        df_edges = df_edges.reindex(['from','to','count'], axis=1)
        if len(df_edges)==0:
            return None
        else:
            def max_relative_score(row):
                i = row.name
                score = row['count'] / (0.01+
                               np.min([df_states[self.CLUSTER_CENTER[self.A[i]]], 
                                       df_states[self.CLUSTER_CENTER[self.B[i]]]]))
                return score
            def mean_relative_score(row):
                i = row.name
                
                score = row['count'] / (0.01+df_states[self.CLUSTER_CENTER[self.A[i]]]+
                                         df_states[self.CLUSTER_CENTER[self.B[i]]])
                return score
            df_edges['max_relative_score'] = df_edges.apply(max_relative_score, axis=1)
            df_edges['mean_relative_score'] = df_edges.apply(mean_relative_score, axis=1)
            df_edges = df_edges.sort_values(self.metric, ascending=False)
            return df_edges


    def build_graph(self, df_edges):
        edges = list(df_edges.index)  
        graph = np.zeros((self.NUM_CLUSTER,self.NUM_CLUSTER))
        graph[self.A[edges], self.B[edges]] = np.array(df_edges[self.metric])
        G = nx.from_numpy_array(graph)

    #     if names:        
    #         mapping = {i:names[i] for i in cluster_center}
    #         G = nx.relabel_nodes(G, mapping)

        if self.no_loop and not nx.is_tree(G):
            # prune and merge points if there are loops            
            T = nx.maximum_spanning_tree(G)
            del_edges = [self.C[i] for i in G.edges if i not in T.edges]
            for i in del_edges:
                self.c[(self.c==i)&(self.w<0.5)] = self.A[i]
                self.c[(self.c==i)&(self.w>=0.5)] = self.B[i]    
            G = T
            
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


    def init_inference(self, c, w, mu, z, proj_c, proj_z_M, 
                       metric='max_relative_score', no_loop=False):
        self.c = c
        self.w = w     
        self.mu = mu             
        self.no_loop = no_loop
        self.metric = metric
        
        # Score edges
        df_edges = self.get_edges_score(c)        
        if df_edges is None:
            # only plot nodes
            return None

        # Build graph
        self.G = self.build_graph(df_edges)
        ind_edges = np.array([self.C[e] for e in self.G.edges])
        self.df_edges = df_edges[df_edges.index.isin(ind_edges)]        
        print(self.df_edges)
        
        # Umap
        self.embed_z, self.embed_mu, embed_edges = self.get_umap(z, mu, proj_z_M)

        # Smooth lines
        self.lines = self.smooth_line(ind_edges, self.embed_mu, embed_edges, proj_c)

    
    def plot_trajectory(self, labels, cutoff=None):
        if cutoff is None:
            self.select_edges = self.df_edges.sort_values(
                self.metric).iloc[-self.NUM_CLUSTER+1:]
        else:
            self.select_edges = self.df_edges[self.df_edges[self.metric]>=cutoff]
        
        
        if labels is None:
            fig, ax1 = plt.subplots(1, figsize=(7, 5))
        else:
            n_labels = len(np.unique(labels))
            colors = [plt.cm.jet(float(i)/n_labels) for i in range(n_labels)]
            labels = np.array(labels)
            fig, (ax1,ax2) = plt.subplots(1,2, figsize=(14, 5))
            for i,x in enumerate(np.unique(labels)):
                ax2.scatter(*self.embed_z[labels==x].T, c=[colors[i]],
                    s=3, alpha=0.8, label=str(x))
            box = ax2.get_position()
            ax2.set_position([box.x0, box.y0 + box.height * 0.1,
                             box.width, box.height * 0.9])
            ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                      fancybox=True, shadow=True, markerscale=5,
                      ncol=np.min([5,50//np.max([len(i) for i in np.unique(labels)])]))
            ax2.set_title('Ground Truth')
            plt.setp(ax2, xticks=[], yticks=[])
            ax1.set_title('Prediction')
            
        colors = [plt.cm.jet(float(i)/self.NUM_STATE) for i in range(self.NUM_STATE)]
        ax1.scatter(*self.embed_z.T, c=np.array([colors[i] for i in self.c]), s=1, alpha=0.5)
        for i in self.select_edges.index:
            ax1.plot(*self.lines[i].T, color="black", alpha=0.5)

        for idx,i in enumerate(self.CLUSTER_CENTER):
            ax1.scatter(*self.embed_mu[idx:idx+1,:].T,
                        c=[colors[i]], edgecolors='white',
                        s=200, marker='*', label=str(idx))
        plt.setp(ax1, xticks=[], yticks=[])
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                  fancybox=True, shadow=True, ncol=5)
        
        plt.suptitle('Trajectory')
        plt.show()  
        
    def build_milestone_net(self, subgraph, subG_mu, init_node):
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
    
    def plot_pseudotime(self, node):
        dist_mu = pairwise_distances(self.mu.T)
        
        G = self.build_graph(self.select_edges)
        connected_comps = nx.node_connected_component(G, node)
        subG = G.subgraph(connected_comps)
        subG_mu = nx.from_numpy_array(dist_mu).subgraph(connected_comps)
        
        # compute milestone network
        milestone_net = self.build_milestone_net(subG,subG_mu,node)

        # compute pseudotime
        pseudotime = - np.ones_like(self.c) 
        pseudotime_node = - np.ones_like(self.CLUSTER_CENTER) 
        for i in range(len(milestone_net)):
            _from, _to = milestone_net[i,:2]
            _from, _to = int(_from), int(_to)
            if i==0:
                pseudotime[self.c==self.CLUSTER_CENTER[_from]] = \
                    pseudotime_node[_from] = 0
            pseudotime[self.c==self.CLUSTER_CENTER[_to]] = \
                pseudotime_node[_to] = np.sum(milestone_net[:i+1,-1])

            flag = False
            if _from>_to:
                flag = True
                _from,_to = _to,_from
            state = self.C[_from,_to]

            if flag:
                pseudotime[self.c == state] = ((1-self.w[self.c == state]) * 
                                               milestone_net[i,-1] + 
                                               np.sum(milestone_net[:i,-1]))
            else:
                pseudotime[self.c == state] = (self.w[self.c == state] * 
                                               milestone_net[i,-1] + 
                                               np.sum(milestone_net[:i,-1]))

        fig, ax = plt.subplots(1, figsize=(8, 5))
             
        cmap = matplotlib.cm.get_cmap('viridis')

        if np.sum(pseudotime>-1)>0:
            norm = matplotlib.colors.Normalize(vmin=np.min(pseudotime[pseudotime>-1]), vmax=np.max(pseudotime))
            sc = plt.scatter(*self.embed_z[pseudotime>-1,:].T,
                norm=norm,
                c=pseudotime[pseudotime>-1],
                s=2, alpha=0.5)
            plt.colorbar(sc)
        else:
            norm = None
            
        if np.sum(pseudotime==-1)>0:
            plt.scatter(*self.embed_z[pseudotime==-1,:].T, 
                        c='gray', s=1, alpha=0.4)

        for idx,i in enumerate(self.CLUSTER_CENTER):
            if pseudotime_node[idx]==-1:
                c = 'gray'
            else:
                c = [cmap(norm(pseudotime_node[idx]))]
            plt.scatter(*self.embed_mu[idx:idx+1,:].T, c=c,
                        edgecolors='white', # linewidths=10,
                        norm=norm,
                        s=200, marker='*', label=str(idx))
                        
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
            fancybox=True, shadow=True, ncol=5)
        plt.setp(ax, xticks=[], yticks=[])
        plt.title('Pseudotime')
        plt.show()
