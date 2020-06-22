import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
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
        
        
    def build_graphs(self, pc_x, thres=0.5, method='mean'):
        graph = np.zeros((self.NUM_CLUSTER,self.NUM_CLUSTER))
        if method=='mean':
            for i in range(self.NUM_CLUSTER-1):
                for j in range(i+1,self.NUM_CLUSTER):
                    idx = np.sum(pc_x[:,self.C[[i,i,j],[i,j,j]]], axis=1)>thres
                    if np.sum(idx)>0:
                        graph[i,j] = np.sum(pc_x[idx,self.C[i,j]])/np.sum(pc_x[idx][:,self.C[[i,i,j],[i,j,j]]])
        elif method=='map':
            c = np.argmax(pc_x, axis=-1)
            for i in range(self.NUM_CLUSTER-1):
                for j in range(i+1,self.NUM_CLUSTER):
                    if np.sum(c==self.C[i,j])>0:
                        graph[i,j] = np.sum(c==self.C[i,j])/np.sum((c==self.C[i,j])|(c==self.C[i,i])|(c==self.C[j,j]))
        else:
            raise ValueError("Invalid method, must be either 'mean' or 'map'.")
                    
        G = nx.from_numpy_array(graph)
        
        if self.no_loop and not nx.is_tree(G):
            # prune and merge points if there are loops
            G = nx.maximum_spanning_tree(G)
            
        return G


    def modify_wtilde(self, w_tilde, edges):
        w = np.zeros_like(w_tilde)
        
        # projection on edges
        idc = np.tile(np.arange(w.shape[0]), (2,1)).T
        ide = edges[np.argmax(np.sum(w_tilde[:,edges], axis=-1)**2 -
                              4 * np.prod(w_tilde[:,edges], axis=-1) +
                              2*np.sum(w_tilde[:,edges], axis=-1), axis=-1)]
        w[idc, ide] = w_tilde[idc, ide] + (1-np.sum(w_tilde[idc, ide], axis=-1, keepdims=True))/2
        best_proj_err_edge = np.sum(w_tilde**2, axis=-1) - np.sum(w_tilde[idc, ide]**2, axis=-1) + (1-np.sum(w_tilde[idc, ide], axis=-1))**2/2

        # projection on nodes
        best_proj_err_node = np.sum(w_tilde**2, axis=-1) - 2*np.max(w_tilde, axis=-1) +1
        best_proj_err_node_ind = np.argmax(w_tilde, axis=-1)

        idc = (best_proj_err_node<best_proj_err_edge)
        w[idc,:] = np.eye(w_tilde.shape[-1])[best_proj_err_node_ind[idc]]
        return w


    def get_umap(self, z, mu, proj_z_M=None):
        concate_z = np.concatenate((z, mu.T), axis=0)
        mapper = umap.UMAP().fit(concate_z)
        embed_z = mapper.embedding_[:-self.NUM_CLUSTER,:].copy()
        embed_mu = mapper.embedding_[-self.NUM_CLUSTER:,:].copy()
        if proj_z_M is None:
            embed_edge = None
        else:
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


    def init_inference(self, w_tilde, pc_x, thres=0.5, method='mean', no_loop=False):
        self.no_loop = no_loop
        self.w_tilde = w_tilde
        
        # Build graph
        self.G = self.build_graphs(pc_x, thres=thres, method=method)
        edges = np.array(self.G.edges)
        self.edges = [self.C[edges[i,0], edges[i,1]] for i in range(len(edges))]

        return self.G, self.edges
        
    
    def init_embedding(self, z, mu, proj_c, proj_z_M):
        self.mu = mu.copy()
        # Umap
        self.embed_z, self.embed_mu, embed_edges = self.get_umap(z, mu, proj_z_M)

        # Smooth lines
        self.lines = self.smooth_line(self.edges, self.embed_mu, embed_edges, proj_c)
        return None
        
        
    def plot_clusters(self, labels):
        if labels is None:
            print('No clustering labels available!')            
        else:
            n_labels = len(np.unique(labels))
            colors = [plt.cm.jet(float(i)/n_labels) for i in range(n_labels)]
            
            fig, ax = plt.subplots(1,1, figsize=(10, 5))
            for i,x in enumerate(np.unique(labels)):
                ax.scatter(*self.embed_z[labels==x].T, c=[colors[i]],
                    s=3, alpha=0.8, label=str(x))
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                             box.width, box.height * 0.9])
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                      fancybox=True, shadow=True, markerscale=5,
                      ncol=5)
            ax.set_title('Cluster Membership')
            plt.setp(ax, xticks=[], yticks=[])
            plt.show()
        return None
        
    def build_milestone_net(self, subgraph, init_node):
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
    
    
    def comp_pseudotime(self, node, w):
        connected_comps = nx.node_connected_component(self.G, node)
        subG = self.G.subgraph(connected_comps)
        milestone_net = self.build_milestone_net(subG,node)

        # compute pseudotime
        pseudotime = - np.ones(w.shape[0])
        for i in range(len(milestone_net)):
            _from, _to = milestone_net[i,:2]
            _from, _to = int(_from), int(_to)
            
            idc = (w[:,_from]>0)&(w[:,_to]>0)
            pseudotime[idc] = w[idc,_to] + milestone_net[i,-1] - 1
        
        return pseudotime


    def plot_trajectory(self, node, labels=None, cutoff=None):
        # select edges
        if len(self.edges)==0:
            select_edges = select_ind_edges = []
        else:
            if cutoff is None:
                G = nx.maximum_spanning_tree(self.G)
            else:
                graph = nx.to_numpy_matrix(self.G)
                G = nx.from_numpy_array(graph[graph>cutoff])
            select_edges = np.array(G.edges)
            select_ind_edges = [self.C[select_edges[i,0], select_edges[i,1]] for i in range(len(select_edges))]
        
        # modify w_tilde
        w = self.modify_wtilde(self.w_tilde, select_edges)
        
        # compute pseudotime
        pseudotime = self.comp_pseudotime(node, w)
        
        fig, ax = plt.subplots(1,1, figsize=(10, 5))
            
        cmap = matplotlib.cm.get_cmap('viridis')
        colors = [plt.cm.jet(float(i)/self.NUM_CLUSTER) for i in range(self.NUM_CLUSTER)]
        if np.sum(pseudotime>-1)>0:
            norm = matplotlib.colors.Normalize(vmin=np.min(pseudotime[pseudotime>-1]), vmax=np.max(pseudotime))
            sc = ax.scatter(*self.embed_z[pseudotime>-1,:].T,
                norm=norm,
                c=pseudotime[pseudotime>-1],
                s=2, alpha=0.5)
            plt.colorbar(sc, ax=[ax], location='right')
        else:
            norm = None
            
        if np.sum(pseudotime==-1)>0:
            ax.scatter(*self.embed_z[pseudotime==-1,:].T,
                        c='gray', s=1, alpha=0.4)
        
        for i in select_ind_edges:
            ax.plot(*self.lines[i].T, color="black", alpha=0.5)
        
        for i in range(len(self.CLUSTER_CENTER)):
            ax.scatter(*self.embed_mu[i:i+1,:].T, c=[colors[i]],
                        edgecolors='white', # linewidths=10,
                        norm=norm,
                        s=250, marker='*', label=str(i))
        plt.setp(ax, xticks=[], yticks=[])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                            box.width, box.height * 0.9])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
            fancybox=True, shadow=True, ncol=5)
        
        ax.set_title('Trajectory')
        plt.show()
