import os
import warnings
from typing import Optional

import pandas as pd
import numpy as np
from scipy.interpolate import splrep, splev
import networkx as nx
import umap
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from VITAE.utils import get_embedding, _get_smooth_curve

class Inferer(object):
    '''
    The class for doing inference based on posterior estimations.
    '''

    def __init__(self, NUM_CLUSTER: int):
        '''
        Parameters
        ----------
        NUM_CLUSTER : int
            The number of vertices in the latent space.
        '''        
        self.NUM_CLUSTER = NUM_CLUSTER
        self.NUM_STATE = int(NUM_CLUSTER*(NUM_CLUSTER+1)/2)
        self.CLUSTER_CENTER = np.array([int((NUM_CLUSTER+(1-i)/2)*i) for i in range(NUM_CLUSTER)])
        self.A, self.B = np.nonzero(np.triu(np.ones(NUM_CLUSTER)))
        self.C = np.triu(np.ones(NUM_CLUSTER))
        self.C[self.C>0] = np.arange(self.NUM_STATE)
        self.C = self.C.astype(int)
        
    def build_graphs(self, pc_x, method: str = 'mean', thres: float = 0.5):
        '''Build the backbone.
        
        Parameters
        ----------
        pc_x : np.array
            \([N, K]\) The estimated \(p(c_i|Y_i,X_i)\).        
        method : string, optional
            'mean', 'modified_mean', 'map', or 'modified_map'.
        thres : float, optional 
            The threshold used for filtering edges \(e_{ij}\) that \((n_{i}+n_{j}+e_{ij})/N<thres\), only applied to mean method.

        Retruns
        ----------
        G : nx.Graph
            The graph of edge scores.
        '''
        graph = np.zeros((self.NUM_CLUSTER,self.NUM_CLUSTER))
        if method=='mean':
            for i in range(self.NUM_CLUSTER-1):
                for j in range(i+1,self.NUM_CLUSTER):
                    idx = np.sum(pc_x[:,self.C[[i,i,j],[i,j,j]]], axis=1)>=thres
                    if np.sum(idx)>0:
                        graph[i,j] = np.mean(pc_x[idx,self.C[i,j]]/np.sum(pc_x[idx][:,self.C[[i,i,j],[i,j,j]]], axis=-1))
        elif method=='modified_mean':
            for i in range(self.NUM_CLUSTER-1):
                for j in range(i+1,self.NUM_CLUSTER):
                    idx = np.sum(pc_x[:,self.C[[i,i,j],[i,j,j]]], axis=1)>=thres
                    if np.sum(idx)>0:
                        graph[i,j] = np.sum(pc_x[idx,self.C[i,j]])/np.sum(pc_x[idx][:,self.C[[i,i,j],[i,j,j]]])
        elif method=='map':
            c = np.argmax(pc_x, axis=-1)
            for i in range(self.NUM_CLUSTER-1):
                for j in range(i+1,self.NUM_CLUSTER):
                    if np.sum(c==self.C[i,j])>0:
                        graph[i,j] = np.sum(c==self.C[i,j])/np.sum((c==self.C[i,j])|(c==self.C[i,i])|(c==self.C[j,j]))
        elif method=='modified_map':
            c = np.argmax(pc_x, axis=-1)
            for i in range(self.NUM_CLUSTER-1):
                for j in range(i+1,self.NUM_CLUSTER):
                    graph[i,j] = np.sum(c==self.C[i,j])/(np.sum((self.w_tilde[:,i]>0.5)|(self.w_tilde[:,j]>0.5))+1e-16)
        else:
            raise ValueError("Invalid method, must be one of 'mean', 'modified_mean', 'map', and 'modified_map'.")
                    
        G = nx.from_numpy_array(graph)
        
        if self.no_loop and not nx.is_tree(G):
            # prune and merge points if there are loops
            G = nx.maximum_spanning_tree(G)
            
        return G

    def modify_wtilde(self, w_tilde, edges):
        '''Project \(\\tilde{w}\) to the estimated backbone.
        
        Parameters
        ----------
        w_tilde : np.array
            \([N, k]\) The estimated \(\\tilde{w}\).        
        edges : np.array
            \([|\\mathcal{E}(\\widehat{\\mathcal{B}})|, 2]\).
        thres : float, optional 
            The threshold used for filtering edges \(e_{ij}\) that \((n_{i}+n_{j}+e_{ij})/N<thres\), only applied to mean method.

        Retruns
        ----------
        w : np.array
            The projected \(\\tilde{w}\).
        '''
        w = np.zeros_like(w_tilde)
        
        # projection on nodes
        best_proj_err_node = np.sum(w_tilde**2, axis=-1) - 2*np.max(w_tilde, axis=-1) +1
        best_proj_err_node_ind = np.argmax(w_tilde, axis=-1)
        
        if len(edges)>0:
            # projection on edges
            idc = np.tile(np.arange(w.shape[0]), (2,1)).T
            ide = edges[np.argmax(np.sum(w_tilde[:,edges], axis=-1)**2 -
                                  4 * np.prod(w_tilde[:,edges], axis=-1) +
                                  2 * np.sum(w_tilde[:,edges], axis=-1), axis=-1)]
            w[idc, ide] = w_tilde[idc, ide] + (1-np.sum(w_tilde[idc, ide], axis=-1, keepdims=True))/2
            best_proj_err_edge = np.sum(w_tilde**2, axis=-1) - np.sum(w_tilde[idc, ide]**2, axis=-1) + (1-np.sum(w_tilde[idc, ide], axis=-1))**2/2
                         
            idc = (best_proj_err_node<best_proj_err_edge)
            w[idc,:] = np.eye(w_tilde.shape[-1])[best_proj_err_node_ind[idc]]
        else:
            idc = np.arange(w.shape[0])
            w[idc, best_proj_err_node_ind] = 1
        return w

    def init_inference(self, w_tilde, pc_x, thres: float = 0.5, method: str = 'mean', no_loop: bool = False):
        '''Initialze inference.
        
        Parameters
        ----------
        w_tilde : np.array
            \([N, k]\) The estimated \(\\tilde{w}\).        
        pc_x : np.array
            \([N, K]\) The estimated \(p(c_i|Y_i,X_i)\).  
        method : string, optional
            'mean', 'modified_mean', 'map', or 'modified_map'.
        thres : float, optional 
            The threshold used for filtering edges \(e_{ij}\) that \((n_{i}+n_{j}+e_{ij})/N<thres\), only applied to mean method.    
        no_loop : boolean, optional 
            if loops are allowed to exist in The graph.

        Retruns
        ----------
        G : np.array
            The estimated backbone \(\\widehat{\\mathcal{B}}\).
        edges : np.array
            \(|\\mathcal{E}(\\widehat{\\mathcal{B}})|,2\) The edges in the estimated backbone.
        '''
        self.no_loop = no_loop
        self.w_tilde = w_tilde
        
        # Build graph
        self.G = self.build_graphs(pc_x, method=method, thres=thres)
        
        edges = np.array(list(self.G.edges))
        self.edges = [self.C[edges[i,0], edges[i,1]] for i in range(len(edges))]

        return self.G, self.edges
    
    def init_embedding(self, z, mu, dimred: str ='umap', **kwargs):
        '''Initialze embeddings for visualizations.

        Parameters
        ----------
        z : np.array
            \([N,d]\) The latent means.
        mu : np.array
            \([d,k]\) The value of initial \(\\mu\).
        dimred : str, optional 
            The name of dimension reduction algorithms, can be 'umap', 'pca' and 'tsne'. Only used if 'plot_every_num_epoch' is not None. 
        **kwargs : 
            Extra key-value arguments for dimension reduction algorithms.   

        Retruns
        ----------
        embed_z : np.array
            \([N, 2]\) latent variables after dimension reduction.
        '''
        self.mu = mu.copy()
        concate_z = np.concatenate((z, mu.T), axis=0)
        embed = get_embedding(concate_z, dimred, **kwargs)
        
        self.embed_z = embed[:-self.NUM_CLUSTER,:]
        self.embed_mu = embed[-self.NUM_CLUSTER:,:]
        return self.embed_z.copy()    
        
    def plot_clusters(self, labels, plot_labels: bool=False, path: Optional[str] = None):
        '''Plot the embeddings with labels.

        Parameters
        ----------
        labels : np.array     
            \([N, ]\) The clustered labels.
        plot_labels : boolean, optional
            Whether to plot text of labels or not.
        path : str, optional
            The path to save the figure.
        '''  
        if labels is None:
            print('No clustering labels available!')
        else:
            n_labels = len(np.unique(labels))
            colors = [plt.cm.jet(float(i)/n_labels) for i in range(n_labels)]
            
            fig, ax = plt.subplots(1,1, figsize=(20, 10))
            for i,x in enumerate(np.unique(labels)):
                ax.scatter(*self.embed_z[labels==x].T, c=[colors[i]],
                    s=8, alpha=0.6, label=str(x))
                if plot_labels:
                    ax.text(np.mean(self.embed_z[labels==x,0]), 
                            np.mean(self.embed_z[labels==x,1]), str(x), fontsize=16)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                             box.width, box.height * 0.9])
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                      fancybox=True, shadow=True, markerscale=5, ncol=5)
            ax.set_title('Cluster Membership')
            plt.setp(ax, xticks=[], yticks=[])
            if path is not None:
                plt.savefig(path, dpi=300)
            plt.show()
        return None
        
    def build_milestone_net(self, subgraph, init_node: int):
        '''Build the milestone network.

        Parameters
        ----------
        subgraph : nx.Graph
            The connected component of the backbone given the root vertex.
        init_node : int
            The root vertex.
        
        Returns
        ----------
        df_subgraph : pd.DataFrame 
            The milestone network.
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
    
    def comp_pseudotime(self, milestone_net, init_node: int, w):
        '''Compute pseudotime.

        Parameters
        ----------
        milestone_net : pd.DataFrame
            The milestone network.
        init_node : int
            The root vertex.
        w : np.array
            \([N, k]\) The projected \(\\tilde{w}\).
        
        Returns
        ----------
        pseudotime : np.array
            \([N, k]\) The estimated pseudtotime.
        '''
        pseudotime = - np.ones(w.shape[0])
        pseudotime[w[:,init_node]==1] = 0
        
        if len(milestone_net)>0:
            for i in range(len(milestone_net)):
                _from, _to = milestone_net[i,:2]
                _from, _to = int(_from), int(_to)

                idc = ((w[:,_from]>0)&(w[:,_to]>0)) | (w[:,_to]==1)
                pseudotime[idc] = w[idc,_to] + milestone_net[i,-1] - 1
        
        return pseudotime


    def infer_trajectory(self, init_node: int, labels = None, cutoff: Optional[float] = None, is_plot: bool = True, path: Optional[str] = None):
        '''Infer the trajectory.        

        Parameters
        ----------
        init_node : int
            The initial node for the inferred trajectory.
        cutoff : string, optional
            The threshold for filtering edges with scores less than cutoff.
        is_plot : boolean, optional
            Whether to plot or not.
        path : string, optional  
            The path to save figure, or don't save if it is None.

        Returns
        ----------
        G : nx.Graph 
            The modified graph that indicates the inferred trajectory.
        w : np.array
            \([N,k]\) The modified \(\\tilde{w}\).
        pseudotime : np.array
            \([N,]\) The pseudotime based on projected trajectory.      
        '''
        # select edges
        if len(self.edges)==0:
            select_edges = []
            G = nx.Graph()
            G.add_nodes_from(self.G.nodes)
        else:
            if self.no_loop:
                G = nx.maximum_spanning_tree(self.G)
            else:
                G = self.G
            if cutoff is None:
                cutoff = 0.01
            graph = nx.to_numpy_matrix(G)
            graph[graph<=cutoff] = 0
            G = nx.from_numpy_array(graph)
            connected_comps = nx.node_connected_component(G, init_node)
            subG = G.subgraph(connected_comps)
            if len(subG.edges)>0:                
                milestone_net = self.build_milestone_net(subG,init_node)
                select_edges = milestone_net[:,:2]
                select_edges_score = graph[select_edges[:,0], select_edges[:,1]]
                if select_edges_score.max() - select_edges_score.min() == 0:
                    select_edges_score = select_edges_score/select_edges_score.max()
                else:
                    select_edges_score = (select_edges_score - select_edges_score.min())/(select_edges_score.max() - select_edges_score.min())*3                    
            else:
                milestone_net = select_edges = []                    
        
        # modify w_tilde
        w = self.modify_wtilde(self.w_tilde, select_edges)
        
        # compute pseudotime
        pseudotime = self.comp_pseudotime(milestone_net, init_node, w)
        
        if is_plot:
            fig, ax = plt.subplots(1,1, figsize=(20, 10))
                
            cmap = matplotlib.cm.get_cmap('viridis')
            colors = [plt.cm.jet(float(i)/self.NUM_CLUSTER) for i in range(self.NUM_CLUSTER)]
            if np.sum(pseudotime>-1)>0:
                norm = matplotlib.colors.Normalize(vmin=np.min(pseudotime[pseudotime>-1]), vmax=np.max(pseudotime))
                sc = ax.scatter(*self.embed_z[pseudotime>-1,:].T,
                    norm=norm,
                    c=pseudotime[pseudotime>-1],
                    s=8, alpha=0.5)
                plt.colorbar(sc, ax=[ax], location='right')
            else:
                norm = None
                
            if np.sum(pseudotime==-1)>0:
                ax.scatter(*self.embed_z[pseudotime==-1,:].T,
                            c='gray', s=8, alpha=0.4)
            
            for i in range(len(select_edges)):
                points = self.embed_z[np.sum(w[:,select_edges[i,:]]>0, axis=-1)==2,:]
                points = points[points[:,0].argsort()]                
                try:
                    x_smooth, y_smooth = _get_smooth_curve(
                        points, 
                        self.embed_mu[select_edges[i,:], :]
                        )
                except:
                    x_smooth, y_smooth = self.embed_mu[select_edges[i,:], 0], self.embed_mu[select_edges[i,:], 1]
                ax.plot(x_smooth, y_smooth, 
                    '-', 
                    linewidth= 1 + select_edges_score[0,i],
                    color="black", 
                    alpha=0.8, 
                    path_effects=[pe.Stroke(linewidth=1+select_edges_score[0,i]+1.5, 
                                            foreground='white'), pe.Normal()],
                    zorder=1
                    )

                delta_x = self.embed_mu[select_edges[i,1], 0]-x_smooth[-2]
                delta_y = self.embed_mu[select_edges[i,1], 1]-y_smooth[-2]
                length = np.sqrt(delta_x**2 + delta_y**2) * 1.5                
                ax.arrow(
                        self.embed_mu[select_edges[i,1], 0]-delta_x/length, 
                        self.embed_mu[select_edges[i,1], 1]-delta_y/length, 
                        delta_x/length,
                        delta_y/length,
                        color='black', alpha=1.0,
                        shape='full', lw=0, length_includes_head=True, head_width=0.4, zorder=2)
            
            for i in range(len(self.CLUSTER_CENTER)):
                ax.scatter(*self.embed_mu[i:i+1,:].T, c=[colors[i]],
                            edgecolors='white', # linewidths=10,
                            norm=norm,
                            s=250, marker='*', label=str(i))
                ax.text(self.embed_mu[i,0], self.embed_mu[i,1], '%02d'%i, fontsize=16)
                
            plt.setp(ax, xticks=[], yticks=[])
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                                box.width, box.height * 0.9])
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                fancybox=True, shadow=True, ncol=5)
            
            ax.set_title('Trajectory')
            if path is not None:
                plt.savefig(path, dpi=300)
            plt.show()
        return G, w, pseudotime
            