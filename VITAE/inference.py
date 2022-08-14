import warnings
#from typing import Optional

import pandas as pd
import numpy as np
#from scipy.interpolate import splrep, splev
import networkx as nx
#import umap


class Inferer(object):
    '''
    The class for doing inference based on posterior estimations.
    '''

    def __init__(self, n_states: int):
        '''
        Parameters
        ----------
        n_states : int
            The number of vertices in the latent space.
        '''        
        self.n_states = n_states
        self.n_categories = int(n_states*(n_states+1)/2)
      #  self.A, self.B = np.nonzero(np.triu(np.ones(n_states)))
       ## indicator of the catagories
        self.C = np.triu(np.ones(n_states))
        self.C[self.C>0] = np.arange(self.n_categories)
        self.C = self.C.astype(int)
        
    def build_graphs(self, w_tilde, pc_x, method: str = 'mean', thres: float = 0.5, no_loop: bool = False, 
            cutoff = 0):
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
        self.no_loop = no_loop
    #    self.w_tilde = w_tilde

        graph = np.zeros((self.n_states,self.n_states))
        if method=='mean':
            for i in range(self.n_states-1):
                for j in range(i+1,self.n_states):
                    idx = np.sum(pc_x[:,self.C[[i,i,j],[i,j,j]]], axis=1)>=thres
                    if np.sum(idx)>0:
                        graph[i,j] = np.mean(pc_x[idx,self.C[i,j]]/np.sum(pc_x[idx][:,self.C[[i,i,j],[i,j,j]]], axis=-1))
        elif method=='modified_mean':
            for i in range(self.n_states-1):
                for j in range(i+1,self.n_states):
                    idx = np.sum(pc_x[:,self.C[[i,i,j],[i,j,j]]], axis=1)>=thres
                    if np.sum(idx)>0:
                        graph[i,j] = np.sum(pc_x[idx,self.C[i,j]])/np.sum(pc_x[idx][:,self.C[[i,i,j],[i,j,j]]])
        elif method=='map':
            c = np.argmax(pc_x, axis=-1)
            for i in range(self.n_states-1):
                for j in range(i+1,self.n_states):
                    if np.sum(c==self.C[i,j])>0:
                        graph[i,j] = np.sum(c==self.C[i,j])/np.sum((c==self.C[i,j])|(c==self.C[i,i])|(c==self.C[j,j]))
        elif method=='modified_map':
            c = np.argmax(pc_x, axis=-1)
            for i in range(self.n_states-1):
                for j in range(i+1,self.n_states):
                    graph[i,j] = np.sum(c==self.C[i,j])/(np.sum((w_tilde[:,i]>0.5)|(w_tilde[:,j]>0.5))+1e-16)
        elif method=='raw_map':
            c = np.argmax(pc_x, axis=-1)
            for i in range(self.n_states-1):
                for j in range(i+1,self.n_states):
                    if np.sum(c==self.C[i,j])>0:
                        graph[i,j] = np.sum(c==self.C[i,j])/np.sum(np.isin(c, np.diagonal(self.C)) == False)
        else:
            raise ValueError("Invalid method, must be one of 'mean', 'modified_mean', 'map', and 'modified_map'.")
        
        graph[graph<=cutoff] = 0
        G = nx.from_numpy_array(graph)
        
        if self.no_loop and not nx.is_tree(G):
            # prune if there are no loops
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
        pseudotime = np.empty(w.shape[0])
        pseudotime.fill(np.nan)
        pseudotime[w[:,init_node]==1] = 0
        
        if len(milestone_net)>0:
            for i in range(len(milestone_net)):
                _from, _to = milestone_net[i,:2]
                _from, _to = int(_from), int(_to)

                idc = ((w[:,_from]>0)&(w[:,_to]>0)) | (w[:,_to]==1)
                pseudotime[idc] = w[idc,_to] + milestone_net[i,-1] - 1
        
        return pseudotime


