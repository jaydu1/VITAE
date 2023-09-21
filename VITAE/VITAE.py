from typing import Optional, Union
import warnings
import os

import numpy as np
import pandas as pd
from scipy import stats

import VITAE.model as model 
import VITAE.train as train 
from VITAE.inference import Inferer
from VITAE.utils import get_igraph, leidenalg_igraph, \
   DE_test, _comp_dist, _get_smooth_curve
from VITAE.metric import topology, get_GRI
import tensorflow as tf

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

import scanpy as sc
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


class VITAE():
    """
    Variational Inference for Trajectory by AutoEncoder.
    """
    def __init__(self, adata: sc.AnnData,
               covariates = None, pi_covariates = None,
               model_type: str = 'Gaussian',
               npc: int = 64,
               adata_layer_counts = None,
               copy_adata: bool = False,
               hidden_layers = [32],
               latent_space_dim: int = 16,
               conditions = None):
        '''
        Get input data for model. Data need to be first processed using scancy and stored as an AnnData object
         The 'UMI' or 'non-UMI' model need the original count matrix, so the count matrix need to be saved in
         adata.layers in order to use these models.


        Parameters
        ----------
        adata : sc.AnnData
            The scanpy AnnData object. adata should already contain adata.var.highly_variable
        covariates : list, optional
            A list of names of covariate vectors that are stored in adata.obs
        pi_covariates: list, optional
            A list of names of covariate vectors used as input for pilayer
        model_type : str, optional
            'UMI', 'non-UMI' and 'Gaussian', default is 'Gaussian'.
        npc : int, optional
            The number of PCs to use when model_type is 'Gaussian'. The default is 64.
        adata_layer_counts: str, optional
            the key name of adata.layers that stores the count data if model_type is
            'UMI' or 'non-UMI'
        copy_adata: bool, optional. Set to True if we don't want VITAE to modify the original adata. If set to True, self.adata will be an independent copy of the original adata. 
        hidden_layers : list, optional
            The list of dimensions of layers of autoencoder between latent space and original space. Default is to have only one hidden layer with 32 nodes
        latent_space_dim : int, optional
            The dimension of latent space.
        gamme : float, optional
            The weight of the MMD loss
        conditions : str or list, optional
            The conditions of different cells


        Returns
        -------
        None.

        '''
        self.dict_method_scname = {
            'PCA' : 'X_pca',
            'UMAP' : 'X_umap',
            'TSNE' : 'X_tsne',
            'diffmap' : 'X_diffmap',
            'draw_graph' : 'X_draw_graph_fa'
        }

        if model_type != 'Gaussian':
            if adata_layer_counts is None:
                raise ValueError("need to provide the name in adata.layers that stores the raw count data")
            if 'highly_variable' not in adata.var:
                raise ValueError("need to first select highly variable genes using scanpy")

        self.model_type = model_type

        if copy_adata:
            self.adata = adata.copy()
        else:
            self.adata = adata

        if covariates is not None:
            if isinstance(covariates, str):
                covariates = [covariates]
            covariates = np.array(covariates)
            id_cat = (adata.obs[covariates].dtypes == 'category')
            # add OneHotEncoder & StandardScaler as class variable if needed
            if np.sum(id_cat)>0:
                covariates_cat = OneHotEncoder(drop='if_binary', handle_unknown='ignore'
                    ).fit_transform(adata.obs[covariates[id_cat]]).toarray()
            else:
                covariates_cat = np.array([]).reshape(adata.shape[0],0)

            # temporarily disable StandardScaler
            if np.sum(~id_cat)>0:
                #covariates_con = StandardScaler().fit_transform(adata.obs[covariates[~id_cat]])
                covariates_con = adata.obs[covariates[~id_cat]]
            else:
                covariates_con = np.array([]).reshape(adata.shape[0],0)

            self.covariates = np.c_[covariates_cat, covariates_con].astype(tf.keras.backend.floatx())
        else:
            self.covariates = None

        if conditions is not None:
            ## observations with np.nan will not participant in calculating mmd_loss
            if isinstance(conditions, str):
                conditions = [conditions]
            conditions = np.array(conditions)
            if np.any(adata.obs[conditions].dtypes != 'category'):
                raise ValueError("Conditions should all be categorical.")

            self.conditions = OrdinalEncoder(dtype=int, encoded_missing_value=-1).fit_transform(adata.obs[conditions]) + int(1)
        else:
            self.conditions = None

        if pi_covariates is not None:
            self.pi_cov = adata.obs[pi_covariates].to_numpy()
            if self.pi_cov.ndim == 1:
                self.pi_cov = self.pi_cov.reshape(-1, 1)
                self.pi_cov = self.pi_cov.astype(tf.keras.backend.floatx())
        else:
            self.pi_cov = np.zeros((adata.shape[0],1), dtype=tf.keras.backend.floatx())
            
        self.model_type = model_type
        self._adata = sc.AnnData(X = self.adata.X, var = self.adata.var)
        self._adata.obs = self.adata.obs
        self._adata.uns = self.adata.uns


        if model_type == 'Gaussian':
            sc.tl.pca(adata, n_comps = npc)
            self.X_input = self.X_output = adata.obsm['X_pca']
            self.scale_factor = np.ones(self.X_output.shape[0])
        else:
            print(f"{adata.var.highly_variable.sum()} highly variable genes selected as input") 
            self.X_input = adata.X[:, adata.var.highly_variable]
            self.X_output = adata.layers[adata_layer_counts][ :, adata.var.highly_variable]
            self.scale_factor = np.sum(self.X_output, axis=1, keepdims=True)/1e4

        self.dimensions = hidden_layers
        self.dim_latent = latent_space_dim

        self.vae = model.VariationalAutoEncoder(
            self.X_output.shape[1], self.dimensions,
            self.dim_latent, self.model_type,
            False if self.covariates is None else True,
            )

        if hasattr(self, 'inferer'):
            delattr(self, 'inferer')
        

    def pre_train(self, test_size = 0.1, random_state: int = 0,
            learning_rate: float = 1e-3, batch_size: int = 256, L: int = 1, alpha: float = 0.10, gamma: float = 0,
            phi : float = 1,num_epoch: int = 200, num_step_per_epoch: Optional[int] = None,
            early_stopping_patience: int = 10, early_stopping_tolerance: float = 0.01, 
            early_stopping_relative: bool = True, verbose: bool = False,path_to_weights: Optional[str] = None):
        '''Pretrain the model with specified learning rate.

        Parameters
        ----------
        test_size : float or int, optional
            The proportion or size of the test set.
        random_state : int, optional
            The random state for data splitting.
        learning_rate : float, optional
            The initial learning rate for the Adam optimizer.
        batch_size : int, optional 
            The batch size for pre-training.  Default is 256. Set to 32 if number of cells is small (less than 1000)
        L : int, optional 
            The number of MC samples.
        alpha : float, optional
            The value of alpha in [0,1] to encourage covariate adjustment. Not used if there is no covariates.
        gamma : float, optional
            The weight of the mmd loss if used.
        phi : float, optional
            The weight of Jocob norm of the encoder.
        num_epoch : int, optional 
            The maximum number of epochs.
        num_step_per_epoch : int, optional 
            The number of step per epoch, it will be inferred from number of cells and batch size if it is None.            
        early_stopping_patience : int, optional 
            The maximum number of epochs if there is no improvement.
        early_stopping_tolerance : float, optional 
            The minimum change of loss to be considered as an improvement.
        early_stopping_relative : bool, optional
            Whether monitor the relative change of loss as stopping criteria or not.
        path_to_weights : str, optional 
            The path of weight file to be saved; not saving weight if None.
        conditions : str or list, optional
            The conditions of different cells
        '''

        id_train, id_test = train_test_split(
                                np.arange(self.X_input.shape[0]), 
                                test_size=test_size, 
                                random_state=random_state)
        if num_step_per_epoch is None:
            num_step_per_epoch = len(id_train)//batch_size+1
        self.train_dataset = train.warp_dataset(self.X_input[id_train].astype(tf.keras.backend.floatx()), 
                                                None if self.covariates is None else self.covariates[id_train].astype(tf.keras.backend.floatx()),
                                                batch_size, 
                                                self.X_output[id_train].astype(tf.keras.backend.floatx()), 
                                                self.scale_factor[id_train].astype(tf.keras.backend.floatx()),
                                                conditions = None if self.conditions is None else self.conditions[id_train].astype(tf.keras.backend.floatx()))
        self.test_dataset = train.warp_dataset(self.X_input[id_test], 
                                                None if self.covariates is None else self.covariates[id_test].astype(tf.keras.backend.floatx()),
                                                batch_size, 
                                                self.X_output[id_test].astype(tf.keras.backend.floatx()), 
                                                self.scale_factor[id_test].astype(tf.keras.backend.floatx()),
                                                conditions = None if self.conditions is None else self.conditions[id_test].astype(tf.keras.backend.floatx()))

        self.vae = train.pre_train(
            self.train_dataset,
            self.test_dataset,
            self.vae,
            learning_rate,                        
            L, alpha, gamma, phi,
            num_epoch,
            num_step_per_epoch,
            early_stopping_patience,
            early_stopping_tolerance,
            early_stopping_relative,
            verbose)
        
        self.update_z()

        if path_to_weights is not None:
            self.save_model(path_to_weights)
            

    def update_z(self):
        self.z = self.get_latent_z()        
        self._adata_z = sc.AnnData(self.z)
        sc.pp.neighbors(self._adata_z)

            
    def get_latent_z(self):
        ''' get the posterier mean of current latent space z (encoder output)

        Returns
        ----------
        z : np.array
            \([N,d]\) The latent means.
        ''' 
        c = None if self.covariates is None else self.covariates
        return self.vae.get_z(self.X_input, c)
            
    
    def visualize_latent(self, method: str = "UMAP", 
                         color = None, **kwargs):
        '''
        visualize the current latent space z using the scanpy visualization tools

        Parameters
        ----------
        method : str, optional
            Visualization method to use. The default is "draw_graph" (the FA plot). Possible choices include "PCA", "UMAP", 
            "diffmap", "TSNE" and "draw_graph"
        color : TYPE, optional
            Keys for annotations of observations/cells or variables/genes, e.g., 'ann1' or ['ann1', 'ann2'].
            The default is None. Same as scanpy.
        **kwargs :  
            Extra key-value arguments that can be passed to scanpy plotting functions (scanpy.pl.XX).   

        Returns
        -------
        None.

        '''
          
        if method not in ['PCA', 'UMAP', 'TSNE', 'diffmap', 'draw_graph']:
            raise ValueError("visualization method should be one of 'PCA', 'UMAP', 'TSNE', 'diffmap' and 'draw_graph'")
        
        temp = list(self._adata_z.obsm.keys())
        if method == 'PCA' and not 'X_pca' in temp:
            print("Calculate PCs ...")
            sc.tl.pca(self._adata_z)
        elif method == 'UMAP' and not 'X_umap' in temp:  
            print("Calculate UMAP ...")
            sc.tl.umap(self._adata_z)
        elif method == 'TSNE' and not 'X_tsne' in temp:
            print("Calculate TSNE ...")
            sc.tl.tsne(self._adata_z)
        elif method == 'diffmap' and not 'X_diffmap' in temp:
            print("Calculate diffusion map ...")
            sc.tl.diffmap(self._adata_z)
        elif method == 'draw_graph' and not 'X_draw_graph_fa' in temp:
            print("Calculate FA ...")
            sc.tl.draw_graph(self._adata_z)
            
  
        self._adata.obsp = self._adata_z.obsp
#        self._adata.uns = self._adata_z.uns
        self._adata.obsm = self._adata_z.obsm
    
        if method == 'PCA':
            axes = sc.pl.pca(self._adata, color = color, **kwargs)
        elif method == 'UMAP':            
            axes = sc.pl.umap(self._adata, color = color, **kwargs)
        elif method == 'TSNE':
            axes = sc.pl.tsne(self._adata, color = color, **kwargs)
        elif method == 'diffmap':
            axes = sc.pl.diffmap(self._adata, color = color, **kwargs)
        elif method == 'draw_graph':
            axes = sc.pl.draw_graph(self._adata, color = color, **kwargs)
        return axes


    def init_latent_space(self, cluster_label = None, log_pi = None, res: float = 1.0, 
                          ratio_prune= None, dist_thres = 0.5, pilayer = False):
        '''Initialize the latent space.

        Parameters
        ----------
        cluster_label : str, optional
            the name of vector of labels that can be found in self.adata.obs. 
            Default is None, which will perform leiden clustering on the pretrained z to get clusters
        mu : np.array, optional
            \([d,k]\) The value of initial \(\\mu\).
        log_pi : np.array, optional
            \([1,K]\) The value of initial \(\\log(\\pi)\).
        res: 
            The resolution of leiden clustering, which is a parameter value controlling the coarseness of the clustering. 
            Higher values lead to more clusters. Deafult is 1.
        ratio_prune : float, optional
            The ratio of edges to be removed before estimating.
        '''   
    
        
        if cluster_label is None:
            print("Perform leiden clustering on the latent space z ...")
            g = get_igraph(self.z)
            cluster_labels = leidenalg_igraph(g, res = res)
            cluster_labels = cluster_labels.astype(str) 
            uni_cluster_labels = np.unique(cluster_labels)
        else:
            if isinstance(cluster_label,str):
                cluster_labels = self.adata.obs[cluster_label].to_numpy()
                uni_cluster_labels = np.array(self.adata.obs[cluster_label].cat.categories)
            else:
                ## if cluster_label is a list
                cluster_labels = cluster_label
                uni_cluster_labels = np.unique(cluster_labels)

        n_clusters = len(uni_cluster_labels)

        if not hasattr(self, 'z'):
            self.update_z()        
        z = self.z
        mu = np.zeros((z.shape[1], n_clusters))
        for i,l in enumerate(uni_cluster_labels):
            mu[:,i] = np.mean(z[cluster_labels==l], axis=0)
   #         mu[:,i] = z[cluster_labels==l][np.argmin(np.mean((z[cluster_labels==l] - mu[:,i])**2, axis=1)),:]
       
        ### update cluster centers if some cluster centers are too close
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=dist_thres,
            linkage='complete'
            ).fit(mu.T/np.sqrt(mu.shape[0]))
        n_clusters_new = clustering.n_clusters_
        if n_clusters_new < n_clusters:
            print("Merge clusters for cluster centers that are too close ...")
            n_clusters = n_clusters_new
            for i in range(n_clusters):    
                temp = uni_cluster_labels[clustering.labels_ == i]
                idx = np.isin(cluster_labels, temp)
                cluster_labels[idx] = ','.join(temp)
                if np.sum(clustering.labels_==i)>1:
                    print('Merge %s'% ','.join(temp))
            uni_cluster_labels = np.unique(cluster_labels)
            mu = np.zeros((z.shape[1], n_clusters))
            for i,l in enumerate(uni_cluster_labels):
                mu[:,i] = np.mean(z[cluster_labels==l], axis=0)
            
        self.adata.obs['vitae_init_clustering'] = cluster_labels
        self.adata.obs['vitae_init_clustering'] = self.adata.obs['vitae_init_clustering'].astype('category')
        print("Initial clustering labels saved as 'vitae_init_clustering' in self.adata.obs.")

   
        if (log_pi is None) and (cluster_labels is not None) and (n_clusters>3):                         
            n_states = int((n_clusters+1)*n_clusters/2)
            d = _comp_dist(z, cluster_labels, mu.T)

            C = np.triu(np.ones(n_clusters))
            C[C>0] = np.arange(n_states)
            C = C.astype(int)

            log_pi = np.zeros((1,n_states))
            ## pruning to throw away edges for far-away clusters if there are too many clusters
            if ratio_prune is not None:
                log_pi[0, C[np.triu(d)>np.quantile(d[np.triu_indices(n_clusters, 1)], 1-ratio_prune)]] = - np.inf
            else:
                log_pi[0, C[np.triu(d)> np.quantile(d[np.triu_indices(n_clusters, 1)], 5/n_clusters) * 3]] = - np.inf

        self.n_states = n_clusters
        self.labels = cluster_labels

        # Not sure if storing the this will be useful
        # self.init_labels_name = cluster_label
        
        labels_map = pd.DataFrame.from_dict(
            {i:label for i,label in enumerate(uni_cluster_labels)}, 
            orient='index', columns=['label_names'], dtype=str
            )
        
        self.labels_map = labels_map
        self.vae.init_latent_space(self.n_states, mu, log_pi)
        self.inferer = Inferer(self.n_states)
        self.mu = self.vae.latent_space.mu.numpy()
        self.pi = np.triu(np.ones(self.n_states))
        self.pi[self.pi > 0] = tf.nn.softmax(self.vae.latent_space.pi).numpy()[0]

        if pilayer:
            self.vae.create_pilayer()

    def update_latent_space(self, dist_thres: float=0.5):
        pi = self.pi[np.triu_indices(self.n_states)]
        mu = self.mu    
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=dist_thres,
            linkage='complete'
            ).fit(mu.T/np.sqrt(mu.shape[0]))
        n_clusters = clustering.n_clusters_   

        if n_clusters<self.n_states:      
            print("Merge clusters for cluster centers that are too close ...")
            mu_new = np.empty((self.dim_latent, n_clusters))
            C = np.zeros((self.n_states, self.n_states))
            C[np.triu_indices(self.n_states, 0)] = pi
            C = np.triu(C, 1) + C.T
            C_new = np.zeros((n_clusters, n_clusters))
            
            uni_cluster_labels = self.labels_map['label_names'].to_numpy()
            returned_order = {}
            cluster_labels = self.labels
            for i in range(n_clusters):
                temp = uni_cluster_labels[clustering.labels_ == i]
                idx = np.isin(cluster_labels, temp)
                cluster_labels[idx] = ','.join(temp)
                returned_order[i] = ','.join(temp)
                if np.sum(clustering.labels_==i)>1:
                    print('Merge %s'% ','.join(temp))
            uni_cluster_labels = np.unique(cluster_labels) 
            for i,l in enumerate(uni_cluster_labels):  ## reorder the merged clusters based on the cluster names
                k = np.where(returned_order == l)
                mu_new[:, i] = np.mean(mu[:,clustering.labels_==k], axis=-1)
                # sum of the aggregated pi's
                C_new[i, i] = np.sum(np.triu(C[clustering.labels_==k,:][:,clustering.labels_==k]))
                for j in range(i+1, n_clusters):
                    k1 = np.where(returned_order == uni_cluster_labels[j])
                    C_new[i, j] = np.sum(C[clustering.labels_== k, :][:, clustering.labels_==k1])

#            labels_map_new = {}
#            for i in range(n_clusters):                       
#                # update label map: int->str
#                labels_map_new[i] = self.labels_map.loc[clustering.labels_==i, 'label_names'].str.cat(sep=',')
#                if np.sum(clustering.labels_==i)>1:
#                    print('Merge %s'%labels_map_new[i])
#                # mean of the aggregated cluster means
#                mu_new[:, i] = np.mean(mu[:,clustering.labels_==i], axis=-1)
#                # sum of the aggregated pi's
#                C_new[i, i] = np.sum(np.triu(C[clustering.labels_==i,:][:,clustering.labels_==i]))
#                for j in range(i+1, n_clusters):
#                    C_new[i, j] = np.sum(C[clustering.labels_== i, :][:, clustering.labels_==j])
            C_new = np.triu(C_new,1) + C_new.T

            pi_new = C_new[np.triu_indices(n_clusters)]
            log_pi_new = np.log(pi_new, out=np.ones_like(pi_new)*(-np.inf), where=(pi_new!=0)).reshape((1,-1))
            self.n_states = n_clusters
            self.labels_map = pd.DataFrame.from_dict(
                {i:label for i,label in enumerate(uni_cluster_labels)},
                orient='index', columns=['label_names'], dtype=str
                )
            self.labels = cluster_labels
#            self.labels_map = pd.DataFrame.from_dict(
#                labels_map_new, orient='index', columns=['label_names'], dtype=str
#            )
            self.vae.init_latent_space(self.n_states, mu_new, log_pi_new)
            self.inferer = Inferer(self.n_states)
            self.mu = self.vae.latent_space.mu.numpy()
            self.pi = np.triu(np.ones(self.n_states))
            self.pi[self.pi > 0] = tf.nn.softmax(self.vae.latent_space.pi).numpy()[0]



    def train(self, stratify = False, test_size = 0.1, random_state: int = 0,
            learning_rate: float = 1e-3, batch_size: int = 256,
            L: int = 1, alpha: float = 0.10, beta: float = 1, gamma: float = 0, phi: float = 1,
            num_epoch: int = 200, num_step_per_epoch: Optional[int] =  None,
            early_stopping_patience: int = 10, early_stopping_tolerance: float = 0.01, 
            early_stopping_relative: bool = True, early_stopping_warmup: int = 0,
            path_to_weights: Optional[str] = None,
            verbose: bool = False, **kwargs):
        '''Train the model.

        Parameters
        ----------
        stratify : np.array, None, or False
            If an array is provided, or `stratify=None` and `self.labels` is available, then they will be used to perform stratified shuffle splitting. Otherwise, general shuffle splitting is used. Set to `False` if `self.labels` is not intended for stratified shuffle splitting.
        test_size : float or int, optional
            The proportion or size of the test set.
        random_state : int, optional
            The random state for data splitting.
        learning_rate : float, optional  
            The initial learning rate for the Adam optimizer.
        batch_size : int, optional  
            The batch size for training. Default is 256. Set to 32 if number of cells is small (less than 1000)
        L : int, optional  
            The number of MC samples.
        alpha : float, optional  
            The value of alpha in [0,1] to encourage covariate adjustment. Not used if there is no covariates.
        beta : float, optional  
            The value of beta in beta-VAE.
        gamma : float, optional
            The weight of mmd_loss.
        phi : float, optional
            The weight of Jacob norm of encoder.
        num_epoch : int, optional  
            The number of epoch.
        num_step_per_epoch : int, optional 
            The number of step per epoch, it will be inferred from number of cells and batch size if it is None.
        early_stopping_patience : int, optional 
            The maximum number of epochs if there is no improvement.
        early_stopping_tolerance : float, optional 
            The minimum change of loss to be considered as an improvement.
        early_stopping_relative : bool, optional
            Whether monitor the relative change of loss or not.            
        early_stopping_warmup : int, optional 
            The number of warmup epochs.            
        path_to_weights : str, optional 
            The path of weight file to be saved; not saving weight if None.
        **kwargs :  
            Extra key-value arguments for dimension reduction algorithms.        
        '''
        if gamma == 0 or self.conditions is None:
            conditions = np.array([np.nan] * self.adata.shape[0])
        else:
            conditions = self.conditions

        if stratify is None:
            stratify = self.labels
        elif stratify is False:
            stratify = None    
        id_train, id_test = train_test_split(
                                np.arange(self.X_input.shape[0]), 
                                test_size=test_size, 
                                stratify=stratify, 
                                random_state=random_state)
        if num_step_per_epoch is None:
            num_step_per_epoch = len(id_train)//batch_size+1
        c = None if self.covariates is None else self.covariates.astype(tf.keras.backend.floatx())
        self.train_dataset = train.warp_dataset(self.X_input[id_train].astype(tf.keras.backend.floatx()),
                                                None if c is None else c[id_train],
                                                batch_size, 
                                                self.X_output[id_train].astype(tf.keras.backend.floatx()), 
                                                self.scale_factor[id_train].astype(tf.keras.backend.floatx()),
                                                conditions = conditions[id_train],
                                                pi_cov = self.pi_cov[id_train])
        self.test_dataset = train.warp_dataset(self.X_input[id_test].astype(tf.keras.backend.floatx()),
                                                None if c is None else c[id_test],
                                                batch_size, 
                                                self.X_output[id_test].astype(tf.keras.backend.floatx()), 
                                                self.scale_factor[id_test].astype(tf.keras.backend.floatx()),
                                                conditions = conditions[id_test],
                                                pi_cov = self.pi_cov[id_test])
                                   
        self.vae = train.train(
            self.train_dataset,
            self.test_dataset,
            self.vae,
            learning_rate,
            L,
            alpha,
            beta,
            gamma,
            phi,
            num_epoch,
            num_step_per_epoch,
            early_stopping_patience,
            early_stopping_tolerance,
            early_stopping_relative,
            early_stopping_warmup,  
            verbose,
            **kwargs            
            )
        
        self.update_z()
        self.mu = self.vae.latent_space.mu.numpy()
        self.pi = np.triu(np.ones(self.n_states))
        self.pi[self.pi > 0] = tf.nn.softmax(self.vae.latent_space.pi).numpy()[0]
            
        if path_to_weights is not None:
            self.save_model(path_to_weights)
    

    def output_pi(self, pi_cov):
        """return a matrix n_states by n_states and a mask for plotting, which can be used to cover the lower triangular(except the diagnoals) of a heatmap"""
        p = self.vae.pilayer
        pi_cov = tf.expand_dims(tf.constant([pi_cov], dtype=tf.float32), 0)
        pi_val = tf.nn.softmax(p(pi_cov)).numpy()[0]
        # Create heatmap matrix
        n = self.vae.n_states
        matrix = np.zeros((n, n))
        matrix[np.triu_indices(n)] = pi_val
        mask = np.tril(np.ones_like(matrix), k=-1)
        return matrix, mask


    def return_pilayer_weights(self):
        """return parameters of pilayer, which has dimension dim(pi_cov) + 1 by n_categories, the last row is biases"""
        return np.vstack((model.vae.pilayer.weights[0].numpy(), model.vae.pilayer.weights[1].numpy().reshape(1, -1)))


    def posterior_estimation(self, batch_size: int = 32, L: int = 50, **kwargs):
        '''Initialize trajectory inference by computing the posterior estimations.        

        Parameters
        ----------
        batch_size : int, optional
            The batch size when doing inference.
        L : int, optional
            The number of MC samples when doing inference.
        **kwargs :  
            Extra key-value arguments for dimension reduction algorithms.              
        '''
        c = None if self.covariates is None else self.covariates.astype(tf.keras.backend.floatx())
        self.test_dataset = train.warp_dataset(self.X_input.astype(tf.keras.backend.floatx()), 
                                               c,
                                               batch_size)
        _, _, self.pc_x,\
            self.cell_position_posterior,self.cell_position_variance,_ = self.vae.inference(self.test_dataset, L=L)
            
        uni_cluster_labels = self.labels_map['label_names'].to_numpy()
        self.adata.obs['vitae_new_clustering'] = uni_cluster_labels[np.argmax(self.cell_position_posterior, 1)]
        self.adata.obs['vitae_new_clustering'] = self.adata.obs['vitae_new_clustering'].astype('category')
        print("New clustering labels saved as 'vitae_new_clustering' in self.adata.obs.")
        return None


    def infer_backbone(self, method: str = 'modified_map', thres = 0.5,
            no_loop: bool = True, cutoff: float = 0,
            visualize: bool = True, color = 'vitae_new_clustering',path_to_fig = None,**kwargs):
        ''' Compute edge scores.

        Parameters
        ----------
        method : string, optional
            'mean', 'modified_mean', 'map', or 'modified_map'.
        thres : float, optional
            The threshold used for filtering edges \(e_{ij}\) that \((n_{i}+n_{j}+e_{ij})/N<thres\), only applied to mean method.
        no_loop : boolean, optional
            Whether loops are allowed to exist in the graph. If no_loop is true, will prune the graph to contain only the
            maximum spanning true
        cutoff : string, optional
            The score threshold for filtering edges with scores less than cutoff.
        visualize: boolean
            whether plot the current trajectory backbone (undirected graph)

        Returns
        ----------
        G : nx.Graph
            The weighted graph with weight on each edge indicating its score of existence.
        '''
        # build_graph, return graph
        self.backbone = self.inferer.build_graphs(self.cell_position_posterior, self.pc_x,
                method, thres, no_loop, cutoff)
        self.cell_position_projected = self.inferer.modify_wtilde(self.cell_position_posterior, 
                np.array(list(self.backbone.edges)))
        
        uni_cluster_labels = self.labels_map['label_names'].to_numpy()
        temp_dict = {i:label for i,label in enumerate(uni_cluster_labels)}
        nx.relabel_nodes(self.backbone, temp_dict)
       
        self.adata.obs['vitae_new_clustering'] = uni_cluster_labels[np.argmax(self.cell_position_projected, 1)]
        self.adata.obs['vitae_new_clustering'] = self.adata.obs['vitae_new_clustering'].astype('category')
        print("'vitae_new_clustering' updated based on the projected cell positions.")

        self.uncertainty = np.sum((self.cell_position_projected - self.cell_position_posterior)**2, axis=-1) \
            + np.sum(self.cell_position_variance, axis=-1)
        self.adata.obs['projection_uncertainty'] = self.uncertainty
        print("Cell projection uncertainties stored as 'projection_uncertainty' in self.adata.obs")
        if visualize:
            self._adata.obs = self.adata.obs.copy()
            self.ax = self.plot_backbone(directed = False,color = color, **kwargs)
            if path_to_fig is not None:
                self.ax.figure.savefig(path_to_fig)
            self.ax.figure.show()
        return None


    def select_root(self, days, method: str = 'proportion'):
        '''Order the vertices/states based on cells' collection time information to select the root state.      

        Parameters
        ----------
        day : np.array 
            The day information for selected cells used to determine the root vertex.
            The dtype should be 'int' or 'float'.
        method : str, optional
            'sum' or 'mean'. 
            For 'proportion', the root is the one with maximal proportion of cells from the earliest day.
            For 'mean', the root is the one with earliest mean time among cells associated with it.

        Returns
        ----------
        root : int 
            The root vertex in the inferred trajectory based on given day information.
        '''
        ## TODO: change return description
        if days is not None and len(days)!=self.X_input.shape[0]:
            raise ValueError("The length of day information ({}) is not "
                "consistent with the number of selected cells ({})!".format(
                    len(days), self.X_input.shape[0]))
        if not hasattr(self, 'cell_position_projected'):
            raise ValueError("Need to call 'infer_backbone' first!")

        collection_time = np.dot(days, self.cell_position_projected)/np.sum(self.cell_position_projected, axis = 0)
        earliest_prop = np.dot(days==np.min(days), self.cell_position_projected)/np.sum(self.cell_position_projected, axis = 0)
        
        root_info = self.labels_map.copy()
        root_info['mean_collection_time'] = collection_time
        root_info['earliest_time_prop'] = earliest_prop
        root_info.sort_values('mean_collection_time', inplace=True)
        return root_info


    def plot_backbone(self, directed: bool = False, 
                      method: str = 'UMAP', color = 'vitae_new_clustering', **kwargs):
        '''Plot the current trajectory backbone (undirected graph).

        Parameters
        ----------
        directed : boolean, optional
            Whether the backbone is directed or not.
        method : str, optional
            The dimension reduction method to use. The default is "UMAP".
        color : str, optional
            The key for annotations of observations/cells or variables/genes, e.g., 'ann1' or ['ann1', 'ann2'].
            The default is 'vitae_new_clustering'.
        **kwargs :
            Extra key-value arguments that can be passed to scanpy plotting functions (scanpy.pl.XX).
        '''
        if not isinstance(color,str):
            raise ValueError('The color argument should be of type str!')
        ax = self.visualize_latent(method = method, color=color, show=False, **kwargs)
        dict_label_num = {j:i for i,j in self.labels_map['label_names'].to_dict().items()}
        uni_cluster_labels = self.adata.obs['vitae_init_clustering'].cat.categories
        cluster_labels = self.adata.obs['vitae_new_clustering'].to_numpy()
        embed_z = self._adata.obsm[self.dict_method_scname[method]]
        embed_mu = np.zeros((len(uni_cluster_labels), 2))
        for l in uni_cluster_labels:
            embed_mu[dict_label_num[l],:] = np.mean(embed_z[cluster_labels==l], axis=0)

        if directed:
            graph = self.directed_backbone
        else:
            graph = self.backbone
        edges = list(graph.edges)
        edge_scores = np.array([d['weight'] for (u,v,d) in graph.edges(data=True)])
        if max(edge_scores) - min(edge_scores) == 0:
            edge_scores = edge_scores/max(edge_scores)
        else:
            edge_scores = (edge_scores - min(edge_scores))/(max(edge_scores) - min(edge_scores))*3

        value_range = np.maximum(np.diff(ax.get_xlim())[0], np.diff(ax.get_ylim())[0])
        y_range = np.min(embed_z[:,1]), np.max(embed_z[:,1], axis=0)
        for i in range(len(edges)):
            points = embed_z[np.sum(self.cell_position_projected[:, edges[i]]>0, axis=-1)==2,:]
            points = points[points[:,0].argsort()]
            try:
                x_smooth, y_smooth = _get_smooth_curve(
                    points,
                    embed_mu[edges[i], :],
                    y_range
                    )
            except:
                x_smooth, y_smooth = embed_mu[edges[i], 0], embed_mu[edges[i], 1]
            ax.plot(x_smooth, y_smooth,
                '-',
                linewidth= 1 + edge_scores[i],
                color="black",
                alpha=0.8,
                path_effects=[pe.Stroke(linewidth=1+edge_scores[i]+1.5,
                                        foreground='white'), pe.Normal()],
                zorder=1
                )

            if directed:
                delta_x = embed_mu[edges[i][1], 0] - x_smooth[-2]
                delta_y = embed_mu[edges[i][1], 1] - y_smooth[-2]
                length = np.sqrt(delta_x**2 + delta_y**2) / 50 * value_range
                ax.arrow(
                        embed_mu[edges[i][1], 0]-delta_x/length,
                        embed_mu[edges[i][1], 1]-delta_y/length,
                        delta_x/length,
                        delta_y/length,
                        color='black', alpha=1.0,
                        shape='full', lw=0, length_includes_head=True,
                        head_width=np.maximum(0.01*(1 + edge_scores[i]), 0.03) * value_range,
                        zorder=2) 
        
        colors = self._adata.uns['vitae_new_clustering_colors']
            
        for i,l in enumerate(uni_cluster_labels):
            ax.scatter(*embed_mu[dict_label_num[l]:dict_label_num[l]+1,:].T, 
                       c=[colors[i]], edgecolors='white', # linewidths=10,  norm=norm,
                       s=250, marker='*', label=l)

        plt.setp(ax, xticks=[], yticks=[])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                            box.width, box.height * 0.9])
        if directed:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                fancybox=True, shadow=True, ncol=5)

        return ax


    def plot_center(self, color = "vitae_new_clustering", plot_legend = True, legend_add_index = True,
                    method: str = 'UMAP',ncol = 2,font_size = "medium",
                    add_egde = False, add_direct = False,**kwargs):
        '''Plot the center of each cluster in the latent space.

        Parameters
        ----------
        color : str, optional
            The color of the center of each cluster. Default is "vitae_new_clustering".
        plot_legend : bool, optional
            Whether to plot the legend. Default is True.
        legend_add_index : bool, optional
            Whether to add the index of each cluster in the legend. Default is True.
        method : str, optional
            The dimension reduction method used for visualization. Default is 'UMAP'.
        ncol : int, optional
            The number of columns in the legend. Default is 2.
        font_size : str, optional
            The font size of the legend. Default is "medium".
        add_egde : bool, optional
            Whether to add the edges between the centers of clusters. Default is False.
        add_direct : bool, optional
            Whether to add the direction of the edges. Default is False.
        '''
        if color not in ["vitae_new_clustering","vitae_init_clustering"]:
            raise ValueError("Can only plot center of vitae_new_clustering or vitae_init_clustering")
        dict_label_num = {j: i for i, j in self.labels_map['label_names'].to_dict().items()}
        if legend_add_index:
            self._adata.obs["index_"+color] = self._adata.obs[color].map(lambda x: dict_label_num[x])
            ax = self.visualize_latent(method=method, color="index_" + color, show=False, legend_loc="on data",
                                        legend_fontsize=font_size,**kwargs)
            colors = self._adata.uns["index_" + color + '_colors']
        else:
            ax = self.visualize_latent(method=method, color = color, show=False,**kwargs)
            colors = self._adata.uns[color + '_colors']
        uni_cluster_labels = self.adata.obs[color].cat.categories
        cluster_labels = self.adata.obs[color].to_numpy()
        embed_z = self._adata.obsm[self.dict_method_scname[method]]
        embed_mu = np.zeros((len(uni_cluster_labels), 2))
        for l in uni_cluster_labels:
            embed_mu[dict_label_num[l], :] = np.mean(embed_z[cluster_labels == l], axis=0)

        leg = (self.labels_map.index.astype(str) + " : " + self.labels_map.label_names).values
        for i, l in enumerate(uni_cluster_labels):
            ax.scatter(*embed_mu[dict_label_num[l]:dict_label_num[l] + 1, :].T,
                       c=[colors[i]], edgecolors='white', # linewidths=3,
                       s=250, marker='*', label=leg[i])
        if plot_legend:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=ncol, markerscale=0.8, frameon=False)
        plt.setp(ax, xticks=[], yticks=[])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        if add_egde:
            if add_direct:
                graph = self.directed_backbone
            else:
                graph = self.backbone
            edges = list(graph.edges)
            edge_scores = np.array([d['weight'] for (u, v, d) in graph.edges(data=True)])
            if max(edge_scores) - min(edge_scores) == 0:
                edge_scores = edge_scores / max(edge_scores)
            else:
                edge_scores = (edge_scores - min(edge_scores)) / (max(edge_scores) - min(edge_scores)) * 3

            value_range = np.maximum(np.diff(ax.get_xlim())[0], np.diff(ax.get_ylim())[0])
            y_range = np.min(embed_z[:, 1]), np.max(embed_z[:, 1], axis=0)
            for i in range(len(edges)):
                points = embed_z[np.sum(self.cell_position_projected[:, edges[i]] > 0, axis=-1) == 2, :]
                points = points[points[:, 0].argsort()]
                try:
                    x_smooth, y_smooth = _get_smooth_curve(
                        points,
                        embed_mu[edges[i], :],
                        y_range
                    )
                except:
                    x_smooth, y_smooth = embed_mu[edges[i], 0], embed_mu[edges[i], 1]
                ax.plot(x_smooth, y_smooth,
                        '-',
                        linewidth=1 + edge_scores[i],
                        color="black",
                        alpha=0.8,
                        path_effects=[pe.Stroke(linewidth=1 + edge_scores[i] + 1.5,
                                                foreground='white'), pe.Normal()],
                        zorder=1
                        )

                if add_direct:
                    delta_x = embed_mu[edges[i][1], 0] - x_smooth[-2]
                    delta_y = embed_mu[edges[i][1], 1] - y_smooth[-2]
                    length = np.sqrt(delta_x ** 2 + delta_y ** 2) / 50 * value_range
                    ax.arrow(
                        embed_mu[edges[i][1], 0] - delta_x / length,
                        embed_mu[edges[i][1], 1] - delta_y / length,
                        delta_x / length,
                        delta_y / length,
                        color='black', alpha=1.0,
                        shape='full', lw=0, length_includes_head=True,
                        head_width=np.maximum(0.01 * (1 + edge_scores[i]), 0.03) * value_range,
                        zorder=2)
        self.ax = ax
        self.ax.figure.show()
        return None


    def infer_trajectory(self, root: Union[int,str], color = "pseudotime",
                         visualize: bool = True, path_to_fig = None,  **kwargs):
        '''Infer the trajectory.

        Parameters
        ----------
        root : int or string
            The root of the inferred trajectory. Can provide either an int (vertex index) or string (label name)
        cutoff : string, optional
            The threshold for filtering edges with scores less than cutoff.
        visualize: boolean
            Whether plot the current trajectory backbone (directed graph)
        path_to_fig : string, optional  
            The path to save figure, or don't save if it is None.
        **kwargs : dict, optional
            Other keywords arguments for plotting.
        '''
        if isinstance(root,str):
            if root not in self.labels_map.values:
                raise ValueError("Root {} is not in the label names!".format(root))
            root = self.labels_map[self.labels_map['label_names']==root].index[0]

        connected_comps = nx.node_connected_component(self.backbone, root)
        subG = self.backbone.subgraph(connected_comps)
        
        ## generate directed backbone which contains no loops
        DG = nx.DiGraph(nx.to_directed(self.backbone))
        temp = DG.subgraph(connected_comps)
        DG.remove_edges_from(temp.edges - nx.dfs_edges(DG, root))
        self.directed_backbone = DG


        if len(subG.edges)>0:
            milestone_net = self.inferer.build_milestone_net(subG, root)
            if self.inferer.no_loop is False and milestone_net.shape[0]<len(self.backbone.edges):
                warnings.warn("The directed graph shown is a minimum spanning tree of the estimated trajectory backbone to avoid arbitrary assignment of the directions.")
            self.pseudotime = self.inferer.comp_pseudotime(milestone_net, root, self.cell_position_projected)
        else:
            warnings.warn("There are no connected states for starting from the giving root.")
            self.pseudotime = -np.ones(self._adata.shape[0])

        self.adata.obs['pseudotime'] = self.pseudotime
        print("Cell projection uncertainties stored as 'pseudotime' in self.adata.obs")

        if visualize:
            self._adata.obs['pseudotime'] = self.pseudotime
            self.ax = self.plot_backbone(directed = True, color = color, **kwargs)
            if path_to_fig is not None:
                self.ax.figure.savefig(path_to_fig)
            self.ax.figure.show()

        return None



    def differential_expression_test(self, alpha: float = 0.05, cell_subset = None, order: int = 1):
        '''Differentially gene expression test. All (selected and unselected) genes will be tested 
        Only cells in `selected_cell_subset` will be used, which is useful when one need to
        test differentially expressed genes on a branch of the inferred trajectory.

        Parameters
        ----------
        alpha : float, optional
            The cutoff of p-values.
        cell_subset : np.array, optional
            The subset of cells to be used for testing. If None, all cells will be used.
        order : int, optional
            The maxium order we used for pseudotime in regression.

        Returns
        ----------
        res_df : pandas.DataFrame
            The test results of expressed genes with two columns,
            the estimated coefficients and the adjusted p-values.
        '''
        if not hasattr(self, 'pseudotime'):
            raise ReferenceError("Pseudotime does not exist! Please run 'infer_trajectory' first.")
        if cell_subset is None:
            cell_subset = np.arange(self.X_input.shape[0])
            print("All cells are selected.")
        if order < 1:
            raise  ValueError("Maximal order of pseudotime in regression must be at least 1.")

        # Prepare X and Y for regression expression ~ rank(PDT) + covariates
        Y = self.adata.X[cell_subset,:]
#        std_Y = np.std(Y, ddof=1, axis=0, keepdims=True)
#        Y = np.divide(Y-np.mean(Y, axis=0, keepdims=True), std_Y, out=np.empty_like(Y)*np.nan, where=std_Y!=0)
        X = stats.rankdata(self.pseudotime[cell_subset])
        X = ((X-np.mean(X))/np.std(X, ddof=1)).reshape((-1,1))
        X = np.c_[np.ones_like(X), X]
        if order > 1:
            for _order in range(2, order+1):
                X = np.c_[X, X**_order]
        if self.covariates is not None:
            X = np.c_[X, self.covariates[cell_subset, :]]

        res_df = DE_test(Y, X, self.adata.var_names, i_test = np.array(list(range(1,order+1))), alpha = alpha)
        return res_df[res_df.pvalue_adjusted_1 != 0]


 

    def evaluate(self, milestone_net, begin_node_true, grouping = None,
                thres: float = 0.5, no_loop: bool = True, cutoff: Optional[float] = None,
                method: str = 'mean', path: Optional[str] = None):
        ''' Evaluate the model.

        Parameters
        ----------
        milestone_net : pd.DataFrame
            The true milestone network. For real data, milestone_net will be a DataFrame of the graph of nodes.
            Eg.

            from|to
            ---|---
            cluster 1 | cluster 1
            cluster 1 | cluster 2

            For synthetic data, milestone_net will be a DataFrame of the (projected)
            positions of cells. The indexes are the orders of cells in the dataset.
            Eg.

            from|to|w
            ---|---|---
            cluster 1 | cluster 1 | 1
            cluster 1 | cluster 2 | 0.1
        begin_node_true : str or int
            The true begin node of the milestone.
        grouping : np.array, optional
            \([N,]\) The labels. For real data, grouping must be provided.

        Returns
        ----------
        res : pd.DataFrame
            The evaluation result.
        '''
        if not hasattr(self, 'labels_map'):
            raise ValueError("No given labels for training.")

        '''
        # Evaluate for the whole dataset will ignore selected_cell_subset.
        if len(self.selected_cell_subset)!=len(self.cell_names):
            warnings.warn("Evaluate for the whole dataset.")
        '''
        
        # If the begin_node_true, need to encode it by self.le.
        # this dict is for milestone net cause their labels are not merged
        # all keys of label_map_dict are str
        label_map_dict = dict()
        for i in range(self.labels_map.shape[0]):
            label_mapped = self.labels_map.loc[i]
            ## merged cluster index is connected by comma
            for each in label_mapped.values[0].split(","):
                label_map_dict[each] = i
        if isinstance(begin_node_true, str):
            begin_node_true = label_map_dict[begin_node_true]
            
        # For generated data, grouping information is already in milestone_net
        if 'w' in milestone_net.columns:
            grouping = None
            
        # If milestone_net is provided, transform them to be numeric.
        if milestone_net is not None:
            milestone_net['from'] = [label_map_dict[x] for x in milestone_net["from"]]
            milestone_net['to'] = [label_map_dict[x] for x in milestone_net["to"]]

        # this dict is for potentially merged clusters.
        label_map_dict_for_merged_cluster = dict(zip(self.labels_map["label_names"],self.labels_map.index))
        mapped_labels = np.array([label_map_dict_for_merged_cluster[x] for x in self.labels])
        begin_node_pred = int(np.argmin(np.mean((
            self.z[mapped_labels==begin_node_true,:,np.newaxis] -
            self.mu[np.newaxis,:,:])**2, axis=(0,1))))

        if cutoff is None:
            cutoff = 0.01

        G = self.backbone
        w = self.cell_position_projected
        pseudotime = self.pseudotime

        # 1. Topology
        G_pred = nx.Graph()
        G_pred.add_nodes_from(G.nodes)
        G_pred.add_edges_from(G.edges)
        nx.set_node_attributes(G_pred, False, 'is_init')
        G_pred.nodes[begin_node_pred]['is_init'] = True

        G_true = nx.Graph()
        G_true.add_nodes_from(G.nodes)
        # if 'grouping' is not provided, assume 'milestone_net' contains proportions
        if grouping is None:
            G_true.add_edges_from(list(
                milestone_net[~pd.isna(milestone_net['w'])].groupby(['from', 'to']).count().index))
        # otherwise, 'milestone_net' indicates edges
        else:
            if milestone_net is not None:             
                G_true.add_edges_from(list(
                    milestone_net.groupby(['from', 'to']).count().index))
            grouping = [label_map_dict[x] for x in grouping]
            grouping = np.array(grouping)
        G_true.remove_edges_from(nx.selfloop_edges(G_true))
        nx.set_node_attributes(G_true, False, 'is_init')
        G_true.nodes[begin_node_true]['is_init'] = True
        res = topology(G_true, G_pred)
            
        # 2. Milestones assignment
        if grouping is None:
            milestones_true = milestone_net['from'].values.copy()
            milestones_true[(milestone_net['from']!=milestone_net['to'])
                           &(milestone_net['w']<0.5)] = milestone_net[(milestone_net['from']!=milestone_net['to'])
                                                                      &(milestone_net['w']<0.5)]['to'].values
        else:
            milestones_true = grouping
        milestones_true = milestones_true
        milestones_pred = np.argmax(w, axis=1)
        res['ARI'] = (adjusted_rand_score(milestones_true, milestones_pred) + 1)/2
        
        if grouping is None:
            n_samples = len(milestone_net)
            prop = np.zeros((n_samples,n_samples))
            prop[np.arange(n_samples), milestone_net['to']] = 1-milestone_net['w']
            prop[np.arange(n_samples), milestone_net['from']] = np.where(np.isnan(milestone_net['w']), 1, milestone_net['w'])
            res['GRI'] = get_GRI(prop, w)
        else:
            res['GRI'] = get_GRI(grouping, w)
        
        # 3. Correlation between geodesic distances / Pseudotime
        if no_loop:
            if grouping is None:
                pseudotime_true = milestone_net['from'].values + 1 - milestone_net['w'].values
                pseudotime_true[np.isnan(pseudotime_true)] = milestone_net[pd.isna(milestone_net['w'])]['from'].values            
            else:
                pseudotime_true = - np.ones(len(grouping))
                nx.set_edge_attributes(G_true, values = 1, name = 'weight')
                connected_comps = nx.node_connected_component(G_true, begin_node_true)
                subG = G_true.subgraph(connected_comps)
                milestone_net_true = self.inferer.build_milestone_net(subG, begin_node_true)
                if len(milestone_net_true)>0:
                    pseudotime_true[grouping==int(milestone_net_true[0,0])] = 0
                    for i in range(len(milestone_net_true)):
                        pseudotime_true[grouping==int(milestone_net_true[i,1])] = milestone_net_true[i,-1]
            pseudotime_true = pseudotime_true[pseudotime>-1]
            pseudotime_pred = pseudotime[pseudotime>-1]
            res['PDT score'] = (np.corrcoef(pseudotime_true,pseudotime_pred)[0,1]+1)/2
        else:
            res['PDT score'] = np.nan
            
        # 4. Shape
        # score_cos_theta = 0
        # for (_from,_to) in G.edges:
        #     _z = self.z[(w[:,_from]>0) & (w[:,_to]>0),:]
        #     v_1 = _z - self.mu[:,_from]
        #     v_2 = _z - self.mu[:,_to]
        #     cos_theta = np.sum(v_1*v_2, -1)/(np.linalg.norm(v_1,axis=-1)*np.linalg.norm(v_2,axis=-1)+1e-12)

        #     score_cos_theta += np.sum((1-cos_theta)/2)

        # res['score_cos_theta'] = score_cos_theta/(np.sum(np.sum(w>0, axis=-1)==2)+1e-12)
        return res


    def save_model(self, path_to_file: str = 'model.checkpoint',save_adata: bool = False):
        '''Saving model weights.

        Parameters
        ----------
        path_to_file : str, optional
            The path to weight files of pre-trained or trained model
        save_adata : boolean, optional
            Whether to save adata or not.
        '''
        self.vae.save_weights(path_to_file)
        if hasattr(self, 'labels') and self.labels is not None:
            with open(path_to_file + '.label', 'wb') as f:
                np.save(f, self.labels)
        with open(path_to_file + '.config', 'wb') as f:
            self.dim_origin = self.X_input.shape[1]
            np.save(f, np.array([
                self.dim_origin, self.dimensions, self.dim_latent,
                self.model_type, 0 if self.covariates is None else self.covariates.shape[1]], dtype=object))
        if hasattr(self, 'inferer') and hasattr(self, 'uncertainty'):
            with open(path_to_file + '.inference', 'wb') as f:
                np.save(f, np.array([
                    self.pi, self.mu, self.pc_x, self.cell_position_posterior, self.uncertainty,
                    self.z,self.cell_position_variance], dtype=object))
        if save_adata:
            self.adata.write(path_to_file + '.adata.h5ad')


    def load_model(self, path_to_file: str = 'model.checkpoint', load_labels: bool = False, load_adata: bool = False):
        '''Load model weights.

        Parameters
        ----------
        path_to_file : str, optional
            The path to weight files of pre trained or trained model
        load_labels : boolean, optional
            Whether to load clustering labels or not.
            If load_labels is True, then the LatentSpace layer will be initialized basd on the model.
            If load_labels is False, then the LatentSpace layer will not be initialized.
        load_adata : boolean, optional
            Whether to load adata or not.
        '''
        if not os.path.exists(path_to_file + '.config'):
            raise AssertionError('Config file not exist!')
        if load_labels and not os.path.exists(path_to_file + '.label'):
            raise AssertionError('Label file not exist!')

        with open(path_to_file + '.config', 'rb') as f:
            [self.dim_origin, self.dimensions,
             self.dim_latent, self.model_type, cov_dim] = np.load(f, allow_pickle=True)
        self.vae = model.VariationalAutoEncoder(
            self.dim_origin, self.dimensions,
            self.dim_latent, self.model_type, False if cov_dim == 0 else True
        )

        if load_labels:
            with open(path_to_file + '.label', 'rb') as f:
                cluster_labels = np.load(f, allow_pickle=True)
            self.init_latent_space(cluster_labels, dist_thres=0)
            if os.path.exists(path_to_file + '.inference'):
                with open(path_to_file + '.inference', 'rb') as f:
                    arr = np.load(f, allow_pickle=True)
                    if len(arr) == 8:
                        [self.pi, self.mu, self.pc_x, self.cell_position_posterior, self.uncertainty,
                         self.D_JS, self.z,self.cell_position_variance] = arr
                    else:
                        [self.pi, self.mu, self.pc_x, self.cell_position_posterior, self.uncertainty,
                         self.z,self.cell_position_variance] = arr
                self._adata_z = sc.AnnData(self.z)
                sc.pp.neighbors(self._adata_z)
        ## initialize the weight of encoder and decoder
        self.vae.encoder(np.zeros((1, self.dim_origin + cov_dim)))
        self.vae.decoder(np.expand_dims(np.zeros((1,self.dim_latent + cov_dim)),1))

        self.vae.load_weights(path_to_file)

        if load_adata:
            if not os.path.exists(path_to_file + '.adata.h5ad'):
                raise AssertionError('AnnData file not exist!')
            self.adata = sc.read_h5ad(path_to_file + '.adata.h5ad')
            # Jingshu, what are the difference between adata, _adata, adata_z?
            self._adata.obs = self.adata.obs.copy()