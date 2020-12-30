# -*- coding: utf-8 -*-
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skmisc import loess
from sklearn import preprocessing
import warnings
from sklearn.decomposition import PCA
from VITAE.utils import _check_expression, _check_variability

def log_norm(x, K = 1e4):
    '''Normalize the gene expression counts for each cell by the total expression counts, 
    divide this by a size scale factor, which is determined by total counts and a constant K
    then log-transforms the result.

    Parameters
    ----------
    x : np.array
        \([N, G^{raw}]\) The raw count data.
    K : float, optional
        The normalizing constant.

    Returns
    ----------
    x_normalized : np.array
        \([N, G^{raw}]\) The log-normalized data.
    scale_factor : np.array
        \([N, ]\) The scale factors.
    '''          
    scale_factor = np.sum(x,axis=1, keepdims=True)/K
    x_normalized = np.log(x/scale_factor + 1)
    print('min normailized value: ' + str(np.min(x_normalized)))
    print('max normailized value: ' + str(np.max(x_normalized)))
    return x_normalized, scale_factor


def feature_select(x, gene_num = 2000):
    '''Select highly variable genes (HVGs)
    (See [Stuart *et al*, (2019)](https://www.nature.com/articles/nbt.4096) and its early version [preprint](https://www.biorxiv.org/content/10.1101/460147v1.full.pdf)
    Page 12-13: Data preprocessing - Feature selection for individual datasets).

    Parameters
    ----------
    x : np.array
        \([N, G^{raw}]\) The raw count data.
    gene_num : int, optional
        The number of genes to retain.

    Returns
    ----------
    x : np.array
        \([N, G]\) The count data after gene selection.
    index : np.array
        \([G, ]\) The selected index of genes.
    '''     
    

    n, p = x.shape

    # mean and variance of each gene of the unnormalized data  
    mean, var = np.mean(x, axis=0), np.var(x, axis=0, ddof=1)

    # model log10(var)~log10(mean) by local fitting of polynomials of degree 2
    loess_model = loess.loess(np.log10(mean), np.log10(var), 
                    span = 0.3, degree = 2, family='gaussian'
                    )
    loess_model.fit()
    fitted = loess_model.outputs.fitted_values

    # standardized feature
    z = (x - mean)/np.sqrt(10**fitted)

    # clipped the standardized features to remove outliers
    z = np.clip(z, -np.inf, np.sqrt(n))

    # the variance of standardized features across all cells represents a measure of
    # single cell dispersion after controlling for mean expression    
    feature_score = np.sum(z**2, axis=0)/(n-1)
    
    # feature selection
    index = feature_score.argsort()[::-1][0:gene_num]

    # plot scores
    plt.plot(np.log(np.sort(feature_score)))
    threshold = feature_score[index[-1]]
    plt.hlines(np.log(threshold), 1, p)
    plt.show()
    
    return x[:, index], index


def preprocess(adata = None, processed = None, dimred: bool = None, 
            x = None, c = None, label_names = None, raw_cell_names = None, raw_gene_names = None,  
            K: float = 1e4, gene_num: int = 2000, data_type: str = 'UMI', 
            npc: int = 64, random_state=0):
    '''Preprocess count data.

    Parameters
    ----------
    adata : AnnData, optional
        The scanpy object.
    processed : boolean
        Whether adata has been processed.
    dimred : boolean
        Whether the processed adata is after dimension reduction.
    x : np.array, optional
        \([N^{raw}, G^{raw}]\) The raw count matrix.
    c : np.array
        \([N^{raw}, s]\) The covariate matrix.
    label_names : np.array 
        \([N^{raw}, ]\) The true or estimated cell types.
    raw_cell_names : np.array  
        \([N^{raw}, ]\) The names of cells.
    raw_gene_names : np.array
        \([G^{raw}, ]\) The names of genes.
    K : int, optional
        The normalizing constant.
    gene_num : int, optional
        The number of genes to retain.
    data_type : str, optional
        'UMI', 'non-UMI', or 'Gaussian'.
    npc : int, optional
        The number of PCs to retain, only used if `data_type='Gaussian'`.
    random_state : int, optional
        The random state for PCA. With different random states, the resulted PCA scores are slightly different.

    Returns
    ----------
    x_normalized : np.array
        \([N, G]\) The preprocessed matrix.
    expression : np.array
        \([N, G^{raw}]\) The expression matrix after log-normalization and before scaling.
    x : np.array
        \([N, G]\) The raw count matrix after gene selections.
    c : np.array
        \([N, s]\) The covariates.
    cell_names : np.array
        \([N, ]\) The cell names.
    gene_names : np.array
        \([G^{raw}, ]\) The gene names.
    selected_gene_names : 
        \([G, ]\) The selected gene names.
    scale_factor : 
        \([N, ]\) The scale factors.
    labels : np.array
        \([N, ]\) The encoded labels.
    label_names : np.array
        \([N, ]\) The label names.
    le : sklearn.preprocessing.LabelEncoder
        The label encoder.
    gene_scalar : sklearn.preprocessing.StandardScaler
        The gene scaler.
    '''
    # if input is a scanpy data
    if adata is not None:
        import scanpy as sc
        
        # if the input scanpy is processed
        if processed: 
            x_normalized = x = adata.X
            gene_names = adata.var_names.values
            expression = None
            scale_factor = None
        # if the input scanpy is not processed
        else: 
            dimred = False
            x = adata.X.copy()
            adata, expression, gene_names, cell_mask, gene_mask, gene_mask2 = _recipe_seurat(adata, gene_num)
            x_normalized = adata.X.copy()
            scale_factor = adata.obs.counts_per_cell.values / 1e4
            x = x[cell_mask,:][:,gene_mask][:,gene_mask2]
        try:
            label_names = adata.obs.cell_types
        except:
            if label_names is not None:
                label_names = label_names[cell_mask]
        
        cell_names = adata.obs_names.values
        selected_gene_names = adata.var_names.values
        gene_scalar = None
    
    # if input is a count matrix
    else:
        # remove cells that have no expression
        expressed = _check_expression(x)
        print('Removing %d cells without expression.'%(np.sum(expressed==0)))
        x = x[expressed==1,:]
        if c is not None:
            c = c[expressed==1,:]
        if label_names is None:
            label_names = label_names[expressed==1]        
        
        # remove genes without variability
        variable = _check_variability(x)
        print('Removing %d genes without variability.'%(np.sum(variable==0)))
        x = x[:, variable==1]
        gene_names = raw_gene_names[variable==1]

        # log-normalization
        expression, scale_factor = log_norm(x, K)
        
        # feature selection
        x, index = feature_select(x, gene_num)
        selected_expression = expression[:, index]
        
        # per-gene standardization
        gene_scalar = preprocessing.StandardScaler()
        x_normalized = gene_scalar.fit_transform(selected_expression)
    
        cell_names = raw_cell_names[expressed==1]
        selected_gene_names = gene_names[index]


    if (data_type=='Gaussian') and (dimred is False):
        pca = PCA(n_components = npc, random_state=random_state)
        x_normalized = x = pca.fit_transform(x_normalized)

    if c is not None:
        c_scalar = preprocessing.StandardScaler()
        c = c_scalar.fit_transform(c)

    if label_names is None:
        warnings.warn('No labels for cells!')
        labels = None
        le = None
    else:
        le = preprocessing.LabelEncoder()
        labels = le.fit_transform(label_names)
        print('Number of cells in each class: ')
        table = pd.value_counts(label_names)
        table.index = pd.Series(le.transform(table.index).astype(str)) \
            + ' <---> ' + table.index
        table = table.sort_index()
        print(table)
        
    return (x_normalized, expression, x, c, 
        cell_names, gene_names, selected_gene_names, 
        scale_factor, labels, label_names, le, gene_scalar)


def _recipe_seurat(adata, gene_num):
    """
    Normalization and filtering as of Seurat [Satija15]_.
    This uses a particular preprocessing
    """
    cell_mask = sc.pp.filter_cells(adata, min_genes=200, inplace=False)[0]
    adata = adata[cell_mask,:]
    gene_mask = sc.pp.filter_genes(adata, min_cells=3, inplace=False)[0]
    adata = adata[:,gene_mask]
    gene_names = adata.var_names.values

    sc.pp.normalize_total(adata, target_sum=1e4, key_added='counts_per_cell')
    filter_result = sc.pp.filter_genes_dispersion(
        adata.X, min_mean=0.0125, max_mean=3, min_disp=0.5, log=False, n_top_genes=gene_num)
    
    sc.pp.log1p(adata)
    expression = adata.X.copy()
    adata._inplace_subset_var(filter_result.gene_subset)  # filter genes
    sc.pp.scale(adata, max_value=10)
    return adata, expression, gene_names, cell_mask, gene_mask, filter_result.gene_subset