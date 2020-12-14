# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skmisc import loess
from sklearn import preprocessing
import warnings
from sklearn.decomposition import PCA
from VITAE.utils import _check_expression, _check_variability

def log_norm(x, K = 1e4):
    """
    Normalize the gene expression counts for each cell by the total expression counts, 
    divide this by a size scale factor, which is determined by total counts and a constant K
    then log-transforms the result.
    """
    scale_factor = np.sum(x,axis=1, keepdims=True)/K
    x_normalized = np.log(x/scale_factor + 1)
    print('min normailized value: ' + str(np.min(x_normalized)))
    print('max normailized value: ' + str(np.max(x_normalized)))
    return x_normalized, scale_factor


def feature_select(x, gene_num = 2000):
    # https://www.biorxiv.org/content/biorxiv/early/2018/11/02/460147.full.pdf
    # Page 12-13: Data preprocessing - Feature selection for individual datasets

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
    

def label_encoding(labels):
    # encode the class label by number (order of character)
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(labels)
    return y, le


def preprocess(adata, processed, dimred, x, c, label_names, raw_cell_names,
               raw_gene_names, data_type, K = 1e4, gene_num = 2000,  npc = 64):
    '''
    Preprocess count data.
    Params:
        adata           - a scanpy object
        processed       - whether adata has been processed
        dimred          - whether the processed adata is after dimension reduction
        x               - raw count matrix
        c               - covariate matrix
        label_names     - true or estimated cell types
        raw_cell_names  - names of cells
        raw_gene_names  - names of genes
        K               - sum number related to scale_factor
        gene_num        - total number of genes to select    
        data_type       - 'UMI', 'non-UMI' and 'Gaussian'
        npc             - Number of PCs, used if 'data_type' is 'Gaussian'
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
            adata, expression, gene_names, cell_mask, gene_mask, gene_mask2 = recipe_seurat(adata, gene_num)
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
        pca = PCA(n_components = npc)
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


def recipe_seurat(adata, gene_num):
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