# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import localreg
from sklearn import preprocessing
import warnings
from sklearn.decomposition import PCA

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

    # no expression cell
    expressed = np.where(np.sum(x,axis=1) != 0)[0]
    x = x[expressed,:]

    # mean and variance of each gene of the unnormalized data
    mean, var = np.mean(x, axis=0), np.var(x, axis=0)
    x = x[:, (mean > 0) & (var > 0)]
    n, p = x.shape
    mean, var = np.mean(x, axis=0), np.var(x, axis=0)

    # model log10(var)~log10(mean) by local fitting of polynomials of degree 2
    fitted = localreg.localreg(np.log10(mean), np.log10(var), frac = 0.3,
                               degree = 2, kernel = localreg.gaussian)
    # standardized feature
    z = (x - mean)/np.sqrt(10**fitted)

    # clipped the standardized features to remove outliers
    z = np.clip(z, -np.sqrt(n), np.sqrt(n))

    # the variance of standardized features across all cells represents a measure of
    # single cell dispersion after controlling for mean expression
    feature_score = np.var(z, axis=0)
    
    # feature selection
    index = feature_score.argsort()[::-1][0:gene_num]

    # plot scores
    plt.plot(np.log(np.sort(feature_score)))
    threshold = feature_score[index[-1]]
    plt.hlines(np.log(threshold), 1, p)
    plt.show()
    
    return x[:, index], index, expressed
    

def label_encoding(labels):
    # encode the class label by number (order of character)
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(labels)
    return y, le


def preprocess(x, label_names, raw_cell_names, raw_gene_names, 
                data_type, K = 1e4, gene_num = 2000,  npc = 64):
    '''
    Params: 
    x               - raw count matrix
    label_names     - true or estimated cell types
    raw_cell_names  - names of cells
    raw_gene_names  - names of genes
    K               - sum number related to scale_factor
    gene_num        - total number of genes to select    
    data_type       - 'UMI', 'non-UMI' and 'Gaussian'
    npc             - Number of PCs, used if 'data_type' is 'Gaussian'
    '''
    # log-normalization
    x_normalized, scale_factor = log_norm(x, K)
    
    # feature selection
    x, index, expressed = feature_select(x, gene_num)
    x_normalized = x_normalized[expressed, :][:, index]
    scale_factor = scale_factor[expressed, :]
    
    # per-gene standardization
    gene_scalar = preprocessing.StandardScaler()
    x_normalized = gene_scalar.fit_transform(x_normalized)

    if data_type=='Gaussian':
        pca = PCA(n_components = npc)
        x_normalized = x = pca.fit_transform(x_normalized)

    if label_names is None:
        warnings.warn('No labels for cells!')
        labels = None
        le = None
    else:
        label_names = label_names[expressed]
        le = preprocessing.LabelEncoder()
        labels = le.fit_transform(label_names)
        print('Number of cells in each class: ')
        table = pd.value_counts(label_names)
        table.index = pd.Series(le.transform(table.index).astype(str)) \
            + ' <---> ' + table.index
        table = table.sort_index()
        print(table)
        
    cell_names = None if raw_cell_names is None else raw_cell_names[expressed]
    gene_names = None if raw_gene_names is None else raw_gene_names[index]

    return x_normalized, x, cell_names, gene_names, scale_factor, labels, label_names, le, gene_scalar


