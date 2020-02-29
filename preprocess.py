# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import localreg
from sklearn import preprocessing

def normalization(x, K = 1e4):
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
    
    return x[:,index], index
    

def label_encoding(grouping):
    # encode the class label by number (order of character)
    le = preprocessing.LabelEncoder()
    le.fit(np.unique(grouping))
    y = le.transform(grouping)
    return y, le


def preprocess(x, grouping = None, K = 1e4, gene_num = 2000):
    '''
    input 
    x: raw count matrix
    K: sum number related to scale_factor
    gene_num: total number of genes to select
    grouping: true labels
    '''
    if grouping is None:
        import warnings
        warnings.warn('No labels for cells!')
        label = np.ones(len(x))
        le = None
    else:
        label, le = label_encoding(grouping)
        print('Number of cells in each class: ')
        table = pd.value_counts(grouping)
        table.index = pd.Series(le.transform(table.index).astype(str)) + ' <---> ' + table.index
        print(table)

    x_normalized, scale_factor = normalization(x, K)
    x, index = feature_select(x, gene_num)
    return x_normalized[:,index], x, scale_factor, label, le


