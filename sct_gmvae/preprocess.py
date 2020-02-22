# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

import localreg
from sklearn import preprocessing

def normalization():
    """
    LogNormalize that normalizes the feature expression measurements for each cell by the total expression, multiplies this by a size scale factor, and log-transforms the result.
    """

    scale_factor = np.sum(x,axis=1, keepdims=True)/1e4
    x_normalized = np.log(x/scale_factor + 1)
    print(np.min(x_normalized), np.max(x_normalized))

    return None


def feature_select():
    # https://www.biorxiv.org/content/biorxiv/early/2018/11/02/460147.full.pdf
    # Page 12-13: Data preprocessing - Feature selection for individual datasets

    # mean and variance of each gene of the unnormalized data
    mean, var = np.mean(x, axis=0), np.var(x, axis=0)

    # model log10(var)~log10(mean) by local fitting of polynomials of degree 2
    fitted = localreg.localreg(np.log10(mean), np.log10(var),
                               x0=None, degree=2, kernel=localreg.gaussian, frac=0.3)

    # standardized feature
    z = (x - mean)/np.sqrt(10**fitted)

    # clipped the standardized features to remove outliers
    z[z>np.sqrt(n)] = np.sqrt(n)
    # the variance of standardized features across all cells represents a measure of
    # single cell dispersion after controlling for mean expression
    feature_score = np.var(z, axis=0)

    return feature_score
    
def plot_feature_score():
    plt.plot(np.log(np.sort(feature_score)))
    quart_quantile = np.quantile(feature_score,0.45)
    plt.hlines(np.log(quart_quantile), 1, p)
    return None


def label_encoding():
    y = grouping    
    le = preprocessing.LabelEncoder()
    le.fit(np.unique(grouping))
    y = le.transform(grouping)
    le.classes_
    return None

