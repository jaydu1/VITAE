import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dat_NB = pd.read_csv('result/result_VITAE_NB.csv',
                        index_col=0)#.drop(['type'], axis=1)
dat_NB = dat_NB[dat_NB.method == 'modified_map']
dat_NB.method = 'VITAE_NB'

dat_Gaussian = pd.read_csv('result/result_VITAE_Gaussian.csv',
                        index_col=0)#.drop(['type'], axis=1)
dat_Gaussian = dat_Gaussian[dat_Gaussian.method == 'modified_map']
dat_Gaussian.method = 'VITAE_Gauss'

dat_other = pd.read_csv('result/result_other_methods.csv',
                        index_col=0)#.drop(['type'], axis=1)

dat = pd.concat([dat_NB, dat_Gaussian, dat_other])

sources = ['dyngen','our model','real']
scores = ['GED score','IM score','ARI','GRI','PDT score']
cmaps = ['YlOrRd_r', 'YlGn_r', 'RdPu_r', 'PuBu_r', 'BuGn_r']

rotation_xticklabels = 25

sns.set(font_scale=1.5, rc={'axes.facecolor':(0.85,0.85,0.85), 'xtick.labelsize':14, 'ytick.labelsize':11})
fig, ax = plt.subplots(3, 5, gridspec_kw={'height_ratios':[6, 2, 8], 'width_ratios' :[1,1,1,1,1]}, figsize = (20,10))
for i in range(5):
    for j in range(3):
        vmin = dat[scores[i]].min()
        vmax = dat[scores[i]].max() 
        dat_t = dat[dat.source == sources[j]]
        dat_t = dat_t[['data','method',scores[i]]].pivot('data', 'method', scores[i])
        if  j == 0:
            ax[j][i].set_title(scores[i], fontweight='bold', fontsize = 18)
        if (j == 2) & (i == 0):
            sns.heatmap(dat_t, ax = ax[j][i], cbar = True, 
                cbar_kws={"orientation": "horizontal", "pad": 0.2}, vmin=vmin, vmax=vmax, cmap = cmaps[i])
            ax[j][i].set_xticklabels(ax[j][i].get_xticklabels(), rotation=rotation_xticklabels, ha="center")
            # ax[j][i].set_yticklabels(ax[j][i].get_yticklabels(), fontsize=12)#rotation=30)
        elif i == 0:
            sns.heatmap(dat_t, ax = ax[j][i], xticklabels=False, cbar = False, vmin=vmin, vmax=vmax, cmap = cmaps[i])
            # ax[j][i].set_yticklabels(ax[j][i].get_yticklabels(), rotation=30)
        elif j == 2:
            sns.heatmap(dat_t, ax = ax[j][i], yticklabels=False, cbar = True, 
                cbar_kws={"orientation": "horizontal", "pad": 0.2}, vmin=vmin, vmax=vmax, cmap = cmaps[i])
            ax[j][i].set_xticklabels(ax[j][i].get_xticklabels(), rotation=rotation_xticklabels, ha="center")
        else:
            sns.heatmap(dat_t, ax = ax[j][i], xticklabels=False, yticklabels=False, cbar = False, vmin=vmin, vmax=vmax, cmap = cmaps[i])
        if i == 4:
            ax[j][i].set_ylabel(sources[j], rotation=270, fontweight='bold', fontsize = 18, labelpad=20)
            ax[j][i].yaxis.set_label_position("right")
        else:
            ax[j][i].set_ylabel(None)
        ax[j][i].set_xlabel(None)
        
plt.tight_layout()
plt.subplots_adjust(wspace=0.2, hspace=0.1)
fig.savefig('result/comp_heatmap.pdf', bbox_inches='tight')