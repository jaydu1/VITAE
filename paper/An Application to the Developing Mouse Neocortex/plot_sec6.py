# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import matplotlib
from VITAE import VITAE, get_igraph, load_data

file_name = 'mouse_brain_merged'
data = load_data(path='data/',
                 file_name=file_name)

model = VITAE()
model.get_data(data['count'],                   # count or expression matrix, (dense or sparse) numpy array 
               labels = data['grouping'],       # (optional) labels, which will be converted to string
               covariate = data['covariates'],  # (optional) covariates
               gene_names = data['gene_names'], # (optional) gene names, which will be converted to string
               cell_names = data['cell_ids']    # (optional) cell names, which will be converted to string
              )
model.preprocess_data(gene_num = 2000,        # (optional) maximum number of influential genes to keep (the default is 2000)
                      data_type = 'Gaussian', # (optional) data_type can be 'UMI', 'non-UMI' or 'Gaussian' (the default is 'UMI')
                      npc = 64                # (optional) number of PCs to keep if data_type='Gaussian' (the default is 64)
                     )

model.load_model(path_to_file='../weight/tutorial_mouse_brain_merged/mouse_brain_inference.checkpoint', load_labels=True)

import networkx as nx
G = model.comp_inference_score(method='modified_map',  # 'mean', 'modified_mean', 'map', and 'modified_map'
                               no_loop=True            # if no_loop=True, then find the maximum spanning tree
                               )   
days = np.array([i[1:3] for i in data['cell_ids']], dtype=np.float32)
begin_node_pred = model.select_root(days, 'sum')
modified_G, modified_w, pseudotime = model.infer_trajectory(init_node=begin_node_pred,  # initial node for computing pseudotime.
                       cutoff=0.09              # (Optional) cutoff score for edges (the default is 0.01).
                       )  


#-------------------------------------------------------------
# Figure 4
#-------------------------------------------------------------
import seaborn as sns
sns.color_palette()
labels = np.ones((len(model.label_names)))
labels[:10261] = 0

color_ggplot_2 = sns.color_palette()[:2]
colors = np.array([color_ggplot_2[int(i)] for i in labels])     

fig, axes = plt.subplots(1,3, figsize=(30, 10))
df = pd.read_csv('result/Seurat_unintegrated_unadjusted.csv', index_col=[0])
axes[0].scatter(*df.values.T, c=colors,
    s=10, alpha=0.4)
plt.setp(axes[0], xticks=[], yticks=[])

df = pd.read_csv('result/Seurat_integrated_adjusted.csv', index_col=[0])
df['UMAP_1'] = -df['UMAP_1']
for i,x in enumerate(['A','B']):
    axes[1].scatter(*df.values[labels==i].T,
            c=colors[labels==i],
        s=10, alpha=0.4, label=x)
plt.setp(axes[1], xticks=[], yticks=[])
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
            fancybox=True, shadow=True, markerscale=5, ncol=9, handletextpad=0.01, prop={'size': 24})    
axes[2].scatter(*model.inferer.embed_z.T, c=colors,
    s=10, alpha=0.4)
plt.setp(axes[2], xticks=[], yticks=[])
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=None)
plt.savefig("Figure4.png", dpi=300, bbox_inches='tight', pad_inches=0.05)

#-------------------------------------------------------------
# Figure 5
#-------------------------------------------------------------
color_ggplot_15 = [
    "#F8766D", "#E58700", "#C99800", "#A3A500", "#6BB100", 
    "#00BA38", "#00BF7D", "#00C0AF", "#00BCD8", "#00B0F6", 
    "#619CFF", "#B983FF", "#E76BF3", "#FD61D1", "#FF67A4"
    ]

df_slingshot = pd.read_csv('result/Slingshot_cluster_network.csv', index_col=[0])
df = pd.read_csv("result/Seurat_clustering.csv", index_col=[0])
cluster_labels = df.iloc[:,2].values - 1
uni_cluster_labels = np.unique(cluster_labels)

labels = data['grouping']
labels[labels == 'Layer V-VI (Hippo)'] = 'Hippocampus'
uni_labels = np.unique(labels)
uni_labels[[10,13]] = uni_labels[[13,10]]

embed =  df.iloc[:,[0,1]].values
embed[:,0] = -embed[:,0]
NUM_Cluster_CLUSTER = len(uni_cluster_labels)
NUM_CLUSTER = len(uni_labels)

embed_mu = np.zeros((NUM_Cluster_CLUSTER,2))
for i,x in enumerate(uni_cluster_labels):
    embed_mu[i,:] = np.mean(embed[cluster_labels==x,:], axis=0, keepdims=True)
    if i==5:
        embed_mu[i,:] += [1.0, -1.5]
    elif i==10:
        embed_mu[i,:] += [2.0, 2.8]
    elif i==13:
        embed_mu[i,:] += [-3.5, -4.5]
    elif i==14:
        embed_mu[i,:] += [-0.5,-0.5]

import networkx as nx
G = nx.Graph()
G.add_nodes_from(np.unique(df_slingshot[['from', 'to']].values.flatten()))
G.add_weighted_edges_from(df_slingshot[['from', 'to','length']].values.tolist())

from VITAE.utils import _get_smooth_curve
import matplotlib.patheffects as pe

init_node = 5
connected_comps = nx.node_connected_component(G, init_node)
subG = G.subgraph(connected_comps)

milestone_net = model.inferer.build_milestone_net(subG,init_node)
select_edges = milestone_net[:,:2]


fig, axes = plt.subplots(1,2, figsize=(24, 10))
for i,x in enumerate(uni_labels):
    axes[0].scatter(*embed[labels==x].T, c=[color_ggplot_15[i]],
        s=16, alpha=0.4, label=str(x))
    dx = dy =0
    if i==1:
        dx = -0.5,
        dy = -0.5
    elif i==3:
        dx = -1.0
        dy = -1.0
    elif i==5:
        dx = 1.5
        dy = 0.5
    elif i==9:
        dx = 0.7
        dy = -1.5
    elif i==10:
        dx = -0.3
    
    elif i==11:
        dx = 0.5
        dy = 0.5
    elif i==2:
        dx = -1.0
        dy = -1.0
    elif i==12:
        dx = -1.0
        dy = 0.0

    axes[0].text(np.mean(embed[labels==x,0])+dx, 
             np.mean(embed[labels==x,1])+dy, str(i+1), fontsize=28)

axes[0].scatter(*embed_mu.T, s=200, color='k', marker='*',
                                linewidth=1.5, edgecolor='white', zorder=3)

for i in range(len(select_edges)):
    points = embed[(cluster_labels==select_edges[i,0])|(cluster_labels==select_edges[i,1]),:]
    points = points[points[:,0].argsort()]    
    points = points[~(
        (np.sum(points < embed_mu[select_edges[i,0], :], axis=-1)==2) & 
        (np.sum(points < embed_mu[select_edges[i,1], :], axis=-1)==2)
        ),:]
    points = points[~(
        (np.sum(points > embed_mu[select_edges[i,0], :], axis=-1)==2) & 
        (np.sum(points > embed_mu[select_edges[i,1], :], axis=-1)==2)
        ),:]
    if i==7:
        fixed_points = np.r_[embed_mu[select_edges[i,0:1], :], 
                             np.mean(embed_mu[select_edges[i,:], :],axis=0,keepdims=True)+[-0.5,0.5],
                             embed_mu[select_edges[i,1:2], :]
                             ]
    else:
        fixed_points = embed_mu[select_edges[i,:], :]
    x_smooth, y_smooth = _get_smooth_curve(
        points, 
        fixed_points
        )
    axes[0].plot(x_smooth, y_smooth, 
        '-', 
        linewidth= 3,
        color="black", 
        alpha=0.8, 
        path_effects=[pe.Stroke(linewidth=1, 
                                foreground='white'), pe.Normal()],
        zorder=1
        )

    if i==7:
        delta_x = embed_mu[select_edges[i,1], 0]-x_smooth[-15]
        delta_y = embed_mu[select_edges[i,1], 1]-y_smooth[-15]
    else:
        delta_x = embed_mu[select_edges[i,1], 0]-x_smooth[-2]
        delta_y = embed_mu[select_edges[i,1], 1]-y_smooth[-2]
    length = np.sqrt(delta_x**2 + delta_y**2) * 1.5                
    axes[0].arrow(
            embed_mu[select_edges[i,1], 0]-delta_x/length, 
            embed_mu[select_edges[i,1], 1]-delta_y/length, 
            delta_x/length,
            delta_y/length,
            color='black', alpha=1.0,
            shape='full', lw=0, length_includes_head=True, head_width=0.6, zorder=2)


# our method
for i,x in enumerate(uni_labels):
    axes[1].scatter(*model.inferer.embed_z[labels==x].T, c=[color_ggplot_15[i]],
        s=16, alpha=0.4, label='%02d'%(i+1)+'. '+str(x))
    
    dx = dy =0
    if i == 2:
        dx = -0.02
        dy = -0.02
    elif i==3:
        dy = 0.01
    elif i==5:
        dx = 0.03
        dy = -0.01
    elif i==9:
        dx = -0.04
    elif i==10:
        dx = -0.05
        dy = -0.07
    elif i==11:
        dy = 0.01
    elif i==12:
        dx = 0.03
    elif i==13:
        dx = -0.06
        dy = -0.02
    axes[1].text(np.mean(model.inferer.embed_z[labels==x,0])+dx, 
            np.mean(model.inferer.embed_z[labels==x,1])+dy, str(i+1), fontsize=28)
    

G = model.inferer.G
cutoff = 0.09
init_node = 3
graph = nx.to_numpy_matrix(G)
graph[graph<=cutoff] = 0
G = nx.from_numpy_array(graph)
if len(G.edges)>0:
    connected_comps = nx.node_connected_component(G, init_node)
    subG = G.subgraph(connected_comps)
    milestone_net = model.inferer.build_milestone_net(subG,init_node)
    select_edges = milestone_net[:,:2]
    select_edges_score = graph[select_edges[:,0], select_edges[:,1]]
    select_edges_score = (select_edges_score - select_edges_score.min()
        )/(select_edges_score.max() - select_edges_score.min())*4
else:
    milestone_net = select_edges = []                    

# modify w_tilde
w = model.inferer.modify_wtilde(model.inferer.w_tilde, select_edges)

# compute pseudotime
pseudotime = model.inferer.comp_pseudotime(milestone_net, init_node, w)
embed_mu = model.inferer.embed_mu.copy()

axes[1].scatter(*embed_mu.T, s=200, color='k', marker='*', 
                linewidth=1.5, edgecolor='white',
                zorder=3)    
for i in range(len(select_edges)):
    points = model.inferer.embed_z[np.sum(w[:,select_edges[i,:]]>0, axis=-1)==2,:]
    points = points[points[:,0].argsort()]  
    if select_edges[i,0]==0 and select_edges[i,1]==5:
        # for better visualization purpose
        points = points[points[:,1]> np.min(model.inferer.embed_mu[select_edges[i,:], 1])-0.02,:]
        points = np.r_[points, 
                       model.inferer.embed_mu[select_edges[i,0], :] + np.arange(50).reshape((50,1))/50 * np.array([[0.05,-0.1]])
                       ]              
    x_smooth, y_smooth = _get_smooth_curve(
        points, 
        model.inferer.embed_mu[select_edges[i,:], :]
        )
    axes[1].plot(x_smooth, y_smooth, 
        '-', 
        linewidth= 2 + select_edges_score[0,i],
        color="black", 
        alpha=0.8, 
        path_effects=[pe.Stroke(linewidth=2+select_edges_score[0,i]+1.5, 
                                foreground='white'), pe.Normal()],
        zorder=1
        )

    delta_x = model.inferer.embed_mu[select_edges[i,1], 0]-x_smooth[-2]
    delta_y = model.inferer.embed_mu[select_edges[i,1], 1]-y_smooth[-2]
    length = np.sqrt(delta_x**2 + delta_y**2) * 50           
    axes[1].arrow(
            model.inferer.embed_mu[select_edges[i,1], 0]-delta_x/length, 
            model.inferer.embed_mu[select_edges[i,1], 1]-delta_y/length, 
            delta_x/length,
            delta_y/length,
            color='black', alpha=1.0,
            shape='full', lw=0, length_includes_head=True, head_width=0.02, zorder=2)

box = axes[1].get_position()
axes[1].set_position([box.x0 + box.width*0.1, box.y0,
                    box.width * 0.9, box.height])
leg = axes[1].legend(loc='upper center', bbox_to_anchor=(1.27, 1.0),
            fancybox=False, shadow=False, markerscale=5, ncol=1, 
            frameon=False, handletextpad=0.01,
            prop={'size': 24})
for lh in leg.legendHandles: 
    lh.set_alpha(1)
axes[0].set_xlabel('(a)', fontsize=28, labelpad=30)
axes[1].set_xlabel('(b)', fontsize=28, labelpad=30)
plt.setp(axes[0], xticks=[], yticks=[])
plt.setp(axes[1], xticks=[], yticks=[])
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=None)
plt.savefig("Figure5a-b.png", dpi=300, bbox_inches='tight', pad_inches=0.05)

uncertainty = np.sum((modified_w - model.w_tilde)**2, axis=-1) + np.sum(model.var_w_tilde, axis=-1)
import matplotlib
fig, ax = plt.subplots(1,1,figsize=(15,10))
cmap = matplotlib.cm.get_cmap('RdBu_r')
sc = ax.scatter(*model.inferer.embed_z.T,
    c=uncertainty,
    cmap=cmap,
    s=16, alpha=1)
cbar = plt.colorbar(sc, ax=ax)
plt.setp(ax, xticks=[], yticks=[])
cbar.ax.tick_params(labelsize=18)
plt.savefig('Figure5c.png', dpi=300, bbox_inches='tight', pad_inches=0.05)

labels = np.array([i[:3] for i in data['cell_ids']])
n_labels = len(np.unique(labels))
colors = [plt.cm.Spectral_r(float(i)/n_labels) for i in range(n_labels)]

embed =  df.iloc[:,[0,1]].values
embed[:,0] = - embed[:,0]

fig, ax = plt.subplots(1,1, figsize=(12, 10))
for i,x in enumerate(np.unique(labels)):
    ax.scatter(*model.inferer.embed_z[labels==x].T, c=[colors[i]],
        s=30, alpha=0.7-i*0.05, label=str(x)+'.5'
        )
    
box = ax.get_position()
ax.set_position([box.x0 + box.width*0.1, box.y0,
                    box.width * 0.9, box.height])
leg = ax.legend(loc='upper center', bbox_to_anchor=(1.15, 1.0),
            fancybox=False, shadow=False, markerscale=5, ncol=1, 
            frameon=False, handletextpad=0.01,
            prop={'size': 24})
for lh in leg.legendHandles: 
    lh.set_alpha(1)
plt.setp(ax, xticks=[], yticks=[])
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=None)
plt.savefig("Figure5d.png", dpi=300, bbox_inches='tight', pad_inches=0.05)


#-------------------------------------------------------------
# subfigures of Figure 6
#-------------------------------------------------------------
labels = data['grouping']
labels[labels == 'Layer V-VI (Hippo)'] = 'Hippocampus'
uni_labels = np.unique(labels)
uni_labels[[10,13]] = uni_labels[[13,10]]
NUM_CLUSTER = len(uni_labels)

import networkx as nx
import matplotlib.patheffects as pe
fig, axes = plt.subplots(1,2, figsize=(24, 10))

id_branch = ((modified_w[:,3] > 0.0)&(modified_w[:,0] > 0.0)) | \
    ((modified_w[:,0] > 0.0)&(modified_w[:,11] > 0.0)) | \
    (modified_w[:,3] > 0.99) | \
    (modified_w[:,0] > 0.99) | \
    (modified_w[:,11] > 0.99)

for i,x in enumerate(uni_labels):
    axes[0].scatter(*model.inferer.embed_z[(labels==x)&id_branch].T, c=[color_ggplot_15[i]],
        s=30, alpha=0.5, label='%02d'%(i+1)+'. '+str(x))
    
    dx = dy =0
    if i == 2:
        dx = -0.02
        dy = -0.02
    elif i==3:
        dy = 0.01
    elif i==5:
        dx = 0.03
        dy = -0.01
    elif i==9:
        dx = -0.04
    elif i==10:
        dx = -0.05
        dy = -0.07
    elif i==11:
        dy = 0.01
    elif i==12:
        dx = 0.03
    elif i==13:
        dx = -0.06
        dy = -0.02
    axes[0].text(np.mean(model.inferer.embed_z[labels==x,0])+dx, 
            np.mean(model.inferer.embed_z[labels==x,1])+dy, str(i+1), fontsize=28)
axes[0].scatter(*model.inferer.embed_z[~id_branch].T, c='gainsboro',
        s=16, alpha=0.2)    

G = model.inferer.G
cutoff = 0.09
init_node = 3
graph = nx.to_numpy_matrix(G)
graph[graph<=cutoff] = 0
G = nx.from_numpy_array(graph)
if len(G.edges)>0:
    connected_comps = nx.node_connected_component(G, init_node)
    subG = G.subgraph(connected_comps)
    milestone_net = model.inferer.build_milestone_net(subG,init_node)
    select_edges = milestone_net[:,:2]
    select_edges_score = graph[select_edges[:,0], select_edges[:,1]]
    select_edges_score = (select_edges_score - select_edges_score.min())/(select_edges_score.max() - select_edges_score.min())*4
else:
    milestone_net = select_edges = []                    

# modify w_tilde
w = model.inferer.modify_wtilde(model.inferer.w_tilde, select_edges)

# compute pseudotime
pseudotime = model.inferer.comp_pseudotime(milestone_net, init_node, w)
embed_mu = model.inferer.embed_mu.copy()

axes[0].scatter(*embed_mu.T, s=200, color='k', marker='*', 
                linewidth=1.5, edgecolor='white',
                zorder=3)    
for i in range(len(select_edges)):
    if np.sum(np.sum(np.abs(select_edges[i:i+1,:] - np.array([[5,7],[7,0],[0,11]])), axis=-1)<1e-12)>0:
        color = 'black'
        alpha = 0.8
    else:
        color = 'gray'
        alpha = 0.6
    points = model.inferer.embed_z[np.sum(w[:,select_edges[i,:]]>0, axis=-1)==2,:]
    points = points[points[:,0].argsort()]                
    x_smooth, y_smooth = _get_smooth_curve(
        points, 
        model.inferer.embed_mu[select_edges[i,:], :]
        )
    axes[0].plot(x_smooth, y_smooth, 
        '-', 
        linewidth= 2 + select_edges_score[0,i],
        color=color, 
        alpha=alpha, 
        path_effects=[pe.Stroke(linewidth=2+select_edges_score[0,i]+1.5, 
                                foreground='white'), pe.Normal()],
        zorder=1
        )

    delta_x = model.inferer.embed_mu[select_edges[i,1], 0]-x_smooth[-2]
    delta_y = model.inferer.embed_mu[select_edges[i,1], 1]-y_smooth[-2]
    length = np.sqrt(delta_x**2 + delta_y**2) * 50               
    axes[0].arrow(
            model.inferer.embed_mu[select_edges[i,1], 0]-delta_x/length, 
            model.inferer.embed_mu[select_edges[i,1], 1]-delta_y/length, 
            delta_x/length,
            delta_y/length,
            color=color, alpha=alpha,
            shape='full', lw=0, length_includes_head=True, head_width=0.02, zorder=2)

# subplot 2
id_branch = ((modified_w[:,0] > 0.0)&(modified_w[:,5] > 0.0)) | \
    ((modified_w[:,5] > 0.0)&(modified_w[:,4] > 0.0)) | \
    ((modified_w[:,4] > 0.0)&(modified_w[:,1] > 0.0)) | \
    (modified_w[:,0] > 0.99) | \
    (modified_w[:,5] > 0.99) | \
    (modified_w[:,4] > 0.99) | \
    (modified_w[:,1] > 0.99)

for i,x in enumerate(uni_labels):
    axes[1].scatter(*model.inferer.embed_z[(labels==x)&id_branch].T, c=[color_ggplot_15[i]],
        s=30, alpha=0.5, label='%02d'%(i+1)+'. '+str(x))
    
    dx = dy =0
    if i == 2:
        dx = -0.02
        dy = -0.02
    elif i==3:
        dy = 0.01
    elif i==5:
        dx = 0.03
        dy = -0.01
    elif i==9:
        dx = -0.04
    elif i==10:
        dx = -0.05
        dy = -0.07
    elif i==11:
        dy = 0.01
    elif i==12:
        dx = 0.03
    elif i==13:
        dx = -0.06
        dy = -0.02
    axes[1].text(np.mean(model.inferer.embed_z[labels==x,0])+dx, 
            np.mean(model.inferer.embed_z[labels==x,1])+dy, str(i+1), fontsize=28)
axes[1].scatter(*model.inferer.embed_z[~id_branch].T, c='gainsboro',
        s=16, alpha=0.2)   
    
axes[1].scatter(*embed_mu.T, s=200, color='k', marker='*', 
                linewidth=1.5, edgecolor='white',
                zorder=3)    
for i in range(len(select_edges)):
    if np.sum(np.sum(np.abs(select_edges[i:i+1,:] - np.array([[7,4],[4,6],[6,1]])), axis=-1)<1e-12)>0:
        color = 'black'
        alpha = 0.8
    else:
        color = 'gray'
        alpha = 0.6
    points = model.inferer.embed_z[np.sum(w[:,select_edges[i,:]]>0, axis=-1)==2,:]
    points = points[points[:,0].argsort()]                
    x_smooth, y_smooth = _get_smooth_curve(
        points, 
        model.inferer.embed_mu[select_edges[i,:], :]
        )
    axes[1].plot(x_smooth, y_smooth, 
        '-', 
        linewidth= 2 + select_edges_score[0,i],
        color=color, 
        alpha=alpha, 
        path_effects=[pe.Stroke(linewidth=2+select_edges_score[0,i]+1.5, 
                                foreground='white'), pe.Normal()],
        zorder=1
        )

    delta_x = model.inferer.embed_mu[select_edges[i,1], 0]-x_smooth[-2]
    delta_y = model.inferer.embed_mu[select_edges[i,1], 1]-y_smooth[-2]
    length = np.sqrt(delta_x**2 + delta_y**2) * 50               
    axes[1].arrow(
            model.inferer.embed_mu[select_edges[i,1], 0]-delta_x/length, 
            model.inferer.embed_mu[select_edges[i,1], 1]-delta_y/length, 
            delta_x/length,
            delta_y/length,
            color=color, alpha=alpha,
            shape='full', lw=0, length_includes_head=True, head_width=0.02, zorder=2)

box = axes[1].get_position()
axes[1].set_position([box.x0 + box.width*0.1, box.y0,
                    box.width * 0.9, box.height])
leg = axes[1].legend(loc='upper center', bbox_to_anchor=(1.27, 1.0),
            fancybox=False, shadow=False, markerscale=5, ncol=1, 
            frameon=False, handletextpad=0.01,
            prop={'size': 24})
for lh in leg.legendHandles: 
    lh.set_alpha(1)
axes[0].set_xlabel('(a)', fontsize=28, labelpad=30)
axes[1].set_xlabel('(b)', fontsize=28, labelpad=30)
plt.setp(axes[0], xticks=[], yticks=[])
plt.setp(axes[1], xticks=[], yticks=[])
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=None)
plt.savefig("subFigure6.png", dpi=300, bbox_inches='tight', pad_inches=0.05)
