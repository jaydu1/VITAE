#!/usr/local/bin/python

import dynclipy
task = dynclipy.main()

# avoid errors due to no $DISPLAY environment variable available when running sc.pl.paga
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import h5py
import json

import scanpy as sc
import anndata
import numba
import warnings

import time
checkpoints = {}

#   ____________________________________________________________________________
#   Load data                                                               ####

counts = task["counts"]

parameters = task["parameters"]

start_id = task["priors"]["start_id"]
if isinstance(start_id, list):
  start_id = start_id[0]

if "groups_id" in task["priors"]:
  groups_id = task["priors"]['groups_id']
else:
  groups_id = None

# create dataset
if groups_id is not None:
  obs = pd.DataFrame(groups_id)
  obs.index = groups_id["cell_id"]
  obs["louvain"] = obs["group_id"].astype("category")
  adata = anndata.AnnData(counts)
  adata.obs = obs
else:
  adata = anndata.AnnData(counts)

checkpoints["method_afterpreproc"] = time.time()

#   ____________________________________________________________________________
#   Basic preprocessing                                                     ####

# normalisation & filtering
if counts.shape[1] < 100 and parameters["filter_features"]:
  print("You have less than 100 features, but the filter_features parameter is true. This will likely result in an error. Disable filter_features to avoid this")

if parameters["filter_features"]:
  n_top_genes = min(2000, counts.shape[1])
  sc.pp.recipe_zheng17(adata, n_top_genes=n_top_genes)

# precalculating some dimensionality reductions
sc.tl.pca(adata, n_comps=parameters["n_comps"])
with warnings.catch_warnings():
  warnings.simplefilter('ignore', numba.errors.NumbaDeprecationWarning)
  sc.pp.neighbors(adata, n_neighbors=parameters["n_neighbors"])

# denoise the graph by recomputing it in the first few diffusion components
if parameters["n_dcs"] != 0:
  sc.tl.diffmap(adata, n_comps=parameters["n_dcs"])

#   ____________________________________________________________________________
#   Cluster, infer trajectory, infer pseudotime, compute dimension reduction ###

# add grouping if not provided
if groups_id is None:
  sc.tl.louvain(adata, resolution=parameters["resolution"])

# run paga
sc.tl.paga(adata)

# compute a layout for the paga graph
# - this simply uses a Fruchterman-Reingold layout, a tree layout or any other
#   popular graph layout is also possible
# - to obtain a clean visual representation, one can discard low-confidence edges
#   using the parameter threshold
sc.pl.paga(adata, threshold=0.01, layout='fr', show=False)

# run dpt for pseudotime information that is overlayed with paga
adata.uns['iroot'] = np.where(adata.obs.index == start_id)[0][0]
if parameters["n_dcs"] == 0:
  sc.tl.diffmap(adata)
sc.tl.dpt(adata, n_dcs = min(adata.obsm["X_diffmap"].shape[1], 10))

# run umap for a dimension-reduced embedding, use the positions of the paga
# graph to initialize this embedding
if parameters["embedding_type"] == 'umap':
  sc.tl.umap(adata, init_pos='paga')
  dimred_name = 'X_umap'
else:
  sc.tl.draw_graph(adata, init_pos='paga')
  dimred_name = "X_draw_graph_" + parameters["embedding_type"]

checkpoints["method_aftermethod"] = time.time()

#   ____________________________________________________________________________
#   Process & save output                                                   ####

# grouping
grouping = pd.DataFrame({"cell_id": adata.obs.index, "group_id": adata.obs.louvain})

# milestone network
milestone_network = pd.DataFrame(
  adata.uns["paga"]["connectivities_tree"].todense(),
  index=adata.obs.louvain.cat.categories,
  columns=adata.obs.louvain.cat.categories
).stack().reset_index()
milestone_network.columns = ["from", "to", "length"]
milestone_network = milestone_network.query("length > 0").reset_index(drop=True)
milestone_network["directed"] = False

print(milestone_network)

# dimred
dimred = pd.DataFrame([x for x in adata.obsm[dimred_name].T]).T
dimred.columns = ["comp_" + str(i+1) for i in range(dimred.shape[1])]
dimred["cell_id"] = adata.obs.index

# branch progressions: the scaled dpt_pseudotime within every cluster
branch_progressions = adata.obs
branch_progressions["dpt_pseudotime"] = branch_progressions["dpt_pseudotime"].replace([np.inf, -np.inf], 1) # replace unreachable pseudotime with maximal pseudotime
branch_progressions["percentage"] = branch_progressions.groupby("louvain")["dpt_pseudotime"].apply(lambda x: (x-x.min())/(x.max() - x.min())).fillna(0.5)
branch_progressions["cell_id"] = adata.obs.index
branch_progressions["branch_id"] = branch_progressions["louvain"].astype(np.str)
branch_progressions = branch_progressions[["cell_id", "branch_id", "percentage"]]


# branches:
# - length = difference between max and min dpt_pseudotime within every cluster
# - directed = not yet correctly inferred
branches = adata.obs.groupby("louvain").apply(lambda x: x["dpt_pseudotime"].max() - x["dpt_pseudotime"].min()).reset_index()
branches.columns = ["branch_id", "length"]
branches["branch_id"] = branches["branch_id"].astype(np.str)
branches["directed"] = True
print(branches)

# branch network: determine order of from and to based on difference in average pseudotime
branch_network = milestone_network[["from", "to"]]
average_pseudotime = adata.obs.groupby("louvain")["dpt_pseudotime"].mean()
for i, (branch_from, branch_to) in enumerate(zip(branch_network["from"], branch_network["to"])):
  if average_pseudotime[branch_from] > average_pseudotime[branch_to]:
    branch_network.at[i, "to"] = branch_from
    branch_network.at[i, "from"] = branch_to
print(branch_network)

# save
dataset = dynclipy.wrap_data(cell_ids = adata.obs.index)
dataset.add_branch_trajectory(
  grouping = grouping,
  branch_progressions = branch_progressions,
  branches = branches,
  branch_network = branch_network
)
dataset.add_dimred(dimred = dimred)
dataset.add_timings(checkpoints)

dataset.write_output(task["output"])
