library(tidyverse)
library(dplyr)
library(dyno)
library(ggplot2)

prepare_traj = function(path, root_milestone_id, gene_name = NULL, Pos_var = F) {
  # IO
  cell_ids = read.csv(paste0(path, 'cell_ids.csv'), header = F, stringsAsFactors = F)[,1]
  feature_ids = read.csv(paste0(path, 'feature_ids.csv'), header = F, stringsAsFactors = F)[,1]
  grouping = read.csv(paste0(path, 'grouping.csv'), header = F, stringsAsFactors = F)[,1]
  if (is.numeric(grouping)) {
    grouping = paste0('M', grouping)
  }
  names(grouping) = cell_ids
  dimred = read.csv(paste0(path, 'dimred.csv'), header = F, sep = '')
  rownames(dimred) = cell_ids
  milestone_network = as_tibble(read.csv(paste0(path, 'milestone_network.csv'), stringsAsFactors = F,header = T))
  if (nrow(milestone_network) > 0) {
    milestone_network$directed = T
  }
  milestone_percentages = as_tibble(read.csv(paste0(path, 'milestone_percentages.csv'), stringsAsFactors = F,header = T))
  pseudotime = read.csv(paste0(path, 'pseudotime.csv'),header = F)[,1]
  names(pseudotime) = cell_ids
  if (Pos_var) {
    pos_var = read.csv(paste0(path, 'pos_var.csv'),header = F)[,1]
    names(pos_var) = cell_ids
  }
  if (!is.null(gene_name)) {
    gene_express = read.csv(paste0(path, 'gene_express.csv'), header = F) 
    rownames(gene_express) = cell_ids
    colnames(gene_express) = gene_name
  }
  # prepare from example traj
  data("example_bifurcating")
  trajectory = example_bifurcating
  trajectory$id = 'scTGMVAE'
  trajectory$cell_ids = cell_ids
  trajectory$cell_info = tibble(cell_id = cell_ids)
  trajectory$source = 'scTGMVAE'
  trajectory$model = 'scTGMVAE'
  trajectory$milestone_ids = unique(milestone_percentages$milestone_id)
  trajectory$milestone_network = milestone_network
  trajectory$divergence_regions = tibble(divergence_id = character(), milestone_id = character(), is_start = logical())
  trajectory$milestone_percentages = milestone_percentages
  percen2progres = function(df, milestone_network){
    if (nrow(df) == 1) {
      return(tibble(from = df$milestone_id[1], to = df$milestone_id[1], percentage = 1))
    } else {
      for (j in 1:nrow(milestone_network)){
        if ((df$milestone_id[1] == milestone_network$from[j]) & (df$milestone_id[2] == milestone_network$to[j])){
          return(tibble(from = df$milestone_id[1], to = df$milestone_id[2], percentage = df$percentage[2]))
        } else if ((df$milestone_id[2] == milestone_network$from[j]) & (df$milestone_id[1] == milestone_network$to[j])){
          return(tibble(from = df$milestone_id[2], to = df$milestone_id[1], percentage = df$percentage[1]))
        } else {
          more = which.max(df$percentage)
          return(tibble(from = df$milestone_id[-more], to = df$milestone_id[more], percentage = 1))
        }
      }
    }
  }
  trajectory$progressions = milestone_percentages %>% group_by(cell_id) %>% 
    group_modify(~ percen2progres(.x, milestone_network = milestone_network))
  trajectory$trajectory_type = 'Unknown'
  trajectory$waypoint_cells = NULL
  trajectory$counts = NULL
  if (!is.null(gene_name)){
    trajectory$expression = gene_express  
  } else {
    trajectory$expression = NULL
  }
  trajectory$expression_projected = NULL
  trajectory$feature_info = tibble(feature_id = feature_ids, housekeeping = FALSE)
  trajectory$tde_overall = tibble(feature_id = feature_ids, differentially_expressed = TRUE)
  trajectory$prior_information = NULL
  trajectory$root_milestone_id = root_milestone_id
  # for plotting
  trajectory$dimred = dimred
  trajectory$grouping = grouping
  trajectory$pseudotime = pseudotime
  if (Pos_var) {
    trajectory$pos_var = pos_var
  }

  return(trajectory)
}

gene_name = 'G15'
trajectory = prepare_traj('', root_milestone_id = 'M1', gene_name, pos_var = T)

pdf('dimred.pdf', width = 16, height = 8, onefile = F)
patchwork::wrap_plots(
  plot_dimred(trajectory, dimred = trajectory$dimred, hex_cells = FALSE) + ggtitle('Cell ordeing'),
  plot_dimred(trajectory, color_cells = 'grouping', grouping = trajectory$grouping, dimred = trajectory$dimred, hex_cells = FALSE) + ggtitle('True grouping'),
  plot_dimred(trajectory, color_cells = 'feature', feature_oi = gene_name, dimred = trajectory$dimred, hex_cells = FALSE) + ggtitle('Gene expression'),
  plot_dimred(trajectory, color_cells = 'pseudotime', pseudotime = trajectory$pseudotime, dimred = trajectory$dimred, hex_cells = FALSE) + ggtitle('Pseudotime'),
  plot_dimred(trajectory, color_cells = 'pseudotime', pseudotime = trajectory$pos_var, dimred = trajectory$dimred, hex_cells = FALSE) +
    viridis::scale_color_viridis("posterior_variance", option = 'plasma') + ggtitle('Posterior variance')
)
dev.off()

pdf('graph.pdf', width = 16, height = 8, onefile = F)
patchwork::wrap_plots(
  plot_graph(trajectory) + ggtitle('Cell ordeing'),
  plot_graph(trajectory, color_cells = 'grouping', grouping = trajectory$grouping) + ggtitle('True grouping'),
  plot_graph(trajectory, color_cells = 'feature', feature_oi = gene_name) + ggtitle('Gene expression'),
  plot_graph(trajectory, color_cells = 'pseudotime', pseudotime = trajectory$pseudotime) + ggtitle('Pseudotime')
)
dev.off()

pdf('dendro.pdf', width = 16, height = 8, onefile = F)
patchwork::wrap_plots(
  plot_dendro(trajectory) + ggtitle('Cell ordeing'),
  plot_dendro(trajectory, color_cells = 'grouping', grouping = trajectory$grouping) + ggtitle('True grouping'),
  plot_dendro(trajectory, color_cells = 'feature', feature_oi = gene_name) + ggtitle('Gene expression'),
  plot_dendro(trajectory, color_cells = 'pseudotime', pseudotime = trajectory$pseudotime) + ggtitle('Pseudotime')
)
dev.off()

pdf('onedim.pdf', width = 16, height = 8, onefile = F)
patchwork::wrap_plots(
  plot_onedim(trajectory, label_milestones = T) + ggtitle('Cell ordeing'),
  plot_onedim(trajectory, label_milestones = T, color_cells = 'grouping', grouping = trajectory$grouping) + ggtitle('True grouping'),
  plot_onedim(trajectory, label_milestones = T, color_cells = 'feature', feature_oi = gene_name) + ggtitle('Gene expression'),
  plot_onedim(trajectory, label_milestones = T, color_cells = 'pseudotime', pseudotime = trajectory$pseudotime) + ggtitle('Pseudotime')
)
dev.off()

