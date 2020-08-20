library(tidyverse)
library(dplyr)
library(dyno)
library(ggplot2)
source('calculate_geodesic_distances.R')
source('groupingColors.R')
source('waypoints.R')

# trajctory: a list containing following
# dimred: a matrix/data.frame with 2 colunms corresponding to 2 dimension reduction for visualization
# milestone_network: a tibble, (from,to,length,directed) e.g. ('M0','M1',0.61,TRUE)
# milestone_ids: a str vector, milestone names, e.g. c('M0)
# pseudotime: a vector
# cell_id: a vector e.g. "C1"
# grouping: a vector e.g. "M3" which milestone each cell belongs to, with names as cell_id
# milestone_percentages: a tibble (cell_id,milestone_id,percentage);
#                        one cell has two rows e.g.('C0','M2',0.221),('C0','M3',0.779);
#                        percentage is weight on each milestone
# gene_express: a vector of gene expression
# gene_name: str, names of the gene to plot
plott = function(trajectory, plotwhat){
  alpha_cells = 1
  size_cells = 2.5
  border_radius_percentage = .1
  size_trajectory = 1
  hex_cells = ifelse(length(trajectory$cell_ids) > 10000, 100, FALSE)
  
  ##
  cell_positions = data.frame(cell_id = trajectory$cell_id)
  cell_positions$comp_1 = trajectory$dimred[,1]
  cell_positions$comp_2 = trajectory$dimred[,2]
  # cell_positions$milestone_id = trajectory$milestone_ids
  if (plotwhat == 'pseudotime') {
    cell_positions$color = trajectory$pseudotime
    color_scale <- viridis::scale_color_viridis("pseudotime")
    fill_scale <- viridis::scale_fill_viridis("pseudotime")
  } else if (plotwhat == 'grouping') {
    cell_positions$color = trajectory$grouping
    groups <- check_groups(trajectory$grouping, NULL)
    color_scale <- scale_color_manual('grouping', values = set_names(groups$color, groups$group_id), guide = guide_legend(ncol = 5))
    fill_scale <- scale_fill_manual('grouping', values = set_names(groups$color, groups$group_id), guide = guide_legend(ncol = 5))
  } else if (plotwhat == 'expression') {
    cell_positions$color = trajectory$gene_express
    color_scale <- scale_color_distiller(paste0(trajectory$gene_name, " expression"), palette = "RdYlBu")
    fill_scale <- scale_fill_distiller(paste0(trajectory$gene_name, " expression"), palette = "RdYlBu")
  } else if (plotwhat == 'milestone') {
    milestones <- tibble(milestone_id = trajectory$milestone_ids)
    milestones <- milestones %>%
      mutate(color = milestone_palette('auto', n = n()))
    milestone_colors <- set_names(milestones$color, milestones$milestone_id) %>% col2rgb %>% t
    
    mix_colors <- function(milid, milpct) {
      color_rgb <- apply(milestone_colors[milid,,drop = FALSE], 2, function(x) sum(x * milpct))
      color_rgb[color_rgb < 0] <- 0
      color_rgb[color_rgb > 256] <- 256
      do.call(rgb, as.list(c(color_rgb, maxColorValue = 256)))
    }
    cell_colors <- trajectory$milestone_percentages %>%
      group_by(cell_id) %>%
      summarise(color = mix_colors(milestone_id, percentage))
    
    cell_positions = left_join(cell_positions, cell_colors, "cell_id")
    color_scale = scale_color_identity(NULL, guide = "none")
    fill_scale = scale_fill_identity(NULL, guide = "none")
  }
  
  ## plot 
  plot <- ggplot(cell_positions, aes(comp_1, comp_2)) +
    theme_graph() +
    theme(legend.position = "bottom", plot.title = element_text(hjust = 0.5))
  
  # add cells
  if (is.numeric(hex_cells)) {
    hex_coordinates <- calculate_hex_coords(cell_positions, hex_cells)
    
    plot +
      geom_polygon(
        aes(group = group, fill = color),
        data = hex_coordinates,
      ) +
      fill_scale
    
    plot <- plot +
      geom_polygon(
        aes(group = group, fill = color),
        data = hex_coordinates,
      ) +
      fill_scale
  } else {
    if (border_radius_percentage > 0) {
      plot <- plot +
        geom_point(size = size_cells, color = "black")
    }
    if (alpha_cells < 1) {
      plot <- plot +
        geom_point(size = size_cells * (1 - border_radius_percentage), color = "white")
    }
    plot <- plot +
      geom_point(aes(color = color), size = size_cells * (1 - border_radius_percentage), alpha = alpha_cells) +
      color_scale
  }
  
  # trajctory
  waypoint_projection <- project_waypoints(
    trajectory = trajectory,
    cell_positions = cell_positions,
    waypoints = select_waypoints(trajectory),
    trajectory_projection_sd = sum(trajectory$milestone_network$length) * 0.05,
    color_trajectory = "none",
    edge_positions = NULL
  )
  
  milestone_positions <-
    waypoint_projection$positions %>%
    filter(!is.na(milestone_id))
  
  # add arrow if directed
  arrow <-
    if (any(trajectory$milestone_network$directed)) {
      arrow(type = "closed", length = (unit(0.1, "inches")))
    } else {
      NULL
    }
  plot <- plot +
    geom_point(color = "#333333", data = milestone_positions, size = 4, alpha = 1)+
    geom_segment(
      aes(comp_1_from, comp_2_from, xend = comp_1_to, yend = comp_2_to),
      data = waypoint_projection$edges %>% filter(arrow),
      color = "#333333",
      arrow = arrow,
      size = 1,
      linejoin = "mitre",
      lineend = "butt"
    )

  plot <- plot +
    geom_segment(
      aes(comp_1_from, comp_2_from, xend = comp_1_to, yend = comp_2_to),
      data = waypoint_projection$edges,
      size = size_trajectory,
      color = "#333333"
    )
  plot
}

# plotwhat = 'grouping'
# plotwhat = 'pseudotime'
# plotwhat = 'expression'
# plotwhat = 'milestone'

source('IO.R')
pdf('combine.pdf', width = 16, height = 8, onefile = F)
patchwork::wrap_plots(
  plott(trajectory, 'milestone') + ggtitle('Cell ordeing'),
  plott(trajectory, 'grouping') + ggtitle('True grouping'),
  plott(trajectory, 'expression') + ggtitle('Gene expression'),
  plott(trajectory, 'pseudotime') + ggtitle('Pseudotime')
)
dev.off()



