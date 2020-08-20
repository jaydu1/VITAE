project_waypoints <- function(
  trajectory,
  cell_positions,
  edge_positions = NULL,
  waypoints = select_waypoints(trajectory),
  trajectory_projection_sd = sum(trajectory$milestone_network$length) * 0.05,
  color_trajectory = "none"
) {
  waypoints$waypoint_network <- waypoints$waypoint_network %>%
    rename(
      milestone_id_from = from_milestone_id,
      milestone_id_to = to_milestone_id
    )
  
  testthat::expect_setequal(cell_positions$cell_id, colnames(waypoints$geodesic_distances))
  
  # project waypoints to dimensionality reduction using kernel and geodesic distances
  # rate <- 5
  # trajectory_projection_sd <- sum(trajectory$milestone_network$length) * 0.05
  # dist_cutoff <- sum(milestone_network$length) * 0.05
  # k <- 3
  # weight_cutoff <- 0.01
  
  # weights <- waypoints$geodesic_distances %>% stats::dexp(rate = 5)
  weights <- waypoints$geodesic_distances %>% stats::dnorm(sd = trajectory_projection_sd)
  testthat::expect_true(all(!is.na(weights)))
  # weights <- waypoints$geodesic_distances < dist_cutoff
  # weights[weights < weight_cutoff] <- 0
  
  weights <- weights / rowSums(weights)
  positions <- cell_positions %>%
    select(cell_id, comp_1, comp_2) %>%
    slice(match(colnames(weights), cell_id)) %>%
    column_to_rownames("cell_id") %>%
    as.matrix()
  
  # make sure weights and positions have the same cell_ids in the same order
  testthat::expect_equal(colnames(weights), rownames(positions))
  
  # calculate positions
  matrix_to_tibble <- function(x, rownames_column) {y <- as_tibble(x);y[[rownames_column]] <- rownames(x);y}
  if (!is.null(edge_positions)) {
    comp_names <- colnames(edge_positions) %>% keep(~grepl("comp_", .))
    waypoint_positions <-
      waypoints$progressions %>%
      group_by(from, to) %>%
      do({
        df <- .
        rel_edge_pos <- edge_positions %>% filter(from == df$from[[1]], to == df$to[[1]])
        for (cn in comp_names) {
          df[[cn]] <- approx(rel_edge_pos$percentage, rel_edge_pos[[cn]], df$percentage)$y
        }
        df
      }) %>%
      ungroup() %>%
      select(!!comp_names, waypoint_id) %>%
      left_join(waypoints$waypoints, "waypoint_id")
  } else {
    waypoint_positions <- (weights %*% positions) %>%
      matrix_to_tibble("waypoint_id") %>%
      left_join(waypoints$waypoints, "waypoint_id")
  }
  
  
  # add color of closest cell
  if (color_trajectory == "nearest") {
    testthat::expect_true("color" %in% colnames(cell_positions))
    
    waypoint_positions <- waypoint_positions %>%
      mutate(closest_cell_ix = (weights %>% apply(1, which.max))[waypoint_id]) %>%
      mutate(closest_cell_id = colnames(weights)[closest_cell_ix]) %>%
      mutate(color = (cell_positions %>% select(cell_id, color) %>% deframe())[closest_cell_id])
  }
  
  # positions of different edges
  waypoint_edges <- waypoints$waypoint_network %>%
    left_join(waypoint_positions %>% rename_all(~paste0(., "_from")) %>% select(-milestone_id_from), c("from" = "waypoint_id_from")) %>%
    left_join(waypoint_positions %>% rename_all(~paste0(., "_to")) %>% select(-milestone_id_to), c("to" = "waypoint_id_to")) %>%
    mutate(length = sqrt((comp_1_to - comp_1_from)**2 + (comp_2_to - comp_2_from)**2))
  
  # add arrows to every milestone to milestone edge
  # an arrow is placed at the waypoint which is in the middle from the start and end
  waypoint_edges <- waypoint_edges %>%
    group_by(milestone_id_from, milestone_id_to) %>%
    mutate(
      distance_to_center = (comp_1_to - mean(c(max(comp_1_from), min(comp_1_from))))^2 + (comp_2_to - mean(c(max(comp_2_from), min(comp_2_from))))^2,
      arrow = row_number() == which.min(distance_to_center)
    ) %>%
    ungroup()
  
  lst(
    positions = waypoint_positions,
    edges = waypoint_edges
  )
}



select_waypoints <- function(
  trajectory,
  n_waypoints = 200,
  trafo = sqrt,
  resolution = sum(trafo(trajectory$milestone_network$length))/n_waypoints,
  recompute = FALSE
) {
  
  # create milestone waypoints
  waypoint_milestone_percentages_milestones <- tibble(
    milestone_id = trajectory$milestone_ids,
    waypoint_id = paste0("W", milestone_id),
    percentage = 1
  )
  
  # create uniform progressions
  # waypoints which lie on a milestone will get a special name, so that they are the same between milestone network edges
  waypoint_progressions <- trajectory$milestone_network %>%
    mutate(percentage = map(trafo(length), ~c(seq(0, ., min(resolution, .))/., 1))) %>%
    select(-length, -directed) %>%
    unnest(percentage) %>%
    group_by(from, to, percentage) %>% # remove duplicate waypoints
    filter(row_number() == 1) %>%
    ungroup() %>%
    mutate(waypoint_id = case_when(
      percentage == 0 ~ paste0("MILESTONE_W", from),
      percentage == 1 ~ paste0("MILESTONE_W", to),
      TRUE ~ paste0("W", row_number())
    )
    )
  
  # create waypoint percentages from progressions
  waypoint_milestone_percentages <- waypoint_progressions %>%
    group_by(waypoint_id) %>%
    filter(row_number() == 1) %>%
    ungroup() %>%
    rename(cell_id = waypoint_id) %>%
    convert_progressions_to_milestone_percentages(
      "this argument is unnecessary, I can put everything I want in here!",
      trajectory$milestone_ids,
      trajectory$milestone_network,
      .
    ) %>%
    rename(waypoint_id = cell_id)
  
  # calculate distance
  waypoint_geodesic_distances <- calculate_geodesic_distances(
    trajectory,
    waypoint_milestone_percentages = waypoint_milestone_percentages
  )[waypoint_progressions$waypoint_id, ]
  
  # also create network between waypoints
  waypoint_network <- waypoint_progressions %>%
    group_by(from, to) %>%
    mutate(from_waypoint = waypoint_id, to_waypoint = lead(waypoint_id, 1)) %>%
    drop_na() %>%
    ungroup() %>%
    select(from = from_waypoint, to = to_waypoint, from_milestone_id = from, to_milestone_id = to)
  
  # create waypoints and their properties
  waypoints <- waypoint_milestone_percentages %>%
    group_by(waypoint_id) %>%
    arrange(-percentage) %>%
    filter(row_number() == 1) %>%
    ungroup() %>%
    mutate(milestone_id = ifelse(percentage == 1, milestone_id, NA)) %>%
    select(-percentage)
  
  lst(
    milestone_percentages = waypoint_milestone_percentages,
    progressions = waypoint_progressions,
    geodesic_distances = waypoint_geodesic_distances,
    waypoint_network,
    waypoints
  )
}


