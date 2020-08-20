calculate_geodesic_distances <- function(
  trajectory,
  waypoint_cells = NULL,
  waypoint_milestone_percentages = NULL,
  directed = FALSE
) {

  if (is.null(waypoint_cells) && is_wrapper_with_waypoint_cells(trajectory)) {
    waypoint_cells <- trajectory$waypoint_cells
  }
  
  calculate_geodesic_distances_(
    cell_ids = trajectory$cell_ids,
    milestone_ids = trajectory$milestone_ids,
    milestone_network = trajectory$milestone_network,
    milestone_percentages = trajectory$milestone_percentages,
    divergence_regions = trajectory$divergence_regions,
    waypoint_cells = waypoint_cells,
    waypoint_milestone_percentages = waypoint_milestone_percentages,
    directed = directed
  )
}

calculate_geodesic_distances_ <- function(
  cell_ids,
  milestone_ids,
  milestone_network,
  milestone_percentages,
  divergence_regions,
  waypoint_cells = NULL,
  waypoint_milestone_percentages = NULL,
  directed = FALSE
) {
  cell_ids_trajectory <- unique(milestone_percentages$cell_id)
  
  # get waypoints and milestone percentages
  if (!is.null(waypoint_cells)) {
    waypoint_ids <- waypoint_cells
  } else if (is.null(waypoint_milestone_percentages)){
    waypoint_ids <- cell_ids_trajectory
  } else {
    waypoint_ids <- c()
  }
  
  if (!is.null(waypoint_milestone_percentages)) {
    waypoint_ids <- unique(c(waypoint_ids, waypoint_milestone_percentages$waypoint_id))
    milestone_percentages <- bind_rows(
      milestone_percentages,
      waypoint_milestone_percentages %>% rename(cell_id = waypoint_id)
    )
  }
  
  if (is.null(divergence_regions)) {
    divergence_regions <- tibble(divergence_id = character(0), milestone_id = character(0), is_start = logical(0))
  }
  
  # rename milestones to avoid name conflicts between cells and milestones
  milestone_trafo_fun <- function(x) paste0("MILESTONE_", x)
  milestone_network <- milestone_network %>% mutate(from = milestone_trafo_fun(from), to = milestone_trafo_fun(to))
  milestone_ids <- milestone_ids %>% milestone_trafo_fun()
  milestone_percentages <- milestone_percentages %>% mutate(milestone_id = milestone_trafo_fun(milestone_id))
  divergence_regions <- divergence_regions %>% mutate(milestone_id = milestone_trafo_fun(milestone_id))
  
  # add 'extra' divergences for transitions not in a divergence
  extra_divergences <-
    milestone_network %>%
    # filter(from != to) %>% # filter self edges
    rowwise() %>%
    mutate(in_divergence = divergence_regions %>% group_by(divergence_id) %>% summarise(match = all(c(from, to) %in% milestone_id)) %>% {any(.$match)}) %>%
    filter(!in_divergence) %>%
    do({tibble(divergence_id = paste0(.$from, "__", .$to), milestone_id = c(.$from, .$to), is_start = c(T, F))}) %>%
    ungroup() %>%
    distinct(divergence_id, milestone_id, .keep_all = TRUE)
  
  divergence_regions <- bind_rows(
    divergence_regions,
    extra_divergences
  )
  
  # extract divergence ids
  divergence_ids <- unique(divergence_regions$divergence_id)
  
  # construct igraph object of milestone network
  is_directed <- any(milestone_network$directed)
  mil_gr <- igraph::graph_from_data_frame(milestone_network, directed = is_directed, vertices = milestone_ids)
  
  # calculate cell-cell distances for pairs of cells that are in the same transition, i.e. an edge or a divergence region
  cell_in_tent_distances <-
    map_df(divergence_ids, function(did) {
      dir <- divergence_regions %>% filter(divergence_id == did)
      mid <- dir %>% filter(is_start) %>% .$milestone_id
      tent <- dir$milestone_id
      
      tent_nomid <- setdiff(tent, mid)
      tent_distances <- igraph::distances(mil_gr, v = mid, to = tent, mode = "out", weights = igraph::E(mil_gr)$length)
      
      relevant_pct <-
        milestone_percentages %>%
        group_by(cell_id) %>%
        filter(all(milestone_id %in% tent)) %>%
        ungroup()
      
      if (nrow(relevant_pct) <= 1) {
        return(NULL)
      }
      
      scaled_dists <-
        relevant_pct %>%
        mutate(dist = percentage * tent_distances[mid, milestone_id])
      
      pct_mat <-
        bind_rows(
          scaled_dists %>% select(from = cell_id, to = milestone_id, length = dist),
          tent_distances %>% as.data.frame() %>% gather(from, length) %>% mutate(to = from)
        ) %>%
        reshape2::acast(from ~ to, value.var = "length", fill = 0)
      
      wp_cells <- rownames(pct_mat)[rownames(pct_mat) %in% waypoint_ids]
      
      distances <- as.matrix(dynutils::calculate_distance(
        pct_mat,
        pct_mat[c(wp_cells, tent), , drop = FALSE],
        "manhattan"
      ))
      
      if (!isFALSE(directed)) {
        # calculate the sign of the distance
        # distance is negative if the cell is closer to the beginning than the waypoint
        begin <- dir %>% filter(is_start) %>% pull(milestone_id)
        
        signs <- sign(-outer(distances[, begin], distances[begin, ], "-"))
        signs[is.na(signs)] <- 1 # when disconnected, sign will be NaN, so that distance remains + Inf
        
        signs[signs == 0] <- 1
        distances <- distances * signs
      }
      
      distances <- distances %>%
        as.matrix() %>%
        reshape2::melt(varnames = c("from", "to"), value.name = "length") %>%
        mutate_at(c("from", "to"), as.character) %>%
        filter(from != to)
      
      distances
    })
  
  if (!isFALSE(directed)) {
    # switch from and to if distance is negative
    cell_in_tent_distances$from2 <- cell_in_tent_distances$from
    cell_in_tent_distances$from <- ifelse(
      cell_in_tent_distances$length < 0,
      cell_in_tent_distances$to,
      cell_in_tent_distances$from2
    )
    cell_in_tent_distances$to <- ifelse(
      cell_in_tent_distances$length < 0,
      cell_in_tent_distances$from2,
      cell_in_tent_distances$to
    )
    cell_in_tent_distances <- cell_in_tent_distances %>% select(-from2)
    cell_in_tent_distances$length <- abs(cell_in_tent_distances$length)
    
    # add reverse edges if distance approx. zero
    # this is necessary because the direction will be taken into account
    cell_in_tent_distances <- bind_rows(
      cell_in_tent_distances,
      cell_in_tent_distances %>%
        filter(length <= 1e-20) %>%
        mutate(from2 = from, from = to, to = from2) %>%
        select(-from2)
    )
  }
  
  # combine all networks into one graph
  gr <-
    bind_rows(milestone_network, cell_in_tent_distances) %>%
    group_by(from, to) %>%
    summarise(length = min(length)) %>%
    ungroup() %>%
    igraph::graph_from_data_frame(directed = directed, vertices = unique(c(milestone_ids, cell_ids_trajectory, waypoint_ids)))
  
  # compute cell-to-cell distances across entire graph
  mode <- dplyr::case_when(
    isTRUE(directed) || directed == "forward" ~ "out",
    directed == "reverse" ~ "in",
    TRUE ~ "all"
  )
  out <- gr %>%
    igraph::distances(
      v = waypoint_ids,
      to = cell_ids_trajectory,
      weights = igraph::E(gr)$length,
      algorithm = "dijkstra",
      mode = mode
    )
  
  # make matrix if only one waypoint
  if (length(waypoint_ids) == 1) {
    out <- matrix(out, nrow = 1, dimnames = list(waypoint_ids, cell_ids_trajectory))
  }
  
  # add distances of cells not within the milestone_percentages
  cell_ids_filtered <- setdiff(cell_ids, cell_ids_trajectory)
  if (length(cell_ids_filtered) > 0) {
    filtered_cell_distance <- sum(milestone_network$length)
    out <- cbind(
      out,
      matrix(
        rep(filtered_cell_distance, length(cell_ids_filtered) * nrow(out)),
        ncol = length(cell_ids_filtered),
        dimnames = list(rownames(out), cell_ids_filtered)
      )
    )
  }
  
  # put the cells in the right order
  out[waypoint_ids, cell_ids, drop = F]
}