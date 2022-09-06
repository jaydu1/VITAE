#' Run slingshot
#'
#' @param expression Expression matrix
#' @param parameters Parameters
#' @param priors Priors
#' @param verbose Verbosity level
#' @param seed Random seed
#'
#' @import dplyr
#' @import purrr
#' @import slingshot
#' @import dynwrap
#' @importFrom cluster pam
#' @importFrom irlba prcomp_irlba
#' @importFrom princurve project_to_curve
#' @importFrom stats lm
#' @importFrom utils head
#' @importFrom tibble deframe
#'
#' @export
run_fun <- function(expression, parameters, priors, verbose, seed) {
  start_id <- priors$start_id
  end_id <- priors$end_id
  dimred <- priors$dimred
  groups_id <- priors$groups_id

  #####################################
  ###        INFER TRAJECTORY       ###
  #####################################

  #   ____________________________________________________________________________
  #   Preprocessing                                                           ####

  start_cell <- if (!is.null(start_id)) { sample(start_id, 1) } else { NULL }

  # TIMING: done with preproc
  checkpoints <- list(method_afterpreproc = as.numeric(Sys.time()))

  #   ____________________________________________________________________________
  #   Dimensionality reduction                                                ####
  # only do dimred if it is not yet given by prior information
  if (is.null(dimred)) {
    ndim <- parameters$ndim
    if (ncol(expression) <= ndim) {
      message(paste0(
        "ndim is ", ndim, " but number of dimensions is ", ncol(expression),
        ". Won't do dimensionality reduction."
      ))
      rd <- as.matrix(expression)
    } else {
      pca <- irlba::prcomp_irlba(expression, n = ndim)

      # select optimal number of dimensions if ndim is large enough
      if (ndim > 3) {
        # this code is adapted from the expermclust() function in TSCAN
        # the only difference is in how PCA is performed
        # (they specify scale. = TRUE and we leave it as FALSE)
        x <- 1:ndim
        optpoint1 <- which.min(sapply(2:10, function(i) {
          x2 <- pmax(0, x - i)
          sum(lm(pca$sdev[1:ndim] ~ x + x2)$residuals^2 * rep(1:2,each = 10))
        }))

        # this is a simple method for finding the "elbow" of a curve, from
        # https://stackoverflow.com/questions/2018178/finding-the-best-trade-off-point-on-a-curve
        x <- cbind(1:ndim, pca$sdev[1:ndim])
        line <- x[c(1, nrow(x)),]
        proj <- princurve::project_to_curve(x, line)
        optpoint2 <- which.max(proj$dist_ind)-1

        # we will take more than 3 PCs only if both methods recommend it
        optpoint <- max(c(min(c(optpoint1, optpoint2)), 3))
      } else {
        optpoint <- ndim
      }

      rd <- pca$x[, seq_len(optpoint)]
      rownames(rd) <- rownames(expression)
    }
  } else {
    message("Using given dimred")
    rd <- dimred
  }


  #   ____________________________________________________________________________
  #   Clustering                                                              ####
  # only do clustering if it is not yet given by prior information
  if (is.null(groups_id)) {
    # max clusters equal to number of cells
    max_clusters <- min(nrow(expression)-1, 10)

    # select clustering
    if (parameters$cluster_method == "pam") {
      if (nrow(rd) > 10000) {
        warning("PAM (the default clustering method) does not scale well to a lot of cells. You might encounter memory issues. This can be resolved by using the CLARA clustering method, i.e. cluster_method = 'clara'.")
      }
      clusterings <- lapply(3:max_clusters, function(K){
        cluster::pam(rd, K) # we generally prefer PAM as a more robust alternative to k-means
      })
    } else if (parameters$cluster_method == "clara") {
      clusterings <- lapply(3:max_clusters, function(K){
        cluster::clara(rd, K) # we generally prefer PAM as a more robust alternative to k-means
      })
    }

    # take one more than the optimal number of clusters based on average silhouette width
    # (max of 10; the extra cluster improves flexibility when learning the topology,
    # silhouette width tends to pick too few clusters, otherwise)
    wh.cl <- which.max(sapply(clusterings, function(x){ x$silinfo$avg.width })) + 1
    labels <- clusterings[[min(c(wh.cl, 8))]]$clustering
  } else {
    message("Using given groups/clustering")
    labels <- groups_id %>% deframe()
  }

  start.clus <-
    if(!is.null(start_cell)) {
      labels[[start_cell]]
    } else {
      NULL
    }
  end.clus <-
    if(!is.null(end_id)) {
      unique(labels[end_id])
    } else {
      NULL
    }

  #   ____________________________________________________________________________
  #   Infer trajectory                                                        ####
  sds <- slingshot::slingshot(
    rd,
    labels,
    start.clus = start.clus,
    end.clus = end.clus,
    shrink = parameters$shrink,
    reweight = parameters$reweight,
    reassign = parameters$reassign,
    thresh = parameters$thresh,
    maxit = parameters$maxit,
    stretch = parameters$stretch,
    smoother = parameters$smoother,
    shrink.method = parameters$shrink.method
  )

  start_cell <- apply(slingshot::slingPseudotime(sds), 1, function(x) min(x, na.rm=T)) %>% sort() %>% head(1) %>% names()
  start.clus <- labels[[start_cell]]

  # TIMING: done with method
  checkpoints$method_aftermethod <- as.numeric(Sys.time())

  #   ____________________________________________________________________________
  #   Create output                                                           ####

  # satisfy r cmd check
  from <- to <- NULL

  # collect milestone network
  lineages <- slingLineages(sds)
  lineage_ctrl <- slingParams(sds)
  cluster_network <- lineages %>%
    map_df(~ tibble(from = .[-length(.)], to = .[-1])) %>%
    unique() %>%
    mutate(
      length = lineage_ctrl$dist[cbind(from, to)],
      directed = TRUE
    )

  # collect dimred
  dimred <- reducedDim(sds)

  # collect clusters
  cluster <- slingClusterLabels(sds)

  # collect progressions
  adj <- slingAdjacency(sds)
  lin_assign <- apply(slingCurveWeights(sds), 1, which.max)

  progressions <- map_df(seq_along(lineages), function(l) {
    ind <- lin_assign == l
    lin <- lineages[[l]]
    pst.full <- slingPseudotime(sds, na = FALSE)[,l]
    pst <- pst.full[ind]
    means <- sapply(lin, function(clID){
      stats::weighted.mean(pst.full, cluster[,clID])
    })
    non_ends <- means[-c(1,length(means))]
    edgeID.l <- as.numeric(cut(pst, breaks = c(-Inf, non_ends, Inf)))
    from.l <- lineages[[l]][edgeID.l]
    to.l <- lineages[[l]][edgeID.l + 1]
    m.from <- means[from.l]
    m.to <- means[to.l]

    pct <- (pst - m.from) / (m.to - m.from)
    pct[pct < 0] <- 0
    pct[pct > 1] <- 1

    tibble(cell_id = names(which(ind)), from = from.l, to = to.l, percentage = pct)
  })

  #   ____________________________________________________________________________
  #   Save output                                                             ####
  output <-
    dynwrap::wrap_data(
      cell_ids = rownames(expression)
    ) %>%
    dynwrap::add_trajectory(
      milestone_network = cluster_network,
      progressions = progressions
    ) %>%
    dynwrap::add_dimred(
      dimred = dimred
    ) %>%
    dynwrap::add_timings(checkpoints)
}

definition <- dynwrap::convert_definition(yaml::read_yaml(system.file("definition.yml", package = "tislingshot")))


#' Infer a trajectory using slingshot
#'
#' @eval dynwrap::generate_parameter_documentation(definition)
#'
#' @examples
#' dataset <- data <- dyntoy::generate_dataset(
#'   num_cells = 99,
#'   num_features = 101,
#'   model = "tree",
#'   normalise = FALSE
#' )
#' model <- dynwrap::infer_trajectory(dataset, ti_slingshot())
#'
#' @export
ti_slingshot <- dynwrap::create_ti_method_r(
  definition = definition,
  run_fun = run_fun,
  return_function = TRUE
)
