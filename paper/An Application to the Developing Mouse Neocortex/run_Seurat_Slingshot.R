library(hdf5r)
library(Matrix)
library(Seurat)
library(ggplot2)
library(cowplot)

file.h5 <- H5File$new('../../data/mouse_brain_merged.h5', mode = "r")
count <- as.matrix(file.h5[["count"]][,])
grouping <- as.matrix(file.h5[["grouping"]][])
cell_ids <- as.matrix(file.h5[["cell_ids"]][])
gene_names <- as.matrix(file.h5[["gene_names"]][])
rownames(count) <- gene_names
colnames(count) <- cell_ids
file.h5$close_all()

# ------------------------------------------------------------------------------
# Integrating
# ------------------------------------------------------------------------------
ptm <- proc.time()
data <- CreateSeuratObject(counts = count, project = "mouse_brain_merged")
groups <- c(rep("Xiaochang", 10261),  rep("Miller", 16651-10261))
names(groups) <- colnames(data)
data <- AddMetaData(object = data, metadata = groups, col.name = "group")
data.list <- SplitObject(data, split.by = "group")
for (i in 1:length(data.list)) {
    data.list[[i]] <- NormalizeData(data.list[[i]], verbose = FALSE)
    data.list[[i]] <- FindVariableFeatures(data.list[[i]], selection.method = "vst", 
                                               nfeatures = 2000, verbose = FALSE)
}
reference.list <- data.list[c("Xiaochang", "Miller")]
data.anchors <- FindIntegrationAnchors(object.list = reference.list, dims = 1:30)
data.integrated <- IntegrateData(anchorset = data.anchors, dims = 1:30)
proc.time() - ptm

# switch to integrated assay. The variable features of this assay are automatically
# set during IntegrateData
DefaultAssay(data.integrated) <- "integrated"
save(data.integrated, file="result/data.integrated.Robj")


# Run the standard workflow for visualization and clustering
# ------------------------------------------------------------------------------
# not adjust
# ------------------------------------------------------------------------------
data.integrated <- ScaleData(data.integrated, verbose = FALSE, assay="integrated")
features <- VariableFeatures(object = data.integrated, assay="integrated")
data.integrated <- RunPCA(data.integrated, 
                          features = VariableFeatures(object = data.integrated), 
                          npcs=64, assay="integrated", verbose = FALSE)
data.integrated <- RunUMAP(data.integrated, reduction = "pca", dims = 1:64, 
                            assay="integrated", verbose = FALSE)
data.integrated[['days']] <- substr(colnames(data.integrated), 2, 3)

write.csv(data.integrated[['umap']][[]], 'result/Seurat_integrated_unadjusted.csv')
save(data.integrated, file="result/data.integrated.unadjust.Robj")

# ------------------------------------------------------------------------------
# adjust cell-cycle covariates
# ------------------------------------------------------------------------------
load("data.integrated.Robj")
source("cc_scores.R")
ptm <- proc.time()
data.integrated <- cc_scores(data.integrated, ref, assay="integrated")
data.integrated <- ScaleData(data.integrated, vars.to.regress = c("S.Score", "G2M.Score"), assay="integrated")
proc.time() - ptm
data.integrated <- RunPCA(data.integrated, 
                          features = VariableFeatures(object = data.integrated), 
                          npcs=64, assay="integrated", verbose = FALSE)
data.integrated <- RunUMAP(data.integrated, reduction = "pca", dims = 1:64, 
                            assay="integrated", verbose = FALSE)

write.csv(data.integrated[['umap']][[]], 'result/Seurat_integrated_adjusted.csv')
save(data.integrated, file="result/data.integrated.adjusted.Robj")


data.integrated[['days']] <- substr(colnames(data.integrated), 2, 3)
data.integrated <- FindNeighbors(data.integrated, dims = 1:10,
                                 verbose = FALSE)
data.integrated <- FindClusters(data.integrated, 
                                random.seed = 0,
                                resolution = 0.5, 
                                verbose = FALSE)
write.csv(cbind(data.integrated[['umap']][[]], data.integrated@active.ident), "result/Seurat_clustering.csv")
save(data.integrated, file="result/data.integrated.clustering.Robj")


# ------------------------------------------------------------------------------
# slingshot
# ------------------------------------------------------------------------------
library(slingshot)
sim <- SingleCellExperiment(assays = list(counts = data.integrated@assays$integrated@scale.data), 
                            reducedDims = SimpleList(UMAP = data.matrix(data.integrated[['umap']][[]][,c(2,1)])))
colData(sim)$louvain <- as.character(data.integrated@active.ident)#as.character(grouping)#
ptm <- proc.time()
sim <- slingshot(sim, 
                 clusterLabels = 'louvain', reducedDim = 'UMAP',
                 start.clus = "5",
                 stretch = 0
                 # approx_points = 100
                 )
proc.time() - ptm


# ------------------------------------------------------------------------------
# use dyno to format the TI results from Slingshot
# ------------------------------------------------------------------------------
library(tidyverse)
library(dplyr)
library(dyno)
library(purrr)
labels <- data.frame(colData(sim)$louvain)
rownames(labels) <- colnames(data.integrated)
start_cell <- apply(slingshot::slingPseudotime(sim), 1, min) %>% sort() %>% head(1) %>% names()
start.clus <- labels[[start_cell]]
# satisfy r cmd check
from <- to <- NULL

# collect milestone network
lineages <- slingLineages(sim)
lineage_ctrl <- slingParams(sim)
cluster_network <- lineages %>%
    map_df(~ tibble(from = .[-length(.)], to = .[-1])) %>%
    unique() %>%
    mutate(
        length = lineage_ctrl$dist[cbind(from, to)],
        directed = TRUE
    )
cluster_network$length <- 1

# collect dimred
dimred <- reducedDim(sim)

# collect clusters
cluster <- slingClusterLabels(sim)

# collect progressions
adj <- slingAdjacency(sim)
lin_assign <- apply(slingCurveWeights(sim), 1, which.max)

progressions <- map_df(seq_along(lineages), function(l) {
    ind <- lin_assign == l
    lin <- lineages[[l]]
    pst.full <- slingPseudotime(sim, na = FALSE)[,l]
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
trajectory <-
    dynwrap::wrap_data(
        cell_ids = rownames(labels)
    ) %>%
    dynwrap::add_trajectory(
        milestone_network = cluster_network,
        progressions = progressions
    ) %>%
    dynwrap::add_dimred(
        dimred = dimred
    )
pseudotime <- calculate_pseudotime(trajectory)


write.csv(cluster_network, "result/Slingshot_cluster_network.csv")
