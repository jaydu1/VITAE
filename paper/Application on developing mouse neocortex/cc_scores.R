#' cc_scores
#'
#' @description
#' calculate cell-cycle scores
#' reference:
#' https://satijalab.org/seurat/v3.1/cell_cycle_vignette.html 
#' https://hbctraining.github.io/scRNA-seq/lessons/cell_cycle_scoring.html
#'
#' @param dat: Seurat object with ccg regressed out
#' @param ref: reference file cell-cycle genes
#' @return Seurat object with 3 new meta.data columns added: S.Score, G2M.Score, Phase
#' 
#'
library(AnnotationHub)
library(dplyr)

cc_scores = function(dat, ref, assay="RNA"){
  dat@active.assay = assay
  # sum(is.na(dat2[['pdt']]$pdt)) # 451
  #dat[['nan_pdt']] = is.na(dat[['pdt']]$pdt)
  #dat = subset(dat, subset=nan_pdt==FALSE) # 9115 cells
  #dat[['SCT']] <- NULL
  
  # reference genes
  cell_cycle_genes = read.csv(ref, header = T)
  # Connect to AnnotationHub
  ah <- AnnotationHub()
  # Access the Ensembl database for organism
  ahDb <- query(ah, pattern = c("Mus musculus", "EnsDb"), ignore.case = TRUE)
  # Acquire the latest annotation files
  id <- ahDb %>% mcols() %>% rownames() %>% tail(n = 1)
  # Download the appropriate Ensembldb database
  edb <- ah[[id]]
  # Extract gene-level information from database
  annotations <- genes(edb, return.type = "data.frame")
  # Select annotations of interest
  annotations <- annotations %>% dplyr::select(gene_id, gene_name, seq_name, gene_biotype, description)
  # Get gene names for Ensembl IDs for each gene
  cell_cycle_markers <- dplyr::left_join(cell_cycle_genes, annotations, by = c("geneID" = "gene_id"))
  # Acquire the S and G2M phase genes
  s_genes <- cell_cycle_markers %>% dplyr::filter(phase == "S") %>% pull("gene_name")
  g2m_genes <- cell_cycle_markers %>% dplyr::filter(phase == "G2/M") %>% pull("gene_name")
  
  # Perform cell cycle scoring
  dat <- CellCycleScoring(dat, g2m.features = g2m_genes, s.features = s_genes)
  # dat[[]] %>% dplyr::select(S.Score, G2M.Score, Phase)
  return (dat)
}

ref <- "https://raw.githubusercontent.com/hbc/tinyatlas/master/cell_cycle/Mus_musculus.csv"

# data <- cc_scores(data, ref)
# score <- cbind(data[["S.Score"]], data[["G2M.Score"]])
# write.csv(x=score, file="score_Miller.csv")