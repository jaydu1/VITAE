library(chromVAR) # BiocManager::install("chromVAR") DirichletMultinomial TFBSTools
library(SummarizedExperiment)

scpeaks <- readRDS('data_raw/scATAC-Healthy-Hematopoiesis-191120.rds')
scATAC <- readRDS('data_raw/scATAC-Cicero-GA-Hematopoiesis-191120.rds')
scRNA <- readRDS('data_raw/scRNA-Healthy-Hematopoiesis-191120.rds')

scchromVAR <- readRDS('data_raw/scATAC-chromVAR-Hematopoiesis-191120.rds')
dev <- deviations(scchromVAR)
rownames(dev) <- as.vector(rowData(scchromVAR)$name)

# remove unkown cell type
dev <- dev[,!endsWith(colData(scATAC)$BioClassification, 'Unk')]
scATAC <- scATAC[,!endsWith(colData(scATAC)$BioClassification, 'Unk')]
scRNA <- scRNA[,!endsWith(colData(scRNA)$BioClassification, 'Unk')]


# keep shared genes
gene_names <- unique(intersect(row.names(scATAC), row.names(scRNA)))
scATAC <- scATAC[match(gene_names, row.names(scATAC)), ]
scRNA <- scRNA[match(gene_names, row.names(scRNA)), ]


dev <- dev[,colSums(assays(scATAC)$gA)!=0]
scATAC <- scATAC[,colSums(assays(scATAC)$gA)!=0]
scRNA <- scRNA[,colSums(assays(scRNA)$counts)!=0]


cell_types_ATAC <- substr(colData(scATAC)$BioClassification, 4, 20)
cell_types_RNA <- substr(colData(scRNA)$BioClassification, 4, 20)

rename_ct <- function(ct){
    ct[startsWith(ct, 'CD4.N')] <- 'CD4.N'
    #ct[startsWith(ct, 'CD14.Mono')] <- 'CD14.Mono'
    ct[startsWith(ct, 'CLP')] <- 'CLP'
    return(ct)
}
cell_types_ATAC <- rename_ct(cell_types_ATAC)
cell_types_RNA <- rename_ct(cell_types_RNA)

uni_cell_types <- union(unique(cell_types_ATAC),unique(cell_types_RNA))


covariates_RNA <- scRNA$Group
covariates_ATAC <- scATAC$Group


library(Seurat)
library(hdf5r)
library(Matrix)

file.h5 <- H5File$new('human_hematopoiesis_scATAC.h5', mode = "w")
file.h5[["count"]] <- as.matrix(assays(scATAC)$gA)
# expression <- NormalizeData(as.matrix(assays(scATAC)$gA))
# file.h5[["expression"]] <- expression
file.h5[["grouping"]] <- as.vector(cell_types_ATAC)
file.h5[["cell_ids"]] <- colnames(scATAC)
file.h5[["gene_names"]] <- rownames(scATAC)
file.h5[["covariates"]] <- matrix(scATAC$Group, ncol=1)
file.h5$close_all()

file.h5 <- H5File$new('human_hematopoiesis_scRNA.h5', mode = "w")
file.h5[["count"]] <- as.matrix(assays(scRNA)$counts)
# expression <- NormalizeData(assays(scRNA)$counts)
# file.h5[["expression"]] <- as.matrix(expression)
file.h5[["grouping"]] <- as.vector(cell_types_RNA)
file.h5[["cell_ids"]] <- colnames(scRNA)
file.h5[["gene_names"]] <- rownames(scRNA)
file.h5[["covariates"]] <- matrix(scRNA$Group, ncol=1)
file.h5$close_all()


file.h5 <- H5File$new('human_hematopoiesis_motif.h5', mode = "w")
file.h5[["count"]] <- as.matrix(dev)
file.h5[["cell_ids"]] <- colnames(dev)
file.h5[["motif_names"]] <- rownames(dev)
file.h5$close_all()