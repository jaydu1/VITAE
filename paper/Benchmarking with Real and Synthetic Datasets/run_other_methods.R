library(dyno)
library(tidyverse)
library(Matrix)
library(hdf5r)
library(Seurat)

method_names <- c('paga','slingshot','monocle')
container_names <- c('jaydu/ti_paga_tree_2','dynverse/ti_slingshot:v2.0.1','ti_monocle3')

# path <- "../../data/*.h5"
# save_path <- "result/other methods/"
path <- '/Users/dujinhong/Downloads/VITAE/data/*.h5'
save_path <- '/Users/dujinhong/Downloads/result/'

for(load_file in Sys.glob(path)){  
    filename <- gsub(".h5", "", basename(load_file))
    if(grepl('mouse_brain',filename)){
        next
    }

    dir.create(file.path(save_path, filename))
    
    file.h5 <- H5File$new(load_file, mode="r")
    counts <- t(as.matrix(file.h5[['count']][,]))
    grouping <- data.frame(cell_id = paste0('c-',as.character(c(1:length(file.h5[['grouping']][])))), 
                           group_id = as.character(as.vector(file.h5[['grouping']][])))    
    milestone_net <- data.frame(file.h5[['milestone_network']][])
    root_milestone_id <- file.h5[['root_milestone_id']][]    
    if('w' %in% colnames(milestone_net)){
        cell_ids <- which(milestone_net$from==root_milestone_id)
        start_id <- paste0('c-',as.character(cell_ids[which.max(milestone_net[cell_ids,'w'])]))
    }else{
        start_id <- paste0('c-',sample(which(grouping$group_id==root_milestone_id), 1))
    }    
    file.h5$close_all()
    rownames(counts) <- paste0('c-',as.character(c(1:dim(counts)[1])))
    colnames(counts) <- paste0('g-',as.character(c(1:dim(counts)[2])))
    
    data <- CreateSeuratObject(counts = t(counts), project = "miller")
    data <- NormalizeData(data, normalization.method = "LogNormalize", scale.factor = 10000)
    data <- FindVariableFeatures(data, selection.method = "vst", nfeatures = 2000)
    data <- ScaleData(data, features = rownames(data))
  
    counts <- Matrix(counts[,VariableFeatures(data)], sparse = TRUE)
    expression <- Matrix(t(data[["RNA"]]@scale.data)[,VariableFeatures(data)], sparse = TRUE)

    dataset <- wrap_expression(
      expression = expression,
      counts = counts
    )

    for(i in c(1:3)){
        method_name <- method_names[i]                

        method <- create_ti_method_container(container_names[i])        
        if(i==2){
            dataset <- dataset %>% add_prior_information(start_id = c(start_id), groups_id = grouping)
        }else{
            dataset <- dataset %>% add_prior_information(start_id = start_id, groups_id = grouping)
        }
        model <- infer_trajectory(dataset, method(), give_priors = c("start_id","groups_id"), verbose = TRUE)
        
        
        ix <- wrapr::match_order(model$progressions$cell_id, grouping$cell_id)
        model$progressions <- model$progressions[ix,]
        rownames(model$progressions) <- NULL
        write.csv(x=model$progressions, file=file.path(save_path, filename, sprintf("%s_progressions.csv", method_name)))
        
        write.csv(x=model$milestone_network, file=file.path(save_path, filename, sprintf("%s_milestone_network.csv", method_name)))
    }
}