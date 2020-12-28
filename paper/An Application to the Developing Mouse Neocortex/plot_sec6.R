library(data.table)
library(sandwich)
library(lmtest)
library(ggplot2)
library(ggpubr)
library(cowplot)
library(hdf5r)
plotGene <- function(gene.name) {
    temp.dat <- lapply(1:length(gene.name), function(i)
        temp.dat <- data.frame(rank(covariates$pdt), exp[gene.name[i], ], 
                               as.factor(covariates$id_data),
                               rep(gene.name[i], ncol(exp))))
    temp.dat <- do.call(rbind, temp.dat)
    colnames(temp.dat) <- c("Order", "expression", "Dataset", "Gene")
    temp.dat$Gene <- factor(temp.dat$Gene, levels=gene.name)
    p <- ggplot(data = temp.dat, aes(x = Order, y = expression, 
                                     col = Gene)
    ) + geom_point(alpha = 0.05) + 
        geom_smooth(method = "loess", se = F, aes(linetype = Dataset)) + theme_classic() +
        ylab("Gene expression") + xlab("Pseudotime order") 
    return(p)
}

cols <- sapply(list(c(0.3686274509803922, 0.30980392156862746, 0.6352941176470588, 1.0),
                    c(0.21607074202229917, 0.5556324490580546, 0.7319492502883507, 1.0),
                    c(0.45305651672433694, 0.7815455594002307, 0.6462898885044214, 1.0),
                    c(0.7477124183006537, 0.8980392156862746, 0.6274509803921569, 1.0),
                    c(0.9442522106881969, 0.9777008842752788, 0.6620530565167244, 1.0),
                    c(0.9977700884275279, 0.930872741253364, 0.6330642060745867, 1.0),
                    c(0.9934640522875817, 0.7477124183006536, 0.43529411764705883, 1.0),
                    c(0.9637831603229527, 0.47743175701653207, 0.28581314878892733, 1.0),
                    c(0.8472126105344099, 0.2612072279892349, 0.30519031141868513, 1.0)), 
               function(v) rgb(v[1], v[2], v[3], v[4]))

titles <- c("NEC -> RGC -> OPC", "RGC -> IPC -> Immature Neuron")
branches <- c('branch 5-7-0-11', 'bracnh 7-4-6-1')
p1 <- list()
p2 <- list()
for(i in 1:2){
    branch <- branches[i]
    
    file.h5 <- H5File$new(sprintf('result/%s/expression.h5', branch), mode = "r")
    exp <- as.matrix(file.h5[["expression"]][,])
    cell.names <- as.vector(file.h5[["cell_ids"]][])
    gene.names <- as.vector(file.h5[["gene_names"]][])
    colnames(exp) <- cell.names
    rownames(exp) <- gene.names
    file.h5$close_all()
    
    covariates <- fread(file.path('result',branch,"covariate.csv"))
    covariates$pdt <- fread(file.path('result',branch,"pseudotime.csv"))$pseudotime
    covariates$Day <- as.factor(fread(file.path('result',branch,"cell_day.csv"))$V2[-1])
    
    system.time(result <- sapply(1:nrow(exp), function(i) {
        if (var(exp[i, ]) == 0) { 
            r2 <- rep(NA, 3)
            r3 <- rep(NA, 3)
        } else {
            runlm <- lm(scale(exp[i, ]) ~ scale(rank(covariates$pdt)) + covariates$S_score + 
                            covariates$G2M_score + covariates$id_data)
            vcov <- vcovHC(runlm, type="HC3")
            ss <- coeftest(runlm, vcov = vcov)
            r2 <- ss[2, c(1, 3, 4)]
            r3 <- ss[5, c(1, 3, 4)]
        }
        if (i %% 500 == 0)
            print(i)
        return(c(r2, r3))
    }))
    
    
    result <- t(result)
    rownames(result) <- rownames(exp)
    colnames(result) <- c("pdt_est", "pdt_t", "pdt_pval",
                          "dataset_est", "dataset_t", "dataset_pval")
    result <- data.frame(result)
    sigma <- mad(result$pdt_t, na.rm = T)
    result$pdt_new_pval <- 2 * pnorm(-abs(result$pdt_t/sigma))
    result$new_adj_pval <- p.adjust(result$pdt_new_pval, "BH")
    
    result <- result[result$new_adj_pval< 0.05 & !is.na(result$new_adj_pval), ]
    
    result.pos <- result[order(result$pdt_est, decreasing = T), ]
    result.neg <- result[order(-result$pdt_est, decreasing = T), ]
    
    p_1 <- plotGene(c(rownames(result.pos)[1:2], rownames(result.neg)[1:2])) + 
        scale_color_brewer(palette = "Set2") + ggtitle(titles[i])
    temp.dat <- data.frame(Order = rank(covariates$pdt), Day = paste0(covariates$Day, ".5"), 
                           value = rep(1, nrow(covariates)))
    p_2 <- ggplot(temp.dat, aes(x = Order, y = value, fill= Day)) + geom_bar(stat = "identity") + 
        scale_fill_manual(values = cols) + 
        theme(legend.position = "bottom", axis.title = element_blank(),
              panel.grid = element_blank(),
              axis.text = element_blank(), axis.ticks = element_blank(),
              panel.background = element_blank() ) + 
        guides(fill = guide_legend(nrow = 1)) + ylim(0, 1)
    p1[[paste('b',i, sep='')]] <- p_1
    p2[[paste('b',i, sep='')]] <- p_2
}

p3 <- ggarrange(p2$b1, p2$b2, common.legend = TRUE, legend = "bottom")
p <- ggdraw() +
    draw_plot(p1$b1, x = 0, y = 0.15, width = .5, height = .85) +
    draw_plot(p1$b2, x = 0.5, y = 0.15, width = .5, height = .85) +
    draw_plot(p3, x = 0.034, y = 0, width = .95, height = .15)
ggsave(plot = p, filename = 'mainFigure6.pdf')

