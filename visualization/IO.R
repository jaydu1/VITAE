cell_ids = read.csv('cell_ids.csv', header = F, stringsAsFactors = F)[,1]
grouping = read.csv('grouping.csv', header = F, stringsAsFactors = F)[,1]
if (is.numeric(grouping)) {
  grouping = paste0('M', grouping)
}
names(grouping) = cell_ids
dimred = read.csv('dimred.csv', header = F, sep = '')
milestone_network = as_tibble(read.csv('milestone_network.csv', stringsAsFactors = F,header = T))
milestone_network$directed = T
milestone_percentages = as_tibble(read.csv('milestone_percentages.csv', stringsAsFactors = F,header = T))
pseudotime = read.csv('pseudotime.csv',header = F)[,1]
gene_express = read.csv('gene_exp.csv', header = F)[,1]

trajectory = list()
trajectory$cell_ids = cell_ids
trajectory$dimred = dimred
trajectory$milestone_network = milestone_network
trajectory$milestone_percentages = milestone_percentages
trajectory$grouping = grouping
trajectory$milestone_ids = unique(milestone_percentages$milestone_id)
trajectory$pseudotime = pseudotime
trajectory$gene_express = gene_express
trajectory$gene_name = 'G252'