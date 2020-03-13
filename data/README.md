# Datasets

To access data more conviniently, we store the R data files into Python pickle files. The following pickle files contain a list of `x`, `grouping`, and `milestone_net`.

File Name | Number of Cells | Number of Genes | Number of Clusters | Type | Trajectory
---|---|---|---|---|---
b\_cd14\_cd56.pkl|21082| 32738 (18750 non zero)|3|UMI|separate
dentate-gyrus-neurogenesis\_hochgerner.pkl|3585|2182|5|UMI|line
aging-hsc-old\_kowalczyk.pkl|873| 2815 |3|non-UMI|line
neonatal-rib-cartilage_mca.pkl| 2221 | 2449 |5|non-UMI|bifurcate
fibroblast-reprogramming_treutlein.pkl| 355 | 3379 |7|non-UMI| bifurcate

To load these data, try the following codes:

```Python
import pickle as pk
with open(path_to_file+'b_cd14_cd56.pkl', 'rb') as f:
	x, grouping, milestone_net = pk.load(f)
```