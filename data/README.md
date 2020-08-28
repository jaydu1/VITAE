# Datasets

To access data more conviniently, we store the R data files into Python pickle files. The following pickle files contain a list of `x`, `grouping`, and `milestone_net`.

File Name | Number of Cells | Number of Genes | Number of Clusters | Type | Trajectory
---|---|---|---|---|---
b\_cd14\_cd56.pkl|21082| 32738 (18750 non zero)|3|UMI|separate
dentate-gyrus-neurogenesis\_hochgerner.pkl|3585|2182|5|UMI|line
neonatal-rib-cartilage_mca.pkl| 2221 | 2449 |5|UMI|bifurcate
aging-hsc-old\_kowalczyk.pkl|873| 2815 |3|non-UMI|line
fibroblast-reprogramming_treutlein.pkl| 355 | 3379 |7|non-UMI| bifurcate

To access files from Python, use to following codes:

```Python
import h5py
import numpy as np
import pandas as pd
file_name = 'planaria-muscle-differentiation_plass'
with h5py.File('data_h5/'+file_name+'.h5', 'r') as f:
    x = np.array(f['count'], dtype=np.float32)
    # strings are stored as 'btype', need to convert them to Unicode. 
    # 'U' stands for Unicode.
    grouping = np.array(f['grouping']).astype('U')
    gene_names = np.array(f['gene_names']) .astype('U')
    milestone_net = pd.DataFrame(
        np.array(np.array(list(f['milestone_network'])).tolist(), dtype='U'), 
        columns=['from','to']
    )
```