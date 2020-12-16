# Datasets


Our datasets contains both real and synthetic data (from [dyngen](https://github.com/dynverse/dyngen) and our model), with both UMI and non-UMI counts, and various trajectory topologies.

type|name|count type|topology|N|G|k|source
---|---|---|---|---|---|---|---
real | aging | non-UMI | linear | 873 | 2815 | 3 | [Kowalczyk, *et al* (2015)](https://doi.org/10.1101/gr.192237.115)
real | human\_embryos | non-UMI | linear | 1289 | 8772 | 5 | [Petropoulos *et al.* (2016)](https://doi.org/10.1016/j.cell.2016.03.023)
real | germline | non-UMI | bifurcation | 272 | 8772 | 7 | [Guo *et al.* (2015)](https://doi.org/10.1016/j.cell.2015.05.015)
real | fibroblast | non-UMI | bifurcation | 355 | 3379 | 7 | [Treutlein *et al.* (2016)](https://doi.org/10.1038/nature18323)
real | mesoderm | non-UMI | tree | 504 | 8772 | 9 | [Loh *et al.* (2016)](https://doi.org/10.1016/j.cell.2016.06.011)
real | cell\_cycle | non-UMI | cycle | 264 | 6812 | 3 | [Petropoulos *et al.* (2016)](https://doi.org/10.1016/j.cell.2016.03.023)
real | dentate | UMI | linear | 3585 | 2182 | 5 | [Hochgerner *et al.* (2018)](https://doi.org/10.1038/s41593-017-0056-2) 
real | planaria\_muscle | UMI | bifurcation | 2338 | 4210 | 3 | [Wolf *et al.* (2019)](https://doi.org/10.1186/s13059-019-1663-x)
real | planaria\_full | UMI | tree | 18837 | 4210 | 33 | [Wolf *et al.* (2019)](https://doi.org/10.1186/s13059-019-1663-x)
real | immune | UMI | disconnected | 21082 | 18750 | 3 | [zheng *et al.* (2017)](https://doi.org/10.1038/ncomms14049)
real | mouse_brain_merged | UMI | tree | 10261 <br> 6390 | 14707 | 15 | [Yuzwa *et al.* (2017)](https://doi.org/10.1016/j.celrep.2017.12.017),<br> Ruan *et al.* (2020+)
synthetic | linear\_1 | non-UMI | linear | 2000 | 991 | 4 | dyngen 
synthetic | linear\_2 | non-UMI | linear | 2000 | 999 | 4 | dyngen 
synthetic | linear\_3 | non-UMI | linear | 2000 | 1000 | 4 | dyngen 
synthetic | bifurcating\_1 | non-UMI | bifurcation |  2000 | 997 | 7 | dyngen 
synthetic | bifurcating\_2 | non-UMI | bifurcation | 2000 | 991 | 7 | dyngen 
synthetic | bifurcating\_3 | non-UMI | bifurcation | 2000 | 1000 | 7 | dyngen 
synthetic | trifurcating\_1 | non-UMI | multifurcating | 2000 | 969 | 10 | dyngen 
synthetic | trifurcating\_2 | non-UMI | multifurcating | 2000 | 995 | 10 | dyngen 
synthetic | converging\_1 | non-UMI | bifurcation | 2000 | 998 | 6 | dyngen 
synthetic | cycle\_1 | non-UMI | cycle | 2000 | 1000 | 3 | dyngen 
synthetic | cycle\_2 | non-UMI | cycle | 2000 | 999 | 3 | dyngen 
synthetic | linear | UMI | linear | 1900 | 1990 | 5 | our model 
synthetic | bifurcation | UMI | bifurcation | 2100 | 1996 | 5 | our model 
synthetic | multifurcating | UMI | multifurcating | 2700 | 2000 | 7 | our model 
synthetic | tree | UMI | tree | 2600 | 2000 | 7 | our model 

