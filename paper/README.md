

# Data

Information of our data is described in the Method section of the manuscript and the supplementary materials. All the data are also publicly available from [https://github.com/jaydu1/VITAE/tree/master/data](https://github.com/jaydu1/VITAE/tree/master/data).

# Code

## Dependency

Our Python package is available on PyPI and the user can install the CPU version with the following command. 

```
>> pip install pyvitae==1.1.4
```
For reproducing the results, please use version v1.1.4.

To enable GPU for Tensorflow, one should install CUDA dependencies and `tensorflow-gpu` package. We also recommend to use `conda`, `miniconda` or `virtualenv` to manage Python environment and install the package in a new environment.


One can also install all dependencies by hand and use the source code in the github repo.

package|version
---|---
Python|>=3.7.0
tensorflow| >=2.3.0 
tensorflow-probability| >=0.11.0
pandas| >=1.1.5
jupyter| >=1.0.0
umap-learn| >=0.4.6
matplotlib |>=3.3.3 
numba| >=0.52.0
seaborn |>=0.11.0
scikit-learn |>=0.23.2
scikit-misc| >=0.1.3
statsmodels | >= 0.12.1
louvain| >=0.7.0
networkx| >=2.5
scanpy| >=1.8.2

For reproducing the results in the manuscript, the following versions of the R language and packages are needed.

package|version
---|---
R|>=4.0.2
Seurat | >=3.2.2
slingshot | >=1.9.1
dyno | >=0.1.2
dplyr | >=1.0.2
tidyverse | >=1.3.0
purrr | >=0.3.4
AnnotationHub | >=2.20.2
Matrix | >=1.2-18
data.table | >=1.13.2
sandwich | >=3.0-0
lmtest | >=0.9-38
ggplot2 | >=3.3.2
ggpubr | >=0.4.0
cowplot | >=1.1.0
hdf5r | >=1.3.3

## Benchmarking with Real and Synthetic Datasets
The folder `Benchmarking with Real and Synthetic Datasets` contains code to reproduce results in Section 5 of the manuscript.

- `run_other_methods.R`: Run PAGA, monocle 3 and slingshot on all datasets. To run the new version of PAGA and monocle 3, one would need to create TI methods from docker with dyno (see [https://dynverse.org/developers/creating-ti-method/create_ti_method_container/](https://dynverse.org/developers/creating-ti-method/create_ti_method_container/)). The source code for them is in the folder `ti_methods`.
- `evaluate_other_methods.py`: Evaluate the trajectory inference results from other methods.
- `run_and_evaluate_VITAE_Gaussian.py`: Run VITAE using Gaussian likelihood with different random seeds and record the evaluation result.
- `run_and_evaluate_VITAE_NB.py`: Run VITAE using Negative Binomial likelihood with different random seeds and record the evaluation result.

The intermediate results of the above scripts are in the sub-folder `result`. Then the figures can be plotted with the following Jupyter notebook:

- `plot_benchmark.py`: Reproduce the Figure 2 in the manuscript.

## Application on developing mouse neocortex
The folder `Application on developing mouse neocortex` contains code to reproduce results on two developing mouse neocortex datasets of the manuscript.

- `cc_scores.R`: Functions to compute cell-cycle scores.
- `run_Seurat_Slingshot`: Run Seurat and Slingshot and record the results.
- `run_VITAE.py`: Run VITAE, and save the trained model and intermediate results.

The intermediate results of the above scripts are in the sub-folder `result`. Then the figures can be plotted with the following Jupyter notebook:

- `plot_mouse_brain.ipynb`: Reproduce the Figure 3 and S3 in the manuscript.


## Application on integrating mouse brain datasets


## Application on scRNA and scATAC datasets

The folder `Application on scRNA and scATAC datasets` contains code to reproduce results on multiomic trajectory inference of the manuscript.

- `run_VITAE.py`: Run VITAE, and save the trained model and intermediate results.

The figures can be plotted with the following Jupyter notebook:

- `plot_human_hematopoiesis.ipynb`: Reproduce the Figure 6 in the manuscript.
