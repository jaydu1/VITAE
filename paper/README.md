

# Data

Information of our data is described in Section 5.1 and Section 6 of the manuscript. All the data are also publicly available from [https://github.com/jaydu1/VITAE/tree/master/data](https://github.com/jaydu1/VITAE/tree/master/data).

# Code

## Dependency

Our Python package is available on PyPI and the user can install the CPU version with the following command. 

```
>> pip install pyvitae
```

To enable GPU for Tensorflow, one should install CUDA dependencies and `tensorflow-gpu` package. We also recommend to use `conda`, `miniconda` or `virtualenv` to manage Python environment and install the package in a new environment.


One can also install all dependencies by hand and use the source code in the github repo.

package|version
---|---
Python|>=3.6.0
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
- `run_and_evaluate_VITAE.py`: Run VITAE with different random seeds and record the evaluation result.

The intermediate results of the above scripts are in the sub-folder `result`. Then the figures in Section 5 is given by the following Python script:

- `plot_sec5.py`: Reproduce the Figure 3 in the manuscript.

## An Application to the Developing Mouse Neocortex
The folder `An Application to the Developing Mouse Neocortex` contains code to reproduce results in Section 6 of the manuscript.

- `cc_scores.R`: Functions to compute cell-cycle scores.
- `run_Seurat_Slingshot`: Run Seurat and Slingshot and record the results.
- `run_VITAE.py`: Run VITAE, and save the trained model and intermediate results.

The intermediate results of the above scripts are in the sub-folder `result`. Then the figures in Section 5 is given by the following Python script:

- `plot_sec6.py`: Reproduce the Figure 4-5 and the subfigures of Figure 6 in the manuscript.
- `plot_sec6.R`: Reproduce the main figures of Figure 6 in the manuscript.

# Reproducibility Workflow

1. Install the required Python and R packages.
2. Clone the github repo [https://github.com/jaydu1/VITAE](https://github.com/jaydu1/VITAE) and use code in folder [paper](https://github.com/jaydu1/VITAE/tree/master/paper) for the following steps. 
3. Reproduce results in Section 5 of the manuscript. Note that we evaluate VITAE with 100 different random seeds on 25 datasets, so it will take much long time on a single desktop machine in serial. However, we have included intermediate results in the subfolder `Benchmarking with Real and Synthetic Datasets/result` and the readers can skip the first two steps and visualize Figure 3 in the manuscript directly.
	- (Optional) Evaluate VITAE by running `run_and_evaluate_VITAE.py`. 
	- (Optional) Evaluate other methods by running `run_other_methods.R` and `evaluate_other_methods.py`.
	- Visualize the results by running `plot_sec5.py`.
4. Reproduce results in Section 6 of the manuscript. 
	- (Optional) Infer trajectory with VITAE by running `run_VITAE.py`.
	- (Optional) Infer trajectory with Seurat+Slingshot by running `run_Seurat_Slingshot.R`.
	- Visualize the results by running `plot_sec6.py` and `plot_sec6.R`.