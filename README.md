[![Python](https://raw.githubusercontent.com/jaydu1/VITAE/master/docs/img/badge_python.svg)](https://www.python.org/)
[![PyPI](https://raw.githubusercontent.com/jaydu1/VITAE/master/docs/img/badge_pypi.svg)](https://pypi.org/project/pyvitae/)
[![docs](https://raw.githubusercontent.com/jaydu1/VITAE/master/docs/img/badge_docs.svg)](https://jaydu1.github.io/VITAE/)


# Joint Trajectory Inference for Single-cell Genomics Using Deep Learning with a Mixture Prior

This is a Python package, VITAE, to perform trajectory inference for single-cell RNA sequencing (scRNA-seq). VITAE is a probabilistic method combining a latent hierarchical mixture model with variational autoencoders to infer trajectories from posterior approximations. VITAE is computationally scalable and can adjust for confounding covariates to learn a shared trajectory from multiple datasets. VITAE also provides uncertainty quantification of the inferred trajectory and cell positions and can find differentially expressed genes along the trajectory. For more information, please check out our [manuscript on bioRXiv](https://www.biorxiv.org/content/10.1101/2020.12.26.424452v3). 

## Tutorials


We provide some example notebooks. You could start working with VITAE on [*tutorial\_dentate*](https://github.com/jaydu1/VITAE/blob/master/tutorials/tutorial_dentate.ipynb).

notebook | system | details | reference
---|---|---|---
[*tutorial\_dentate*](https://github.com/jaydu1/VITAE/blob/master/tutorials/tutorial_dentate.ipynb) | neurons | 3585 cells and 2182 genes, 10x Genomics | [Hochgerner *et al.* (2018)](https://doi.org/10.1038/s41593-017-0056-2)
[*tutorial\_mouse\_brain*](https://github.com/jaydu1/VITAE/blob/master/tutorials/tutorial_mouse_brain.ipynb) | neurons | 16651 cells and 14707 genes | [Yuzwa *et al.* (2017)](https://doi.org/10.1016/j.celrep.2017.12.017),<br> [Ruan *et al.* (2021)](https://doi.org/10.1073/pnas.2018866118)

In case GitHub rendering stops working, [NbViewer](https://nbviewer.jupyter.org/) is an alternative online tool to render Jupyter Notebooks.

[Datasets](https://github.com/jaydu1/VITAE/tree/master/data) and [Documents](https://jaydu1.github.io/VITAE/) are availble.

## Dependency

Our Python package is available on PyPI and the user can install the CPU version with the following command: 

```
>> pip install pyvitae
```
To enable GPU for TensorFlow, one should install CUDA dependencies and the `tensorflow-gpu` package. We also recommend using `conda`, `miniconda`, or `virtualenv` to manage the Python environment and install the package in a new environment.
After installing all required packages, one can open the Jupyter Notebook via the terminal:

```
>>> jupyter notebook
```

The required TensorFlow versions are:

Package|Version
---|---
tensorflow|>=2.3.0
tensorflow_probability|>=0.11.0

## License
This project is licensed under the terms of the MIT license.
