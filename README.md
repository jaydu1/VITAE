[![Python](https://raw.githubusercontent.com/jaydu1/VITAE/master/docs/img/badge_python.svg)](https://www.python.org/)
[![PyPI](https://raw.githubusercontent.com/jaydu1/VITAE/master/docs/img/badge_pypi.svg)](https://pypi.org/project/pyvitae/)
[![PyPI](https://raw.githubusercontent.com/jaydu1/VITAE/master/docs/img/badge_docs.svg)](https://jaydu1.github.io/VITAE/)


# Model-based Trajectory Inference for Single-Cell RNA Sequencing Using Deep Learning with a Mixture Prior



We provide some example notebooks. You could start working with VITAE on [*tutorial\_dentate*](https://github.com/jaydu1/VITAE/blob/master/tutorials/tutorial_dentate.ipynb).

notebook | system | details | reference
---|---|---|---
[*tutorial\_dentate*](https://github.com/jaydu1/VITAE/blob/master/tutorials/tutorial_dentate.ipynb) | neurons | 3585 cells and 2182 genes, 10x Genomics | [Hochgerner *et al.* (2018)](https://www.nature.com/articles/s41593-017-0056-2)
[*tutorial\_mouse\_brain*](https://github.com/jaydu1/VITAE/blob/master/tutorials/tutorial_mouse_brain.ipynb) | neurons | 16651 cells and 14707 genes | [Yuzwa *et al.* (2017)](https://www.cell.com/cell-reports/comments/S2211-1247(17)31813-2), Ruan *et al.* (2020+)


[Documents](https://jaydu1.github.io/VITAE/) of the package is aviavable.

## Dependency

Our Python package is available on PyPI and the user can install the CPU version with the following command. To enable GPU for Tensorflow, one should install CUDA dependencies and `tensorflow-gpu` package. We also recommend to use `conda`, `miniconda` or `virtualenv` to manage Python environment and install the package in a new environment.

```
>> pip install pyvitae
```

After installing all required packages, one can open the Jupyter Notebook via terminal:

```
>>> jupyter notebook
```

The required tensorflow versions are: 

Package|Version
---|---
tensorflow|>=2.3.0
tensorflow_probability|>=0.11.0

## License
This project is licensed under the terms of the MIT license.