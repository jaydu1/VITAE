# Trajectory Inference for Single-Cell RNA Sequencing Data Using A Mixture Model with Variational Deep Autoencoder





## Dependency

We recommend to use `conda` or `miniconda` to manage Python environment. For Mac and Linux users, you can use the following codes to create a new environment with all required packages (without gpu).

```
>>> conda create --name scTrajVAE python=3.6 -y
>>> conda activate scTrajVAE
>>> conda install -c conda-forge pandas jupyter umap-learn matplotlib numba seaborn scikit-learn -y
(optional) >>> conda install -c bioconda scanpy==1.6.0 -y
>>> yes | pip3 install tensorflow==2.3 tensorflow-probability==0.11 louvain scikit-misc networkx python-igraph
```

To enable gpu acceleration, you will need to install `cuda` and `tensorflow-gpu`. Please refer the Tensorflow website for more information.


We use `pip` to install tensorflow2.3 since currently it is not available for Mac and Win users via conda. After installing all required packages, one can open the Jupyter Notebook via terminal:

```
>>> jupyter notebook
```


