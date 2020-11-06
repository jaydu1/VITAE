# Trajectory Inference for Single-Cell RNA Sequencing Data Using A Mixture Model with Variational Deep Autoencoder





## Dependency

We recommend to use `conda` or `miniconda` to manage Python environment. For Mac and Linux users, you can use the following codes to create a new environment with all required packages (without gpu).

```
>>> conda create --name scTrajVAE python=3.6 -y
>>> conda activate scTrajVAE
>>> conda install pandas jupyter umap-learn matplotlib numba seaborn scanpy==1.6.0 scikit-learn -y
>>> yes | pip3 install tensorflow==2.2 tensorflow-probability==0.10.1 louvain scikit-misc networkx python-igraph
```

To enable gpu acceleration, you will need to install `cuda` and `tensorflow-gpu`. Please refer the Tensorflow website for more information.


We use `pip` to install tensorflow2.2 since currently it is not available for Mac and Win users via conda. After installing all required packages, one can open the Jupyter Notebook via terminal:

```
>>> jupyter notebook
```


