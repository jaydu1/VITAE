# Gaussian Mixture Variation Auto-Encoder for single-cell trajectory inference


```
┣━━━ sct_gmvae
┣		┣━━━ __init__.py
┣		┣━━━ scTGMVAE.py
┣		┣━━━ preprocess.py
┣		┣━━━ model.py
┣		┣━━━ train.py	
┣		┣━━━ inference.py
┣		┣━━━ metric.py
┣		┗━━━ util.py	
┗━━━ 
```


# Dependency

We recommend to use `conda` or `miniconda` to manage Python environment. For Mac and Linux users, you can use the following codes to create a new environment with all required packages (without gpu).

```
>>> conda create --name scTrajVAE python=3.6 -y
>>> conda activate scTrajVAE
>>> conda install pandas jupyter umap-learn matplotlib numba seaborn scikit-learn -y
>>> yes | pip3 install tensorflow==2.2 louvain localreg networkx python-igraph
```

To enable gpu acceleration, you will need to install `cuda` and `tensorflow-gpu`. Please refer the Tensorflow website for more information.


We use `pip` to install tensorflow2.2 since currently it is not available for Mac and Win users via conda. After installing all required packages, one can open the Jupyter Notebook via terminal:

```
>>> jupyter notebook
```


