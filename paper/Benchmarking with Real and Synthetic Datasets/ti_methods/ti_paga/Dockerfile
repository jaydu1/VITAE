FROM dynverse/dynwrap_latest:v0.1.0

ARG GITHUB_PAT

# igraph and louvain do not get installed by scanpy
RUN pip install python-igraph louvain

RUN pip install scanpy

# for theislab/anndata#159
RUN pip install scipy==v1.2.1

RUN pip install fa2

COPY definition.yml run.py example.sh /code/

ENTRYPOINT ["/code/run.py"]
