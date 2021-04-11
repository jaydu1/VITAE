FROM dynverse/dynwrap_latest:v0.1.0

ARG GITHUB_PAT

RUN apt-get update && apt-get install -y libcgal-dev libglu1-mesa-dev libgsl-dev

COPY definition.yml run.R example.sh package/ /code/

RUN R -e 'devtools::install("/code/", dependencies = TRUE, quick = TRUE)'

ENTRYPOINT ["/code/run.R"]
