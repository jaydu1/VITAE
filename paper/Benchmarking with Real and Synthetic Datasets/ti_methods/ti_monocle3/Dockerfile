FROM dynverse/dynwrap_latest:v0.1.0

ARG GITHUB_PAT

RUN apt-get update && apt-get install -y libudunits2-dev libgdal-dev libgeos-dev libproj-dev 

COPY definition.yml run.R example.sh package /code/

# temporary fix since Matrix.utils was removed from CRAN
RUN R -e 'devtools::install_github("rcannood/Matrix.utils")'

RUN R -e 'devtools::install("/code", dependencies = TRUE, quick = TRUE)'

ENTRYPOINT ["/code/run.R"]
