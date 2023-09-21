#!/usr/local/bin/Rscript

requireNamespace("dyncli", quietly = TRUE)
task <- dyncli::main()

library(timonocle3, warn.conflicts = FALSE)

output <- timonocle3::run_fun(
  expression = task$expression,
  priors = task$priors,
  parameters = task$parameters,
  seed = task$seed,
  verbose = task$verbose
)

dyncli::write_output(output, task$output)
