#!/usr/bin/env Rscript

requireNamespace("dyncli", quietly = TRUE)
task <- dyncli::main()

library(tislingshot, warn.conflicts = FALSE)

output <- tislingshot::run_fun(
  expression = task$expression,
  priors = task$priors,
  parameters = task$parameters,
  seed = task$seed,
  verbose = task$verbose
)

dyncli::write_output(output, task$output)
