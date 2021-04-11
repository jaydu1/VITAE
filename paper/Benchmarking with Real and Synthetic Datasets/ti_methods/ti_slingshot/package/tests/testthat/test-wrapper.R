context("Testing ti_slingshot")

test_that("ti_slingshot produces a TI method", {
  method <- tislingshot::ti_slingshot()

  expect_true(dynwrap::is_ti_method(method))
})


dataset <- source(system.file("example.sh", package = "tislingshot"))$value
method <- tislingshot::ti_slingshot()

test_that("ti_slingshot is able to produce a trajectory", {
  model <- dynwrap::infer_trajectory(dataset, method, verbose = FALSE)

  expect_true(dynwrap::is_wrapper_with_trajectory(model))
  expect_true(dynwrap::is_wrapper_with_dimred(model))
})

test_that("ti_slingshot is able to produce a trajectory with priors", {
  dimred <- stats::prcomp(dataset$expression)$x[,1:3]
  colnames(dimred) <- paste0("comp_", seq_len(ncol(dimred)))
  dataset$prior_information$dimred <- dimred

  model <- dynwrap::infer_trajectory(dataset, method, give_priors = "groups_id", verbose = FALSE)
  expect_true(dynwrap::is_wrapper_with_trajectory(model))
  expect_true(dynwrap::is_wrapper_with_dimred(model))

  model <- dynwrap::infer_trajectory(dataset, method, give_priors = "dimred", verbose = FALSE)
  expect_true(dynwrap::is_wrapper_with_trajectory(model))
  expect_true(dynwrap::is_wrapper_with_dimred(model))
  expect_equal(model$dimred, dimred, tolerance = 0.01)

  model <- dynwrap::infer_trajectory(dataset, method, give_priors = c("dimred", "groups_id"), verbose = FALSE)
  expect_true(dynwrap::is_wrapper_with_trajectory(model))
  expect_true(dynwrap::is_wrapper_with_dimred(model))
  expect_equal(model$dimred, dimred, tolerance = 0.01)
})
