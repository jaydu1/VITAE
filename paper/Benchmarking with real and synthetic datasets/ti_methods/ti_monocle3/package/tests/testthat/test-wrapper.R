test_that("ti_monocle3 works", {
  dataset <- source(system.file("example.sh", package = "timonocle3"))$value

  model <- dynwrap::infer_trajectory(dataset, timonocle3::ti_monocle3(), verbose = TRUE)

  expect_is(model, "dynwrap::with_trajectory")
})
