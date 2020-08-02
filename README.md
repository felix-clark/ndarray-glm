# ndarray-glm

Rust library for solving linear, logistic, and generalized linear models through
iteratively reweighted least squares, using the
[`ndarray-linalg`](https://docs.rs/crate/ndarray-linalg/) module.

[![Crate](https://img.shields.io/crates/v/ndarray-glm.svg)](https://crates.io/crates/ndarray-glm)
[![Documentation](https://docs.rs/ndarray-glm/badge.svg)](https://docs.rs/ndarray-glm)
[![Build Status](https://travis-ci.org/felix-clark/ndarray-glm.png?branch=master)](https://travis-ci.org/felix-clark/ndarray-glm)

## Status

This package is in early alpha and the interface is likely to undergo many
changes. Functionality may change from one release to the next.

The regression algorithm uses iteratively re-weighted least squares (IRLS) with
a step-halving procedure applied when the next iteration of guesses does not
increase the likelihood.

Much of the logic is done at the type/trait level to avoid compiling code a user does
not need and to allow general implementations that the compiler can optimize in trivial
cases.

## Prerequisites

fortran and BLAS must be installed:
```
sudo apt update && sudo apt install gfortran libblas-dev
```

To use the OpenBLAS backend, install also `libopenblas-dev` and use this crate with the
"openblas-static" feature.

## Example

To use in your crate, add the following to the `Cargo.toml`:

```
ndarray = { version = "0.13", features = ["blas"]}
ndarray-glm = { version = "0.0.7", features = ["openblas-static"] }
```

An example for linear regression is shown below.

``` rust
use ndarray_glm::{array, Linear, ModelBuilder, standardize};

// define some test data
let data_y = array![0.3, 1.3, 0.7];
let data_x = array![[0.1, 0.2], [-0.4, 0.1], [0.2, 0.4]];
// The design matrix can optionally be standardized, where the mean of each independent
// variable is subtracted and each is then divided by the standard deviation of that variable.
let data_x = standardize(data_x);
// The interface takes `ArrayView`s to allow for efficient passing of slices.
let model = ModelBuilder::<Linear>::data(data_y.view(), data_x.view()).build()?;
// L2 (ridge) regularization can be applied with l2_reg().
let fit = model.fit_options().l2_reg(1e-5).fit()?;
// Currently the result is a simple array of the MLE estimators, including the intercept term.
println!("Fit result: {}", fit.result);
```

For logistic regression, the `y` array data must be boolean, and for Poisson
regression it must be an unsigned integer.

Custom non-canonical link functions can be defined by the user, although the
interface is not particularly ergonomic. See `tests/custom_link.rs` for examples.

## Features

- [X] Linear regression
- [X] Logistic regression
- [X] Generalized linear model IRLS
- [X] Linear offsets
- [X] Generic over floating point type
- [X] Non-float domain types
- [X] L2 (ridge) Regularization
- [ ] L1 (lasso) Regularization
  - An experimental smoothed version with an epsilon tolerance is WIP
- [ ] Other exponential family distributions
  - [X] Poisson
  - [X] Binomial (nightly only)
  - [ ] Exponential
  - [ ] Gamma
  - [ ] Inverse Gaussian
- [X] Option for data standardization/normalization
- [ ] Weighted and correlated regressions
- [X] Non-canonical link functions
- [X] Goodness-of-fit tests

## Reference

These [notes on generalized linear models](https://felix-clark.github.io/glm-math)
summarize many of the relevant concepts and provide some additional references.
