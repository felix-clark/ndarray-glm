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

At the moment the master branch of `ndarray-linalg` along with `blas-src = 0.6`
is required to pass CI, but this precludes publishing new versions of this
crate. For the time being you may see some version mismatches in the
instructions below as a result.

## Prerequisites

fortran and BLAS must be installed:
```
sudo apt update && sudo apt install gfortran libblas-dev
```

To use the OpenBLAS backend, install also `libopenblas-dev` and use this crate with the "openblas-src" feature.

## Example

To use in your crate, add the following to the `Cargo.toml`:

```
ndarray = { version = "0.13", features = ["blas"]}
blas-src = { version = "0.4", default-features = false, features = ["openblas"] }
ndarray-glm = { version = "0.0.4", features = ["openblas-static"] }
```

An example for linear regression is shown below.

``` rust
use ndarray::array;
use ndarray_glm::{linear::Linear, model::ModelBuilder, standardize::standardize};

// define some test data
let data_y = array![0.3, 1.3, 0.7];
let data_x = array![[0.1, 0.2], [-0.4, 0.1], [0.2, 0.4]];
// The design matrix can optionally be standardized, where the mean of each independent
// variable is subtracted and each is then divided by the standard deviation of that variable.
let data_x = standardize(data_x);
// The model is generic over floating point type for the independent data variables, and
// the type will be inferred from the type of the arrays passed to data().
// L2 (ridge) regularization can be applied with l2_reg().
let model = ModelBuilder::<Linear>::data(&data_y, &data_x).l2_reg(1e-5).build()?;
let fit = model.fit()?;
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
- [X] Allow non-float domain types
- [X] L2 (ridge) Regularization
- [ ] L1 (lasso) Regularization
- [X] Generic over floating point type
- [ ] Other exponential family distributions
  - [X] Poisson
  - [X] Binomial (nightly only)
  - [ ] Exponential
  - [ ] Gamma (which effectively reduces to exponential with an arbitrary
        dispersion parameter)
  - [ ] Inverse Gaussian
  - [ ] ...
- [X] Option for data standardization/normalization
- [ ] Weighted and correlated regressions
  - [ ] Weight the covariance matrix with point-by-point error bars
  - [ ] Allow for off-diagonal correlations between points
  - [ ] Fix likelihood functions for weighted and/or correlated case
  - [ ] Re-visit the tolerance conditions for termination in these instances.
- [X] Non-canonical link functions
- [ ] Goodness-of-fit tests
  - [X] Likelihood ratio test
  - [X] Score test
  - [X] Wald test
  - [X] Log-likelihood difference from saturated model (deviance analysis)
  - [X] Akaike and Bayesian information criteria
  - [ ] generalized R^2?

### TODO

- [ ] Generalize GLM interface to allow multi-parameter fits like a gamma
      distribution or Gaussian with variable variance. This would demand other
      sufficient statistics besides y (e.g. y^2 for Gaussian w/ variance, log(y)
      for gamma). It might be worth putting off until const generics.
- [ ] More rigorous convergence tests and options for termination
- [ ] Logging system with configurable levels

## Reference

The author's [notes on generalized linear
models](https://felix-clark.github.io/glm-math) summarize many of the relevant
concepts and provide some additional references.
