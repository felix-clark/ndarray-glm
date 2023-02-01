# ndarray-glm

Rust library for solving linear, logistic, and generalized linear models through
iteratively reweighted least squares, using the
[`ndarray-linalg`](https://docs.rs/crate/ndarray-linalg/) module.

[![Crate](https://img.shields.io/crates/v/ndarray-glm.svg)](https://crates.io/crates/ndarray-glm)
[![Documentation](https://docs.rs/ndarray-glm/badge.svg)](https://docs.rs/ndarray-glm)
[![Build Status](https://travis-ci.org/felix-clark/ndarray-glm.png?branch=master)](https://travis-ci.org/felix-clark/ndarray-glm)
![Downloads](https://img.shields.io/crates/d/ndarray-glm)

## Status

This package is in alpha and the interface could undergo changes. Even the
return value of certain functions may change from one release to the next.
Correctness is not guaranteed.

The regression algorithm uses iteratively re-weighted least squares (IRLS) with
a step-halving procedure applied when the next iteration of guesses does not
increase the likelihood.

Suggestions (via issues) and pull requests are welcome.

## Prerequisites

The recommended approach is to use a system BLAS implementation. For instance, to install
OpenBLAS on Debian/Ubuntu:
```
sudo apt update && sudo apt install -y libopenblas-dev
```
Then use this crate with the `openblas-system` feature.

To use an alternative backend or to build a static BLAS implementation, refer to the
`ndarray-linalg`
[documentation](https://github.com/rust-ndarray/ndarray-linalg#backend-features). Use
this crate with the appropriate feature flag and it will be forwarded to
`ndarray-linalg`.

## Example

To use in your crate, add the following to the `Cargo.toml`:

```
ndarray = { version = "0.15", features = ["blas"]}
ndarray-glm = { version = "0.0.12", features = ["openblas-system"] }
```

An example for linear regression is shown below.

``` rust
use ndarray_glm::{array, Linear, ModelBuilder, utility::standardize};

// define some test data
let data_y = array![0.3, 1.3, 0.7];
let data_x = array![[0.1, 0.2], [-0.4, 0.1], [0.2, 0.4]];
// The design matrix can optionally be standardized, where the mean of each independent
// variable is subtracted and each is then divided by the standard deviation of that variable.
let data_x = standardize(data_x);
let model = ModelBuilder::<Linear>::data(&data_y, &data_x).build()?;
// L2 (ridge) regularization can be applied with l2_reg().
let fit = model.fit_options().l2_reg(1e-5).fit()?;
// Currently the result is a simple array of the MLE estimators, including the intercept term.
println!("Fit result: {}", fit.result);
```

Custom non-canonical link functions can be defined by the user, although the
interface is currently not particularly ergonomic. See `tests/custom_link.rs`
for examples.

## Features

- [X] Linear regression
- [X] Logistic regression
- [X] Generalized linear model IRLS
- [X] Linear offsets
- [X] Generic over floating point type
- [X] Non-float domain types
- [ ] Regularization
  - [X] L2 (ridge)
  - [X] L1 (lasso)
  - [ ] Elastic Net
- [ ] Other exponential family distributions
  - [X] Poisson
  - [X] Binomial
  - [ ] Exponential
  - [ ] Gamma
  - [ ] Inverse Gaussian
- [X] Utility function for data standardization/normalization
- [ ] Weighted and correlated regressions
- [X] Non-canonical link functions
- [X] Goodness-of-fit tests

## References

* [notes on generalized linear models](https://felix-clark.github.io/glm-math)
* Generalized Linear Models and Extensions by Hardin & Hilbe
