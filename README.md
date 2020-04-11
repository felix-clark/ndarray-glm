# ndarray-glm

Rust library for solving linear, logistic, and generalized linear models through
iteratively reweighted least squares, using the `ndarray-linalg` module.

[![Crate](https://img.shields.io/crates/v/ndarray-glm.svg)](https://crates.io/crates/ndarray-glm)
[![Documentation](https://docs.rs/ndarray-glm/badge.svg)](https://docs.rs/ndarray-glm)
[![Build Status](https://travis-ci.org/felix-clark/ndarray-glm.png?branch=master)](https://travis-ci.org/felix-clark/ndarray-glm)

## Status

This package is in early alpha and the interface is likely to undergo many changes.

## Prerequisites
fortran and BLAS must be installed:
```
sudo apt update && sudo apt install gfortran libblas-dev
```

To use the OpenBLAS backend, install also `libopenblas-dev` and use this crate with the "openblas-src" feature.

## Example

``` rust
use ndarray::array;
use ndarray_glm::{linear::Linear, model::ModelBuilder, standardize::standardize};

// define some test data
let data_y = array![0.3, 1.3, 0.7];
let data_x = array![[0.1, 0.2], [-0.4, 0.1], [0.2, 0.4]];
// The design matrix can optionally be standardized, where the mean of each independent
// variable is subtracted and each is then divided by the standard deviation of that variable.
let data_x = standardize(data_x);
// The model is general over floating point type.
// If the second argument is left "_", it will be inferred if possible.
// L2 regularization can be applied with l2_reg().
let model = ModelBuilder::<Linear, f32>::new(&data_y, &data_x).l2_reg(1e-5).build()?;
let fit = model.fit()?;
println!("Fit result: {}", fit.result);
```

## Features

- [X] Linear regression
- [X] Logistic regression
- [X] Generalized linear model IRLS
- [X] Linear offsets
- [X] Allow non-float domain types
- [X] L2 (ridge) Regularization
- [ ] L1 (lasso) Regularization
- [X] Generic over floating point type
- [X] Poisson
- [ ] Exponential
- [ ] Gamma (which effectively reduces to exponential with an arbitrary
      dispersion parameter)
- [ ] Inverse Gaussian
- [ ] Other exponential family distributions
- [ ] Option for data standardization/normalization
- [ ] Weighted regressions
  - [ ] Weight the covariance matrix with point-by-point error bars
  - [ ] Allow for off-diagonal correlations between points
  - [ ] Fix likelihood functions
  - [ ] Check the tolerance conditions for termination
- [ ] Non-canonical link functions
- [ ] Goodness-of-fit tests
  - [ ] Log-likelihood difference from saturated model
  - [ ] Aikaike and Bayesian information criteria
  - [ ] generalized R^2?

### TODO

- [ ] Generalize GLM interface to allow multi-parameter fits like a gamma
      distribution.
- [ ] Exact Z-scores by re-minimizing after fixing each parameter to zero (?)
- [ ] Unit tests for correct convergence with linear offsets
- [ ] Calculate/estimate dispersion parameter from the data


## References

* [Author's notes](https://felix-clark.github.io/glm-math)
* https://www.stat.cmu.edu/~ryantibs/advmethods/notes/glm.pdf
* https://bwlewis.github.io/GLM/
* https://statmath.wu.ac.at/courses/heather_turner/glmCourse_001.pdf
* [Maalouf, M., & Siddiqi, M. (2014). Weighted logistic regression for large-scale imbalanced and rare events data. Knowledge-Based Systems, 59, 142–148.](https://doi.org/10.1016/j.knosys.2014.01.012)
* [Disperson parameter lecture](http://people.stat.sfu.ca/~raltman/stat402/402L25.pdf)
* [Convergence problems in GLMs](https://journal.r-project.org/archive/2011-2/RJournal_2011-2_Marschner.pdf)
