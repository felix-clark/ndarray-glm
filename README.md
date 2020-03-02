# glm-regress

Rust library for solving linear, logistic, and generalized linear models through iteratively reweighted least squares

<!-- [![Crate](https://img.shields.io/crates/v/glm-regress.svg)](https://crates.io/crates/glm-regress) -->
<!-- [![Documentation](https://docs.rs/glm-regress/badge.svg)](https://docs.rs/glm-regress) -->
[![Build Status](https://travis-ci.org/felix-clark/glm-regress.png?branch=master)](https://travis-ci.org/felix-clark/glm-regress)

## Status

This package is in early alpha and the interface is likely to undergo many changes.

### Prerequisites
fortran and BLAS must be installed:
```
sudo apt update && sudo apt install gfortran libblas-dev
```

To use the OpenBLAS backend, install also `libopenblas-dev` and use this crate with the "openblas-src" feature.

### Features

- [X] Linear regression
- [X] Logistic regression
- [X] Generalized linear model IRLS
- [X] L2 Regularization
- [X] Generic over floating point type
- [ ] Implement other models
  - [X] Poisson
  - [ ] Exponential
  - [ ] Gamma
  - [ ] Inverse Gaussian
  - [ ] ...

### TODO

- [X] Linear offsets
- [X] Allow non-float domain types (use mapping function from domain for floating-point type)
- [ ] Weighted regressions
  - [ ] Weight solve matrix
  - [ ] likelihood functions
  - [ ] tolerance for termination
- [ ] Other regularization options
  - [ ] Separate scaling for constant term
- [ ] Non-canonical link functions
- [ ] Unit tests for correct convergence with linear offsets

<!-- #### References: -->
<!-- * Maalouf, M., & Siddiqi, M. (2014). Weighted logistic regression for large-scale imbalanced and rare events data. Knowledge-Based Systems, 59, 142–148. doi:10.1016/j.knosys.2014.01.012 -->
<!-- * https://bwlewis.github.io/GLM/ -->
<!-- * https://journal.r-project.org/archive/2011-2/RJournal_2011-2_Marschner.pdf -->
