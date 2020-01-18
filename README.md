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

- [X] Linear regression (exact)
- [X] Logistic regression IRLS
- [X] Generalized linear model IRLS
- [ ] Implement other models
  - [ ] Poisson
  - [ ] ...

### TODO

- [X] Use trait to define class of generalized linear models
- [ ] Linear offsets
- [ ] Allow non-float domain types
- [ ] Weighted regressions
- [ ] Generalize floating point type
