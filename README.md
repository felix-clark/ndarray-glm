# irls

Rust library for solving linear, logistic, and generalized linear models through iteratively reweighted least squares

## Status

This package is in early alpha and the interface is likely to undergo many changes.

### Prerequisites
fortran and BLAS must be installed:
```
sudo apt update && sudo apt install gfortran libblas-dev
```

### Features

- [X] Linear regression (exact)
- [X] Logistic regression IRLS
- [ ] Generalized linear model IRLS

### TODO

- [ ] Use trait to define class of generalized linear models
- [ ] Weighted regressions
- [ ] Generalize floating point type
