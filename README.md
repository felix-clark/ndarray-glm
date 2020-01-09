# glm-regress

Rust library for solving linear, logistic, and generalized linear models through iteratively reweighted least squares

|build_status|_  <!-- |crates|_ -->

.. |build_status| image:: https://api.travis-ci.org/felix-clark/glm-regress.svg?branch=master
.. _build_status: https://travis-ci.org/felix-clark/glm-regress

<!-- .. |crates| image:: http://meritbadge.herokuapp.com/glm-regress -->
<!-- .. _crates: https://crates.io/crates/glm-regress -->

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
