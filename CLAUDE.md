# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Build (requires system OpenBLAS: libopenblas-dev on Debian, openblas on Arch)
cargo build --features openblas-system

# Run all tests
cargo test --features openblas-system

# Run a single test
cargo test --features openblas-system <test_name>

# Run tests in a specific file
cargo test --features openblas-system --test <file_name>

# Check without building (faster feedback)
cargo check --features openblas-system
```

A BLAS feature flag is always required. `openblas-system` is the standard choice for development.

## Architecture

This is a Rust library for Generalized Linear Model (GLM) regression using IRLS (Iteratively Reweighted Least Squares) on ndarray data structures. Generic over `f32`/`f64`.

### Core flow: `ModelBuilder` → `Model` → `Fit`

1. **`ModelBuilder::data(&y, &x)`** creates a `ModelBuilderData` builder where you configure weights, offsets, intercept, etc.
2. **`.build()`** validates data and produces a `Model<M, F>` containing a `Dataset<F>` (y, X, optional weights/offsets/freqs) and model config.
3. **`model.fit()`** or **`model.fit_options().l2_reg(...).fit()`** runs IRLS via `Glm::regression()` and returns a `Fit` object with results and diagnostic methods.

### Key traits

- **`Glm`** (`glm.rs`): Core trait defining a GLM family. Specifies `Link` type, dispersion behavior, variance function, log-partition, and likelihood. Provides `regression()` which drives IRLS. Implemented by `Linear`, `Logistic`, `Poisson`, `Binomial`.
- **`Link`** (`link.rs`): Maps between linear predictor and response mean (`func`/`func_inv`). Extended by `Transform` for non-canonical links (adjusts errors/variance in IRLS).
- **`Response`** (`response.rs`): Converts domain types (bool, usize, float) to float for IRLS.
- **`IrlsReg`** (`regularization.rs`): Regularization strategies (Null, Ridge, Lasso, ElasticNet) that modify the IRLS update step.

### Response families (`src/response/`)

Each family module (e.g. `logistic.rs`) defines a struct, implements `Glm` with its variance function and log-partition, defines its canonical link and any alternative links, and implements `Response` for its domain type.

### Weights

`Dataset` supports two kinds of weights:
- **`weights`** (variance/analytic weights): scale the variance of each observation
- **`freqs`** (frequency weights): integer-like counts, equivalent to duplicating rows

### `Fit` object (`fit.rs`)

Rich diagnostics: AIC/BIC, deviance, dispersion, residuals (response, Pearson, deviance, working, studentized), leverage/hat matrix, LOO influence, likelihood ratio/Wald/score tests, covariance. Fisher information and hat matrix are lazily computed and cached via `RefCell`.

## Math Reference

The mathematical derivations behind this implementation are documented in a PDF at:
https://felix-clark.github.io/src/tex/glm-math/main.pdf

It can be downloaded and read locally (e.g. `curl -sL -o /tmp/glm-math.pdf <url>` then use the Read tool on the PDF).

## Conventions

- Tests comparing against R's `glm()` output are in `tests/` with R scripts generating reference data in `tests/R/`.
- LOO (leave-one-out) with frequency weights: leave out the entire row (set freq to 0), not decrement weight by 1. This matches R's `glm` convention.
