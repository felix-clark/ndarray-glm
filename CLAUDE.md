# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Build (requires system OpenBLAS: libopenblas-dev on Debian, openblas on Arch)
cargo build --features openblas-system

# Run all tests (include stats to run p-value tests)
cargo test --features openblas-system,stats

# Run a single test
cargo test --features openblas-system,stats <test_name>

# Run tests in a specific file
cargo test --features openblas-system,stats --test <file_name>

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

## KaTeX Formulas in Documentation

Doc comments throughout `src/` use KaTeX for rendering math in rustdoc output. The rendering is powered by `katex-header.html` which is injected into rustdoc's HTML.

### Building docs with math rendering

```bash
RUSTDOCFLAGS="--html-in-header katex-header.html" cargo doc --no-deps --open
```

### Syntax

**Block math** (display mode) — use a fenced code block with language `math`:

````
```math
\text{AIC} = D + 2K
```
````

**Inline math** — wrap a backtick-quoted expression in `$...$`:

```
$`\boldsymbol{\beta}`$
```

This renders as an inline code element with class `language-inline-math`, which the KaTeX header script picks up.

### Common notation conventions

- Matrices/vectors: `\mathbf{X}`, `\boldsymbol{\beta}`, `\mathbf{W}`
- Transpose: `\mathsf{T}` (e.g. `\mathbf{X}^\mathsf{T}`)
- Text labels in formulas: `\text{AIC}`, `\text{Var}`
- Estimates: `\hat\phi`, `\hat\mu`
- Greek letters for model quantities: `\eta` (linear predictor), `\mu` (mean), `\omega` (weights)

### Files with math documentation

`src/lib.rs`, `src/fit.rs`, `src/glm.rs`, `src/link.rs`

## Conventions

- Tests comparing against R's `glm()` output are in `tests/` with R scripts generating reference data in `tests/R/`.
- LOO (leave-one-out) with frequency weights: leave out the entire row (set freq to 0), not decrement weight by 1. This matches R's `glm` convention.
