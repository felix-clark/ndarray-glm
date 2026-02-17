//! A rust library for performing GLM regression with data represented in
//! [`ndarray`](file:///home/felix/Projects/ndarray-glm/target/doc/ndarray/index.html)s.
//! The [`ndarray-linalg`](https://docs.rs/ndarray-linalg/) crate is used to allow
//! optimization of linear algebra operations with BLAS.
//!
//! This crate is in beta and the interface may change significantly. The tests include several
//! comparisons with R's `glmnet` package, but some cases may not be covered directly or involve
//! inherent ambiguities or imprecisions.
//!
//! # Feature summary:
//!
//! * Linear, logistic, Poisson, and binomial regression (more to come)
//! * Generic over floating-point type
//! * L1 (lasso), L2 (ridge), and elastic net regularization
//! * Statistical tests of fit result
//! * Alternative and custom link functions
//!
//!
//! # Setting up BLAS backend
//!
//! See the [backend features of
//! `ndarray-linalg`](https://github.com/rust-ndarray/ndarray-linalg#backend-features)
//! for a description of the available BLAS configuartions. You do not need to
//! include `ndarray-linalg` in your crate; simply provide the feature you need to
//! `ndarray-glm` and it will be forwarded to `ndarray-linalg`.
//!
//! Examples using OpenBLAS are shown here. In principle you should also be able to use
//! Netlib or Intel MKL, although these backends are untested.
//!
//! ## System OpenBLAS (recommended)
//!
//! Ensure that the development OpenBLAS library is installed on your system. In
//! Debian/Ubuntu, for instance, this means installing `libopenblas-dev`. Then, put the
//! following into your crate's `Cargo.toml`:
//! ```text
//! ndarray = { version = "0.17", features = ["blas"]}
//! ndarray-glm = { version = "0.0.14", features = ["openblas-system"] }
//! ```
//!
//! ## Compile OpenBLAS from source
//!
//! This option does not require OpenBLAS to be installed on your system, but the
//! initial compile time will be very long. Use the folling lines in your crate's
//! `Cargo.toml`.
//! ```text
//! ndarray = { version = "0.17", features = ["blas"]}
//! ndarray-glm = { version = "0.0.14", features = ["openblas-static"] }
//! ```
//!
//! # Examples:
//!
//! Basic linear regression:
//! ```
//! use ndarray_glm::{array, Linear, ModelBuilder};
//!
//! let data_y = array![0.3, 1.3, 0.7];
//! let data_x = array![[0.1, 0.2], [-0.4, 0.1], [0.2, 0.4]];
//! let model = ModelBuilder::<Linear>::data(&data_y, &data_x).build().unwrap();
//! let fit = model.fit().unwrap();
//! // The result is a flat array with the first term as the intercept.
//! println!("Fit result: {}", fit.result);
//! ```
//!
//! Data standardization and L2 regularization:
//! ```
//! use ndarray_glm::{array, Linear, ModelBuilder, utility::standardize};
//!
//! let data_y = array![0.3, 1.3, 0.7];
//! let data_x = array![[0.1, 0.2], [-0.4, 0.1], [0.2, 0.4]];
//! // The design matrix can optionally be standardized, where the mean of each independent
//! // variable is subtracted and each is then divided by the standard deviation of that variable.
//! let data_x = standardize(data_x);
//! let model = ModelBuilder::<Linear>::data(&data_y, &data_x).build().unwrap();
//! // L2 (ridge) regularization can be applied with l2_reg().
//! let fit = model.fit_options().l2_reg(1e-5).fit().unwrap();
//! println!("Fit result: {}", fit.result);
//! ```
//!
//! Logistic regression with a non-canonical link function. The fit options may need adjusting
//! as these are typically more difficult to converge:
//! ```
//! use ndarray_glm::{array, Logistic, logistic_link::Cloglog, ModelBuilder};
//!
//! let data_y = array![true, false, false, true, true];
//! let data_x = array![[0.5, 0.2], [0.1, 0.3], [0.2, 0.6], [0.6, 0.3], [0.4, 0.4]];
//! let model = ModelBuilder::<Logistic<Cloglog>>::data(&data_y, &data_x).build().unwrap();
//! let fit = model.fit_options().max_iter(64).l2_reg(1e-3).fit().unwrap();
//! println!("Fit result: {}", fit.result);
//! ```
//!
//! # Generalized linear models
//!
//! A generalized linear model (GLM) describes the expected value of a response
//! variable $`y`$ through a *link function* $`g`$ applied to a linear combination of
//! $`K-1`$ feature variables $`x_k`$ (plus an intercept term):
//!
//! ```math
//! g(\text{E}[y]) = \beta_0 + \beta_1 x_1 + \ldots + \beta_{K-1} x_{K-1}
//! ```
//!
//! The right-hand side is the *linear predictor* $`\omega = \mathbf{x}^\top \boldsymbol{\beta}`$.
//! With $`N`$ observations the data matrix $`\mathbf{X}`$ has a leading column of
//! ones (for the intercept) and the model relates the response vector
//! $`\mathbf{y}`$ to $`\mathbf{X}`$ and parameters $`\boldsymbol{\beta}`$.
//!
//! ## Exponential family
//!
//! GLMs assume the response follows a distribution from the exponential family.
//! In canonical form the density is
//!
//! ```math
//! f(y;\eta) = \exp\!\bigl[\eta\, y - A(\eta) + B(y)\bigr]
//! ```
//!
//! where $`\eta`$ is the *natural parameter*, $`A(\eta)`$ is the *log-partition
//! function*, and $`B(y)`$ depends only on the observation.
//! The expected value and variance of $`y`$ follow directly from $`A`$:
//!
//! ```math
//! \text{E}[y] = A'(\eta), \qquad \text{Var}[y] = \phi\, A''(\eta)
//! ```
//!
//! where $`\phi`$ is the *dispersion parameter* (fixed to 1 for logistic and
//! Poisson families, estimated for the linear/Gaussian family).
//!
//! ## Link and variance functions
//!
//! The *link function* $`g`$ maps the expected response $`\mu = \text{E}[y]`$ to
//! the linear predictor:
//!
//! ```math
//! g(\mu) = \omega, \qquad \mu = g^{-1}(\omega)
//! ```
//!
//! The *canonical link* is the one for which $`\eta = \omega`$, i.e. the natural
//! parameter equals the linear predictor. Non-canonical links are also supported
//! (see [`link`]).
//!
//! The *variance function* $`V(\mu)`$ characterizes how the variance of $`y`$
//! depends on the mean, independent of the choice of link:
//!
//! ```math
//! \text{Var}[y] = \phi\, V(\mu)
//! ```
//!
//! ## Supported families
//!
//! | Family | Canonical link | $`V(\mu)`$ | $`\phi`$ |
//! |--------|---------------|-----------|---------|
//! | [`Linear`] (Gaussian) | Identity | $`1`$ | estimated |
//! | [`Logistic`] (Bernoulli) | Logit | $`\mu(1-\mu)`$ | $`1`$ |
//! | [`Poisson`] | Log | $`\mu`$ | $`1`$ |
//! | [`Binomial`] (fixed $`n`$) | Logit | $`\mu(n-\mu)/n`$ | $`1`$ |
//!
//! ## Fitting via IRLS
//!
//! Parameters are estimated by maximum likelihood using iteratively reweighted
//! least squares (IRLS). Each step solves for the update
//! $`\Delta\boldsymbol{\beta}`$:
//!
//! ```math
//! -\mathbf{H}(\boldsymbol{\beta}) \cdot \Delta\boldsymbol{\beta} = \mathbf{J}(\boldsymbol{\beta})
//! ```
//!
//! where $`\mathbf{J}`$ and $`\mathbf{H}`$ are the gradient and Hessian of the
//! log-likelihood. With the canonical link and a diagonal variance matrix
//! $`\mathbf{S}`$ ($`S_{ii} = \text{Var}[y^{(i)} | \eta]`$) this simplifies to:
//!
//! ```math
//! (\mathbf{X}^\top \mathbf{S} \mathbf{X})\, \Delta\boldsymbol{\beta}
//!   = \mathbf{X}^\top \bigl[\mathbf{y} - g^{-1}(\mathbf{X}\boldsymbol{\beta})\bigr]
//! ```
//!
//! Observation weights $`\mathbf{W}`$ (inverse dispersion per observation)
//! generalize this to:
//!
//! ```math
//! (\mathbf{X}^\top \mathbf{W} \mathbf{S} \mathbf{X})\, \Delta\boldsymbol{\beta}
//!   = \mathbf{X}^\top \mathbf{W} \bigl[\mathbf{y} - g^{-1}(\mathbf{X}\boldsymbol{\beta})\bigr]
//! ```
//!
//! The iteration converges when the log-likelihood is concave, which is
//! guaranteed with the canonical link.
//!
//! ## Regularization
//!
//! Optional penalty terms discourage large parameter values:
//!
//! - **L2 (ridge):** adds $`\frac{\lambda_2}{2} \sum |\beta_k|^2`$ to the
//!   negative log-likelihood, equivalent to a Gaussian prior on the
//!   coefficients.
//! - **L1 (lasso):** adds $`\lambda_1 \sum |\beta_k|`$, implemented via ADMM,
//!   which drives small coefficients to exactly zero.
//! - **Elastic net:** combines both L1 and L2 penalties.
//!
//! In all cases the intercept ($`\beta_0`$) is excluded from the penalty.
//!
//! For a more complete mathematical reference, see the
//! [derivation notes](https://felix-clark.github.io/src/tex/glm-math/main.pdf).

#![doc(html_root_url = "https://docs.rs/crate/ndarray-glm")]
pub mod error;
mod fit;
mod glm;
mod irls;
pub mod link;
mod math;
pub mod model;
pub mod num;
mod regularization;
mod response;
pub mod utility;

// Import some common names into the top-level namespace
pub use {
    fit::Fit,
    model::ModelBuilder,
    response::logistic::link as logistic_link,
    response::{binomial::Binomial, linear::Linear, logistic::Logistic, poisson::Poisson},
};

// re-export common structs from ndarray
pub use ndarray::{Array1, Array2, ArrayView1, ArrayView2, array};
