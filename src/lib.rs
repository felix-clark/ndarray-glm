//! A rust library for performing GLM regression with data represented in
//! [`ndarray`](file:///home/felix/Projects/ndarray-glm/target/doc/ndarray/index.html)s.
//! The [`ndarray-linalg`](https://docs.rs/ndarray-linalg/) crate is used to allow
//! optimization of linear algebra operations with BLAS.
//!
//! This crate is early alpha and may change rapidly. No guarantees can be made about
//! the accuracy of the fits.
//!
//! # Feature summary:
//!
//! * Linear, logistic, Poisson, and binomial regression (more to come)
//! * Generic over floating-point type
//! * L2 (ridge) regularization
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
//! ndarray = { version = "0.15", features = ["blas"]}
//! ndarray-glm = { version = "0.0.13", features = ["openblas-system"] }
//! ```
//!
//! ## Compile OpenBLAS from source
//!
//! This option does not require OpenBLAS to be installed on your system, but the
//! initial compile time will be very long. Use the folling lines in your crate's
//! `Cargo.toml`.
//! ```text
//! ndarray = { version = "0.15", features = ["blas"]}
//! ndarray-glm = { version = "0.0.13", features = ["openblas-static"] }
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
//! Logistic regression with a non-canonical link function (fit may need adjusting as these are
//! typically more difficult):
//! ```
//! use ndarray_glm::{array, Logistic, logistic_link::Cloglog, ModelBuilder};
//!
//! let data_y = array![true, false, false, true, true];
//! let data_x = array![[0.5, 0.2], [0.1, 0.3], [0.2, 0.6], [0.6, 0.3], [0.4, 0.4]];
//! let model = ModelBuilder::<Logistic<Cloglog>>::data(&data_y, &data_x).build().unwrap();
//! let fit = model.fit_options().l2_reg(1e-5).fit().unwrap();
//! println!("Fit result: {}", fit.result);
//! ```

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
pub use ndarray::{array, Array1, Array2, ArrayView1, ArrayView2};
