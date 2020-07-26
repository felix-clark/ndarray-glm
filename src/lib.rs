//! A rust library for performing GLM regression with data represented in
//! [`ndarray`](file:///home/felix/Projects/ndarray-glm/target/doc/ndarray/index.html)s.
//! The [`ndarray-linalg`](https://docs.rs/ndarray-linalg/) crate is used to allow
//! optimization of linear algebra operations with BLAS.
//!
//! This crate is early alpha and may change rapidly. No guarantees can be made about
//! the accuracy of the fits.
//!
//! At the moment the docs and CI are blocked by an upstream issue. For more detail see the README.
//!
//! # Examples:
//! ```
//! use ndarray_glm::{array, Linear, ModelBuilder, standardize};
//!
//! let data_y = array![0.3, 1.3, 0.7];
//! let data_x = array![[0.1, 0.2], [-0.4, 0.1], [0.2, 0.4]];
//! // The design matrix can optionally be standardized, where the mean of each independent
//! // variable is subtracted and each is then divided by the standard deviation of that variable.
//! let data_x = standardize(data_x);
//! // The model is generic over floating point type for the independent data variables, and
//! // the type will be inferred from the type of the arrays passed to data().
//! // The interface takes `ArrayView`s to allow for efficient passing of slices.
//! // L2 (ridge) regularization can be applied with l2_reg().
//! let model = ModelBuilder::<Linear>::data(data_y.view(), data_x.view())
//!                 .l2_reg(1e-5).build().unwrap();
//! let fit = model.fit().unwrap();
//! // The result is a flat array with the first term as the intercept.
//! println!("Fit result: {}", fit.result);
//! ```
//!
//! The canonical link function is used by default. An alternative link function can be
//! specified as a type parameter to the response struct.
//! ```
//! use ndarray_glm::{array, Logistic, logistic_link::Cloglog, ModelBuilder};
//!
//! let data_y = array![true, false, false, true, true];
//! let data_x = array![[0.5, 0.2], [0.1, 0.3], [0.2, 0.6], [0.6, 0.3], [0.4, 0.4]];
//! let model = ModelBuilder::<Logistic<Cloglog>>::data(data_y.view(), data_x.view())
//!                 .l2_reg(1e-5).build().unwrap();
//! let fit = model.fit().unwrap();
//! println!("Fit result: {}", fit.result);
//! ```
//!
//! Feature summary:
//! * Generic over floating-point type
//! * Linear, logistic, Poisson, and binomial regression
//! * L2 (ridge) regularization
//! * Statistical tests of fit result
//! * Alternative and custom link functions
//!
//! Requirements:
//!   See the README for dependency requirements.

// enable const_generics if the binomial feature is used. This may be changed as the
// benefits of const generic here are not large.
#![cfg_attr(feature = "binomial", feature(const_generics))]
#[doc(html_root_url = "https://docs.rs/crate/ndarray-glm")]
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
mod standardize;
mod utility;

// Import some common names into the top-level namespace
#[cfg(feature = "binomial")]
pub use response::binomial::Binomial;
pub use {
    model::ModelBuilder,
    // re-export common structs from ndarray
    ndarray::{array, Array1, Array2, ArrayView1, ArrayView2},
    response::logistic::link as logistic_link,
    response::{linear::Linear, logistic::Logistic, poisson::Poisson},
    standardize::standardize,
};
