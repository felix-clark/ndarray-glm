//! library for solving GLM regression
//! TODO: documentation

// enable const_generics if the binomial feature is used.
#![cfg_attr(feature = "binomial", feature(const_generics))]

// opt-in for binomial regression as it utilizes unstable features.
#[cfg(feature = "binomial")]
pub mod binomial;
pub mod error;
mod fit;
mod glm;
pub mod linear;
pub mod link;
pub mod logistic;
mod math;
pub mod model;
pub mod poisson;
mod regularization;
pub mod standardize;
mod utility;

// Import some common names into the top-level namespace
#[cfg(feature = "binomial")]
pub use binomial::Binomial;
pub use {
    linear::Linear, logistic::Logistic, model::ModelBuilder, poisson::Poisson,
    standardize::standardize,
};
