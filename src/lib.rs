//! library for solving GLM regression
//! TODO: documentation

// enable const_generics if the binomial feature is used.
#![cfg_attr(feature = "binomial", feature(const_generics))]

pub mod error;
mod fit;
mod glm;
mod irls;
pub mod link;
mod math;
pub mod model;
mod num;
mod regularization;
mod response;
mod standardize;
mod utility;

// Import some common names into the top-level namespace
#[cfg(feature = "binomial")]
pub use response::binomial::Binomial;
pub use {
    model::ModelBuilder,
    response::{
    linear::Linear, logistic::Logistic, poisson::Poisson,
    },
    standardize::standardize,
};
