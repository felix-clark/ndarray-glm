//! Response functions

// opt-in for binomial regression as it utilizes unstable features.
#[cfg(feature = "binomial")]
pub mod binomial;
pub mod linear;
pub mod logistic;
pub mod poisson;
