//! Response functions

use crate::{error::RegressionResult, glm::Glm, num::Float};

pub mod binomial;
pub mod linear;
pub mod logistic;
pub mod poisson;

/// Describes the domain of the response variable for a GLM, e.g. integer for
/// Poisson, float for Linear, bool for logistic. Implementing this trait for a
/// type Y shows how to convert to a floating point type and allows that type to
/// be used as a response variable.
pub trait Response<M: Glm> {
    /// Converts the domain to a floating-point value for IRLS.
    fn into_float<F: Float>(self) -> RegressionResult<F>;
}
