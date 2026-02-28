//! Response functions

use crate::{error::RegressionResult, glm::Glm, num::Float};
#[cfg(feature = "stats")]
use statrs::statistics::Distribution;

pub mod binomial;
pub mod exponential;
pub mod gamma;
pub mod linear;
pub mod logistic;
pub mod poisson;

/// Describes the domain of the response variable for a GLM, e.g. integer for
/// Poisson, float for Linear, bool for logistic. Implementing this trait for a
/// type Y shows how to convert to a floating point type and allows that type to
/// be used as a response variable.
pub trait Yval<M: Glm> {
    /// Converts the domain to a floating-point value for IRLS.
    fn into_float<F: Float>(self) -> RegressionResult<F, F>;
}

// If this works this should be gated behind stats
#[cfg(feature = "stats")]
pub trait Response {
    type DistributionType: Distribution<f64>;

    /// Get the response distribution in terms of an expected mean, and possibly measured
    /// dispersion parameter. Families without free dispersion will not use the phi parameter.
    fn get_distribution(mu: f64, phi: f64) -> Self::DistributionType;
}
