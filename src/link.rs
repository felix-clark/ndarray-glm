//! Defines traits for link functions

use crate::glm::Glm;
use ndarray::Array1;
use num_traits::Float;

/// Describes the functions to map to and from the linear predictors and the
/// expectation of the response. It is constrained mathematically by the
/// response distribution and the transformation of the linear predictor.
pub trait Link<M: Glm>: Transform {
    /// Maps the expectation value of the response variable to the linear predictor.
    fn func<F: Float>(y: F) -> F;
    /// Maps the linear predictor to the expectation value of the response.
    fn inv_func<F: Float>(lin_pred: F) -> F;
}

/// Describes the transformation of the linear parameters into the natural
/// parameter and the derivatives of this function.
pub trait Transform {
    /// The natural parameter(s) of the exponential distribution as a function
    /// of the linear predictor. For canonical link functions this is the
    /// identity.
    fn nat_param<F: Float>(lin_pred: Array1<F>) -> Array1<F>;
}

/// The canonical transformation by definition equates the linear predictor with
/// the natural parameter of the response distribution. Implementing this trait
/// for a link function automatically defines the trivial transformation
/// functions.
pub trait Canonical {}
impl<T> Transform for T
where
    T: Canonical,
{
    /// By defintion this function is the identity function for canonical links.
    fn nat_param<F: Float>(lin_pred: Array1<F>) -> Array1<F> {
        lin_pred
    }
}
