//! Defines traits for link functions

use crate::glm::Glm;
use ndarray::Array1;
use num_traits::Float;

// pub trait Link<F: Float, M: Glm<F>>: Transform {
pub trait Link<M: Glm>: Transform {
    // fn func<F: Float>(y: F) -> F;
    fn func<F: Float>(y: F) -> F;
    fn inv_func<F: Float>(lin_pred: F) -> F;
    // TODO: parameter transform function, its derivatives, ..., propagate this info to the likelihood
    // the transformation function that takes the linear predictor to the
    // canonical parameter. Should always be identify for canonical link
    // functions.
    // fn canonical(lin_pred: Array1<F>) -> Array1<F>;
}

/// Describes the transformation of the linear parameters and its derivatives.
pub trait Transform {
    fn canonical<F: Float>(lin_pred: Array1<F>) -> Array1<F>;
}

pub trait Canonical {}
impl<T> Transform for T
where
    T: Canonical,
{
    fn canonical<F: Float>(lin_pred: Array1<F>) -> Array1<F> {
        lin_pred
    }
}
