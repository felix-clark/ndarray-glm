//! Defines traits for link functions

use crate::{glm::Glm, num::Float};
use ndarray::Array1;

/// Describes the functions to map to and from the linear predictors and the
/// expectation of the response. It is constrained mathematically by the
/// response distribution and the transformation of the linear predictor.
// TODO: The link function and its inverse are independent of the response
// distribution. This could be refactored to separate the function itself from
// the transformation that works with the distribution.
pub trait Link<M: Glm>: Transform {
    /// Maps the expectation value of the response variable to the linear
    /// predictor. In general this is determined by a composition of the inverse
    /// natural parameter transformation and the canonical link function.
    fn func<F: Float>(y: F) -> F;
    // fn func<F: Float>(y: Array1<F>) -> Array1<F>;
    /// Maps the linear predictor to the expectation value of the response.
    // TODO: There may not be a point in using Array versions of these functions
    // since clones are necessary anyway. Perhaps we could simply define the
    // scalar function and use mapv().
    fn func_inv<F: Float>(lin_pred: F) -> F;
    // fn func_inv<F: Float>(lin_pred: Array1<F>) -> Array1<F>;
}

pub trait Transform {
    /// The natural parameter(s) of the response distribution as a function
    /// of the linear predictor. For canonical link functions this is the
    /// identity. It must be monotonic, invertible, and twice-differentiable.
    /// For link function g and canonical link function g_0 it is equal to
    /// g_0 ( g^{-1}(lin_pred) ) .
    fn nat_param<F: Float>(lin_pred: Array1<F>) -> Array1<F>;
    /// The derivative of the transformation to the natural parameter. If it is
    /// zero in a region that the IRLS is in the algorithm may have difficulty
    /// converging.
    fn d_nat_param<F: Float>(lin_pred: &Array1<F>) -> Array1<F>;
    /// Adjust the error and variance terms of the likelihood function based on
    /// the first and second derivatives of the transformation. The adjustment
    /// is performed simultaneously. The linear predictor must be
    /// un-transformed, i.e. it must be X*beta without the transformation
    /// applied.
    fn adjust_errors_variance<F: Float>(
        errors: Array1<F>,
        variance: Array1<F>,
        lin_pred: &Array1<F>,
    ) -> (Array1<F>, Array1<F>) {
        let eta_d = Self::d_nat_param(lin_pred);
        let err_adj = &eta_d * &errors;
        // The second-derivative term in the variance matrix can lead it to not
        // be positive-definite. In fact, the second term should vanish when
        // taking the expecation of Y to give the Fisher information.
        // let var_adj = &eta_d * &variance * eta_d - eta_dd * errors;
        let var_adj = &eta_d * &variance * eta_d;
        (err_adj, var_adj)
    }
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
    #[inline]
    fn nat_param<F: Float>(lin_pred: Array1<F>) -> Array1<F> {
        lin_pred
    }
    #[inline]
    fn d_nat_param<F: Float>(lin_pred: &Array1<F>) -> Array1<F> {
        Array1::<F>::ones(lin_pred.len())
    }
    /// The canonical link function requires no transformation of the error and variance terms.
    #[inline]
    fn adjust_errors_variance<F: Float>(
        errors: Array1<F>,
        variance: Array1<F>,
        _lin_pred: &Array1<F>,
    ) -> (Array1<F>, Array1<F>) {
        (errors, variance)
    }
}
