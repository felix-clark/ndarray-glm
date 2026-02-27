//! Defines traits for link functions

use crate::{glm::Glm, num::Float};
use ndarray::Array1;

/// Describes the link function $`g`$ that maps between the expected response $`\mu`$ and
/// the linear predictor $`\omega = \mathbf{x}^\mathsf{T}\boldsymbol{\beta}`$:
///
/// ```math
/// g(\mu) = \omega, \qquad \mu = g^{-1}(\omega)
/// ```
pub trait Link<M: Glm>: Transform {
    /// Maps the expectation value of the response variable to the linear
    /// predictor. In general this is determined by a composition of the inverse
    /// natural parameter transformation and the canonical link function.
    fn func<F: Float>(y: F) -> F;
    /// Maps the linear predictor to the expectation value of the response.
    fn func_inv<F: Float>(lin_pred: F) -> F;
}

pub trait Transform {
    /// The natural parameter of the response distribution as a function
    /// of the linear predictor: $`\eta(\omega) = g_0(g^{-1}(\omega))`$ where $`g_0`$ is the
    /// canonical link. For canonical links this is the identity.
    fn nat_param<F: Float>(lin_pred: Array1<F>) -> Array1<F>;
    /// The derivative $`\eta'(\omega)`$ of the transformation to the natural parameter.
    /// If it is zero in a region that the IRLS is in, the algorithm may have difficulty
    /// converging.
    /// It is given in terms of the link and variance functions as $`\eta'(\omega_i) =
    /// \frac{1}{g'(\mu_i) V(\mu_i)}`$.
    fn d_nat_param<F: Float>(lin_pred: &Array1<F>) -> Array1<F>;
    /// Adjust the error/residual terms of the likelihood function based on the first derivative of
    /// the transformation. The linear predictor must be un-transformed, i.e. it must be X*beta
    /// without the transformation applied.
    fn adjust_errors<F: Float>(errors: Array1<F>, lin_pred: &Array1<F>) -> Array1<F> {
        let eta_d = Self::d_nat_param(lin_pred);
        eta_d * errors
    }
    /// Adjust the variance terms of the likelihood function based on the first and second
    /// derivatives of the transformation. The linear predictor must be un-transformed, i.e. it
    /// must be X*beta without the transformation applied.
    fn adjust_variance<F: Float>(variance: Array1<F>, lin_pred: &Array1<F>) -> Array1<F> {
        let eta_d = Self::d_nat_param(lin_pred);
        // The second-derivative term in the variance matrix can lead it to not
        // be positive-definite. In fact, the second term should vanish when
        // taking the expecation of Y to give the Fisher information.
        // let var_adj = &eta_d * &variance * eta_d - eta_dd * errors;
        &eta_d * &variance * eta_d
    }
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
    fn adjust_errors<F: Float>(errors: Array1<F>, _lin_pred: &Array1<F>) -> Array1<F> {
        errors
    }
    #[inline]
    fn adjust_variance<F: Float>(variance: Array1<F>, _lin_pred: &Array1<F>) -> Array1<F> {
        variance
    }
    #[inline]
    fn adjust_errors_variance<F: Float>(
        errors: Array1<F>,
        variance: Array1<F>,
        _lin_pred: &Array1<F>,
    ) -> (Array1<F>, Array1<F>) {
        (errors, variance)
    }
}

/// Implement some common testing methods that every link function should satisfy.
#[cfg(test)]
pub trait TestLink<M: Glm> {
    /// Assert that $`g(g^{-1}(\omega)) = 1`$ for the entire input array. Since the input domain is
    /// that of the linear predictor, this should hold for all normal inputs.
    fn check_closure(xs: &Array1<f64>);

    /// Assert that $`g^{-1}(g(y)) = 1`$ for the entire input array. Since the input domain is
    /// that of the response variable, the input array must be in the domain of y.
    fn check_closure_y(ys: &Array1<f64>);
}

// NOTE: Does TestLink belong on Transform instead? But Transform doesn't have link/link_inv?
#[cfg(test)]
impl<L, M> TestLink<M> for L
where
    M: Glm,
    L: Link<M>,
{
    fn check_closure(x: &Array1<f64>) {
        let x_closed = x.clone().mapv_into(|w| L::func(L::func_inv(w)));
        // We need a relatively generous epsilon since some of these back-and-forths do lose
        // precision
        approx::assert_abs_diff_eq!(*x, x_closed, epsilon = 1e-6);
    }

    fn check_closure_y(y: &Array1<f64>) {
        let y_closed = y.clone().mapv_into(|y| L::func_inv(L::func(y)));
        approx::assert_abs_diff_eq!(*y, y_closed, epsilon = f32::EPSILON as f64);
    }
}
