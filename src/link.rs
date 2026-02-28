//! Traits and utilities for link functions.
//!
//! The link function $`g`$ maps the expected response $`\mu`$ to the linear predictor
//! $`\omega = \mathbf{x}^\mathsf{T}\boldsymbol{\beta}`$. Each family defaults to its canonical
//! link, but an alternative can be selected via the family's type parameter.
//!
//! # Using a provided non-canonical link
//!
//! Alternative links are re-exported for convenience: [`exp_link`](crate::exp_link) for
//! exponential regression and [`logistic_link`](crate::logistic_link) for logistic regression.
//! Provide a link as the family's type parameter:
//!
//! ```
//! use ndarray_glm::{Exponential, ModelBuilder, array, exp_link::Log};
//!
//! fn main() -> ndarray_glm::error::RegressionResult<(), f64> {
//!     let data_y = array![1.0, 2.5, 0.8, 3.1];
//!     let data_x = array![[0.0], [1.0], [0.5], [1.5]];
//!     // Use the log link instead of the default negative-reciprocal canonical link.
//!     let model = ModelBuilder::<Exponential<Log>>::data(&data_y, &data_x).build()?;
//!     let fit = model.fit()?;
//!     Ok(())
//! }
//! ```
//!
//! # Implementing a custom non-canonical link
//!
//! A non-canonical link requires two trait implementations:
//!
//! 1. [`Link<M>`] — the forward map $`g(\mu) = \omega`$ ([`Link::func`]) and its inverse
//!    $`g^{-1}(\omega) = \mu`$ ([`Link::func_inv`]).
//! 2. [`Transform`] — the natural-parameter transformation
//!    $`\eta(\omega) = g_0(g^{-1}(\omega))`$ ([`Transform::nat_param`]) and its derivative
//!    ([`Transform::d_nat_param`]), where $`g_0`$ is the family's canonical link. The derivative
//!    satisfies $`\eta'(\omega) = \frac{1}{g'(\mu)\,V(\mu)}`$ where $`V(\mu)`$ is the family's
//!    variance function evaluated at $`\mu = g^{-1}(\omega)`$.
//!
//! Example: a square-root link $`g(\mu) = \sqrt{\mu}`$ for Poisson regression. The canonical
//! link is $`\log`$ and $`V(\mu) = \mu`$, so
//! $`\eta(\omega) = \log(\omega^2) = 2\log\omega`$ and $`\eta'(\omega) = 2/\omega`$:
//!
//! ```
//! use ndarray_glm::{Poisson, link::{Link, Transform}, num::Float};
//! use ndarray::Array1;
//!
//! pub struct Sqrt;
//!
//! impl Link<Poisson<Sqrt>> for Sqrt {
//!     fn func<F: Float>(mu: F) -> F { num_traits::Float::sqrt(mu) }
//!     fn func_inv<F: Float>(omega: F) -> F { omega * omega }
//! }
//!
//! impl Transform for Sqrt {
//!     fn nat_param<F: Float>(lin_pred: Array1<F>) -> Array1<F> {
//!         lin_pred.mapv(|w| F::two() * num_traits::Float::ln(w))
//!     }
//!     fn d_nat_param<F: Float>(lin_pred: &Array1<F>) -> Array1<F> {
//!         lin_pred.mapv(|w| F::two() / w)
//!     }
//! }
//! ```
//!
//! # Consistency tests with `TestLink`
//!
//! The `TestLink` trait (available only in `#[cfg(test)]` builds) provides canned assertions
//! that every correct link implementation should satisfy. Call them from your test module:
//!
//! ```no_run
//! #[cfg(test)]
//! mod tests {
//!     use super::*;
//!     use ndarray_glm::link::TestLink;
//!     use ndarray::array;
//!
//!     #[test]
//!     fn sqrt_link_checks() {
//!         // Linear-predictor values; must lie in the domain of ω (ω > 0 for sqrt).
//!         let lin_vals = array![0.25, 1.0, 2.0, 4.0, 9.0];
//!
//!         // Verify g(g⁻¹(ω)) ≈ ω.
//!         Sqrt::check_closure(&lin_vals);
//!
//!         // Verify g⁻¹(g(μ)) ≈ μ; values must lie in the response domain (μ > 0 for Poisson).
//!         Sqrt::check_closure_y(&array![0.5, 1.0, 3.0, 10.0]);
//!
//!         // For non-canonical links: verify nat_param(ω) = g₀(g⁻¹(ω)).
//!         // Pass the *canonical* model variant as `Mc`. `Poisson` without a type parameter
//!         // defaults to the canonical log link.
//!         use ndarray_glm::Poisson;
//!         Sqrt::check_nat_par::<Poisson>(&lin_vals);
//!
//!         // Verify d_nat_param matches the numerical derivative.
//!         Sqrt::check_nat_par_d(&lin_vals);
//!     }
//! }
//! ```

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

    /// Check that $`\eta(\omega) = g_0(g^{-1}(\omega))`$ on the input domain, where $`g_0`$ is
    /// the canonical link for `M` (supplied as the type parameter `Mc`) and $`g^{-1}`$ is the
    /// inverse of the link under test. This is the defining property of the `nat_param`
    /// transformation and should hold for all normal linear predictor inputs.
    fn check_nat_par<Mc: Glm>(xs: &Array1<f64>);

    /// Check the derivative of the natural parameter function with numerical difference.
    /// In particular it compares the ratio of the numerical derivative to the analytical one, so
    /// that it can be evaluated with a constant epsilon.
    fn check_nat_par_d(xs: &Array1<f64>);
}

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

    fn check_nat_par<Mc: Glm>(xs: &Array1<f64>) {
        // nat_param(ω) is defined as g_0(g^{-1}(ω)), so verify:
        //   L::nat_param(xs)  ==  Mc::Link::func(L::func_inv(xs_i))
        let nat_par_direct = xs.mapv(|w| Mc::Link::func::<f64>(L::func_inv(w)));
        let nat_par_transform = L::nat_param(xs.clone());
        approx::assert_abs_diff_eq!(nat_par_direct, nat_par_transform, epsilon = 1e-6);
    }

    fn check_nat_par_d(xs: &Array1<f64>) {
        let delta = f32::EPSILON as f64;
        let d_eta = L::d_nat_param(xs);
        let x_plus = xs.clone().mapv_into(|x| x + delta / 2.);
        let x_minus = xs.clone().mapv_into(|x| x - delta / 2.);
        let eta_diff = L::nat_param(x_plus) - L::nat_param(x_minus);
        // Note that this requires d_eta != 0, but that should be the case for a good eta
        // function anyway.
        // The scaling is necessary because for some link functions a modest range of inputs can be
        // mapped over many orders of magnitude (e.g. 10^5 to 10^{-5}) and we want a consistent
        // epsilon over all of them.
        approx::assert_abs_diff_eq!(
            eta_diff / (delta * d_eta),
            Array1::<f64>::ones(xs.len()),
            epsilon = delta
        );
    }
}
