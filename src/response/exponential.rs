//! Exponential response in that y is drawn from the exponential distribution.

#[cfg(feature = "stats")]
use crate::response::Response;
use crate::{
    error::{RegressionError, RegressionResult},
    glm::{DispersionType, Glm},
    link::Link,
    num::Float,
    response::Yval,
};
use ndarray::Array1;
#[cfg(feature = "stats")]
use statrs::distribution::Exp;
use std::marker::PhantomData;

/// Exponential regression
pub struct Exponential<L = link::NegRec>
where
    L: Link<Exponential<L>>,
{
    _link: PhantomData<L>,
}

// Allow floats for the domain. We can't use num_traits::Float because of the
// possibility of conflicting implementations upstream, so manually implement
// for f32 and f64. Note that for exponential regression, y=0 is invalid.
impl<L> Yval<Exponential<L>> for f32
where
    L: Link<Exponential<L>>,
{
    fn into_float<F: Float>(self) -> RegressionResult<F, F> {
        if self <= 0. {
            return Err(RegressionError::InvalidY(self.to_string()));
        }
        F::from(self).ok_or_else(|| RegressionError::InvalidY(self.to_string()))
    }
}
impl<L> Yval<Exponential<L>> for f64
where
    L: Link<Exponential<L>>,
{
    fn into_float<F: Float>(self) -> RegressionResult<F, F> {
        if self <= 0. {
            return Err(RegressionError::InvalidY(self.to_string()));
        }
        F::from(self).ok_or_else(|| RegressionError::InvalidY(self.to_string()))
    }
}

#[cfg(feature = "stats")]
impl<L> Response for Exponential<L>
where
    L: Link<Exponential<L>>,
{
    type DistributionType = Exp;

    fn get_distribution(mu: f64, _phi: f64) -> Self::DistributionType {
        // NOTE: Negative mu is a realistic concern for exponential regression, since the canonical
        // link function does not prevent them. Without complicating the return type, either with
        // dynamic dispatch or an enum between Exp and Dirac that would have to forward every
        // Distribution<f64> and ContinuousCDF<f64> calls, the simplest way to ensure μ > 0 is
        // to clamp at the lowest positive value (~2e-308).
        // Exp::new(rate) where rate = 1/mu, since statrs parameterizes by rate (mean = 1/rate).
        Exp::new(mu.max(f64::MIN_POSITIVE).recip()).unwrap()
    }
}

/// Implementation of GLM functionality for exponential regression.
impl<L> Glm for Exponential<L>
where
    L: Link<Exponential<L>>,
{
    type Link = L;
    const DISPERSED: DispersionType = DispersionType::NoDispersion;

    /// The log-partition function $`A(\eta)`$ for the exponential family, expressed in terms
    /// of the canonical natural parameter $`\eta = -1/\mu`$:
    ///
    /// ```math
    /// A(\eta) = -\ln(-\eta)
    /// ```
    fn log_partition<F: Float>(nat_par: F) -> F {
        -num_traits::Float::ln(-nat_par)
    }

    /// The variance function $`V(\mu) = \mu^2`$, equal to $`A''(\eta)`$ evaluated at
    /// $`\eta = -1/\mu`$.
    fn variance<F: Float>(mean: F) -> F {
        mean * mean
    }

    /// The saturated likelihood is -1 - log(y). This shows part of why exponential regression
    /// can't deal with y=0.
    fn log_like_sat<F: Float>(y: F) -> F {
        -(F::one() + num_traits::Float::ln(y))
    }
}

pub mod link {
    //! Link functions for exponential regression
    use super::*;
    use crate::link::{Canonical, Link, Transform};
    use crate::num::Float;

    /// The canonical link function for exponential regression is the negative reciprocal
    /// $`\eta = -1/mu`$. This fails to prevent negative predicted y-values.
    pub struct NegRec {}
    impl Canonical for NegRec {}
    impl Link<Exponential<NegRec>> for NegRec {
        fn func<F: Float>(y: F) -> F {
            -num_traits::Float::recip(y)
        }
        fn func_inv<F: Float>(lin_pred: F) -> F {
            -num_traits::Float::recip(lin_pred)
        }
    }

    /// The log link $`g(\mu) = \log(\mu)`$ avoids linear predictors that give negative
    /// expectations.
    pub struct Log {}
    impl Link<Exponential<Log>> for Log {
        fn func<F: Float>(y: F) -> F {
            num_traits::Float::ln(y)
        }
        fn func_inv<F: Float>(lin_pred: F) -> F {
            num_traits::Float::exp(lin_pred)
        }
    }
    impl Transform for Log {
        fn nat_param<F: Float>(lin_pred: Array1<F>) -> Array1<F> {
            lin_pred.mapv(|x| -num_traits::Float::exp(-x))
        }
        fn d_nat_param<F: Float>(lin_pred: &Array1<F>) -> Array1<F> {
            lin_pred.mapv(|x| num_traits::Float::exp(-x))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{error::RegressionResult, model::ModelBuilder};
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    /// A simple test where the correct value for the data is known exactly.
    ///
    /// With the canonical NegRec link, the MLE satisfies β₀ = -1/ȳ₀ and β₀+β₁ = -1/ȳ₁,
    /// where ȳ₀ and ȳ₁ are the within-group sample means. Choosing group means 2 and 4
    /// gives β = [-0.5, 0.25], both exactly representable in f64.
    #[test]
    fn exp_ex() -> RegressionResult<(), f64> {
        // Group 0 (x=0): y ∈ {1, 3}, ȳ₀ = 2  → β₀       = -1/2 = -0.5
        // Group 1 (x=1): y ∈ {2, 4, 6}, ȳ₁ = 4 → β₀ + β₁ = -1/4, β₁ = 0.25
        let beta = array![-0.5, 0.25];
        let data_x = array![[0.], [0.], [1.0], [1.0], [1.0]];
        let data_y = array![1.0, 3.0, 2.0, 4.0, 6.0];
        let model = ModelBuilder::<Exponential>::data(&data_y, &data_x).build()?;
        let fit = model.fit()?;
        assert_abs_diff_eq!(beta, fit.result, epsilon = 0.5 * f32::EPSILON as f64);
        let _cov = fit.covariance()?;
        Ok(())
    }

    /// Analogous test using the Log link. With g(μ) = log(μ), the MLE satisfies
    /// β₀ = log(ȳ₀) and β₁ = log(ȳ₁/ȳ₀). Same group data as exp_ex gives
    /// ȳ₀=2, ȳ₁=4, so β = [ln 2, ln 2].
    #[test]
    fn exp_log_link_ex() -> RegressionResult<(), f64> {
        // Group 0 (x=0): y ∈ {1, 3}, ȳ₀ = 2 → β₀      = ln(2)
        // Group 1 (x=1): y ∈ {2, 4, 6}, ȳ₁ = 4 → β₀+β₁ = ln(4), β₁ = ln(2)
        let ln2 = f64::ln(2.);
        let beta = array![ln2, ln2];
        let data_x = array![[0.], [0.], [1.0], [1.0], [1.0]];
        let data_y = array![1.0, 3.0, 2.0, 4.0, 6.0];
        let model = ModelBuilder::<Exponential<link::Log>>::data(&data_y, &data_x).build()?;
        let fit = model.fit()?;
        assert_abs_diff_eq!(beta, fit.result, epsilon = 0.5 * f32::EPSILON as f64);
        let _cov = fit.covariance()?;
        Ok(())
    }

    #[test]
    // Confirm inverse reciprocal closure.
    fn neg_rec_closure() {
        use super::link::NegRec;
        use crate::link::TestLink;
        // Note that the positive values aren't good linear predictor values, but they should be
        // closed under the canonical transformation anyway.
        let x = array![-360., -12., -5., -1.0, -0.002, 0., 0.5, 20.];
        NegRec::check_closure(&x);
        let y = array![1e-5, 0.25, 0.8, 2.5, 10., 256.];
        NegRec::check_closure_y(&y);
    }

    // verify closure for the log link.
    #[test]
    fn log_closure() {
        use crate::link::TestLink;
        use link::Log;
        let mu_test_vals = array![1e-8, 0.01, 0.1, 0.3, 0.9, 1.8, 4.2, 148.];
        Log::check_closure_y(&mu_test_vals);
        let lin_test_vals = array![1e-8, 0.002, 0.5, 2.4, 15., 120.];
        Log::check_closure(&lin_test_vals);
    }

    #[test]
    fn log_nat_par() {
        use crate::link::TestLink;
        use link::Log;
        // nat_param(ω) = -exp(-ω) = g_0(g^{-1}(ω)) = NegRec(exp(ω)) = -1/exp(ω)
        let lin_test_vals = array![-10., -2., -0.5, 0.0, 0.5, 2., 10.];
        Log::check_nat_par::<Exponential<link::NegRec>>(&lin_test_vals);
        Log::check_nat_par_d(&lin_test_vals);
    }
}
