//! Inverse Gaussian response where y is drawn from the inverse Gaussian distribution.

use crate::{
    error::{RegressionError, RegressionResult},
    glm::{DispersionType, Glm},
    link::Link,
    num::Float,
    response::Yval,
};
use ndarray::Array1;
use std::marker::PhantomData;

/// Inverse Gaussian regression with the canonical link function $`g_0(\mu) = -1/\mu^2`$.
pub struct InvGaussian<L = link::NegRecSq>
where
    L: Link<InvGaussian<L>>,
{
    _link: PhantomData<L>,
}

// Note that for inverse gaussian regression, y=0 is invalid.
impl<L> Yval<InvGaussian<L>> for f32
where
    L: Link<InvGaussian<L>>,
{
    fn into_float<F: Float>(self) -> RegressionResult<F, F> {
        if self <= 0. {
            return Err(RegressionError::InvalidY(self.to_string()));
        }
        F::from(self).ok_or_else(|| RegressionError::InvalidY(self.to_string()))
    }
}
impl<L> Yval<InvGaussian<L>> for f64
where
    L: Link<InvGaussian<L>>,
{
    fn into_float<F: Float>(self) -> RegressionResult<F, F> {
        if self <= 0. {
            return Err(RegressionError::InvalidY(self.to_string()));
        }
        F::from(self).ok_or_else(|| RegressionError::InvalidY(self.to_string()))
    }
}

// NOTE: The statrs package doesn't currently support the Inverse Gaussian distribution, so we
// can't easily implement Response at this time.

/// Implementation of GLM functionality for inverse Gaussian regression.
impl<L> Glm for InvGaussian<L>
where
    L: Link<InvGaussian<L>>,
{
    type Link = L;
    const DISPERSED: DispersionType = DispersionType::FreeDispersion;

    /// The log-partition function $`A(\eta)`$ for the inverse Gaussian family, expressed in
    /// terms of the canonical natural parameter $`\eta = -1/\mu^2`$:
    ///
    /// ```math
    /// A(\eta) = -2\sqrt{-\eta}
    /// ```
    fn log_partition<F: Float>(nat_par: F) -> F {
        -F::two() * num_traits::Float::sqrt(-nat_par)
    }

    /// The variance function $`V(\mu) = \mu^3/2`$, equal to $`A''(\eta)`$ evaluated at
    /// $`\eta = -1/\mu^2`$.
    fn variance<F: Float>(mean: F) -> F {
        mean * mean * mean * F::half()
    }

    /// The saturated log-likelihood $`y \eta_{\text{sat}} - A(\eta_{\text{sat}})`$ at
    /// $`\eta_{\text{sat}} = -1/y^2`$, which evaluates to $`1/y`$.
    fn log_like_sat<F: Float>(y: F) -> F {
        num_traits::Float::recip(y)
    }
}

pub mod link {
    //! Link functions for inverse Gaussian regression
    use super::*;
    use crate::link::{Canonical, Link, Transform};
    use crate::num::Float;

    /// The canonical link function for inverse Gaussian regression is the negative reciprocal
    /// square $`\eta = -1/\mu^2`$. This fails to prevent negative predicted y-values.
    pub struct NegRecSq {}
    impl Canonical for NegRecSq {}
    impl Link<InvGaussian<NegRecSq>> for NegRecSq {
        fn func<F: Float>(y: F) -> F {
            -num_traits::Float::recip(y * y)
        }
        fn func_inv<F: Float>(lin_pred: F) -> F {
            num_traits::Float::recip(num_traits::Float::sqrt(-lin_pred))
        }
    }

    /// The log link $`g(\mu) = \log(\mu)`$ avoids linear predictors that give negative
    /// expectations.
    pub struct Log {}
    impl Link<InvGaussian<Log>> for Log {
        fn func<F: Float>(y: F) -> F {
            num_traits::Float::ln(y)
        }
        fn func_inv<F: Float>(lin_pred: F) -> F {
            num_traits::Float::exp(lin_pred)
        }
    }
    impl Transform for Log {
        fn nat_param<F: Float>(lin_pred: Array1<F>) -> Array1<F> {
            // η(ω) = g₀(g⁻¹(ω)) = -1/exp(ω)² = -exp(-2ω)
            lin_pred.mapv(|x| -num_traits::Float::exp(-F::two() * x))
        }
        fn d_nat_param<F: Float>(lin_pred: &Array1<F>) -> Array1<F> {
            // η'(ω) = 2·exp(-2ω) = 1/(g'(μ)·V(μ)) = 1/((1/μ)·(μ³/2)) = 2/μ²
            lin_pred.mapv(|x| F::two() * num_traits::Float::exp(-F::two() * x))
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
    /// With the canonical NegRecSq link g(μ) = -1/μ², the MLE satisfies β₀ = -1/ȳ₀² and
    /// β₀+β₁ = -1/ȳ₁², where ȳ₀ and ȳ₁ are the within-group sample means. Choosing group
    /// means 2 and 4 gives β = [-0.25, 0.1875], both exactly representable in f64.
    #[test]
    fn ig_ex() -> RegressionResult<(), f64> {
        // Group 0 (x=0): y ∈ {1, 3}, ȳ₀ = 2  → β₀       = -1/4  = -0.25
        // Group 1 (x=1): y ∈ {2, 4, 6}, ȳ₁ = 4 → β₀ + β₁ = -1/16, β₁ = 3/16 = 0.1875
        let beta = array![-0.25, 0.1875];
        let data_x = array![[0.], [0.], [1.0], [1.0], [1.0]];
        let data_y = array![1.0, 3.0, 2.0, 4.0, 6.0];
        let model = ModelBuilder::<InvGaussian>::data(&data_y, &data_x).build()?;
        let fit = model.fit()?;
        assert_abs_diff_eq!(beta, fit.result, epsilon = 0.5 * f32::EPSILON as f64);
        let _cov = fit.covariance()?;
        Ok(())
    }

    /// Analogous test using the Log link. With g(μ) = log(μ), the MLE satisfies
    /// β₀ = log(ȳ₀) and β₁ = log(ȳ₁/ȳ₀). Same group data as ig_ex gives
    /// ȳ₀=2, ȳ₁=4, so β = [ln 2, ln 2].
    #[test]
    fn ig_log_link_ex() -> RegressionResult<(), f64> {
        // Group 0 (x=0): y ∈ {1, 3}, ȳ₀ = 2 → β₀      = ln(2)
        // Group 1 (x=1): y ∈ {2, 4, 6}, ȳ₁ = 4 → β₀+β₁ = ln(4), β₁ = ln(2)
        let ln2 = f64::ln(2.);
        let beta = array![ln2, ln2];
        let data_x = array![[0.], [0.], [1.0], [1.0], [1.0]];
        let data_y = array![1.0, 3.0, 2.0, 4.0, 6.0];
        let model = ModelBuilder::<InvGaussian<link::Log>>::data(&data_y, &data_x).build()?;
        let fit = model.fit()?;
        assert_abs_diff_eq!(beta, fit.result, epsilon = 0.5 * f32::EPSILON as f64);
        let _cov = fit.covariance()?;
        Ok(())
    }

    #[test]
    // Confirm inverse reciprocal square closure.
    fn neg_rec_sq_closure() {
        use super::link::NegRecSq;
        use crate::link::TestLink;
        // Positive values can't be used with the canonical link.
        let x = array![-360., -12., -5., -1.0, -1e-4, 0.];
        NegRecSq::check_closure(&x);
        let y = array![1e-5, 0.25, 0.8, 2.5, 10., 256.];
        NegRecSq::check_closure_y(&y);
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
        // nat_param(ω) = -exp(-2ω) = g_0(g^{-1}(ω)) = NegRecSq(exp(ω)) = -1/exp(2ω)
        let lin_test_vals = array![-10., -2., -0.5, 0.0, 0.5, 2., 10.];
        Log::check_nat_par::<InvGaussian<link::NegRecSq>>(&lin_test_vals);
        Log::check_nat_par_d(&lin_test_vals);
    }
}
