//! functions for solving logistic regression

use crate::{
    error::{RegressionError, RegressionResult},
    glm::{DispersionType, Glm},
    link::Link,
    math::prod_log,
    num::Float,
    response::Response,
};
use ndarray::Array1;
use std::marker::PhantomData;

/// Logistic regression
pub struct Logistic<L = link::Logit>
where
    L: Link<Logistic<L>>,
{
    _link: PhantomData<L>,
}

/// The logistic response variable must be boolean (at least for now).
impl<L> Response<Logistic<L>> for bool
where
    L: Link<Logistic<L>>,
{
    fn into_float<F: Float>(self) -> RegressionResult<F> {
        Ok(if self { F::one() } else { F::zero() })
    }
}
// Allow floats for the domain. We can't use num_traits::Float because of the
// possibility of conflicting implementations upstream, so manually implement
// for f32 and f64.
impl<L> Response<Logistic<L>> for f32
where
    L: Link<Logistic<L>>,
{
    fn into_float<F: Float>(self) -> RegressionResult<F> {
        if !(0.0..=1.0).contains(&self) {
            return Err(RegressionError::InvalidY(self.to_string()));
        }
        F::from(self).ok_or_else(|| RegressionError::InvalidY(self.to_string()))
    }
}
impl<L> Response<Logistic<L>> for f64
where
    L: Link<Logistic<L>>,
{
    fn into_float<F: Float>(self) -> RegressionResult<F> {
        if !(0.0..=1.0).contains(&self) {
            return Err(RegressionError::InvalidY(self.to_string()));
        }
        F::from(self).ok_or_else(|| RegressionError::InvalidY(self.to_string()))
    }
}

/// Implementation of GLM functionality for logistic regression.
impl<L> Glm for Logistic<L>
where
    L: Link<Logistic<L>>,
{
    type Link = L;
    const DISPERSED: DispersionType = DispersionType::NoDispersion;

    /// The log of the partition function for logistic regression. The natural
    /// parameter is the logit of p.
    fn log_partition<F: Float>(nat_par: F) -> F {
        num_traits::Float::exp(nat_par).ln_1p()
    }

    /// var = mu*(1-mu)
    fn variance<F: Float>(mean: F) -> F {
        mean * (F::one() - mean)
    }

    /// This function is specialized over the default provided by Glm in order
    /// to handle over/underflow issues more precisely.
    fn log_like_natural<F>(y: F, logit_p: F) -> F
    where
        F: Float,
    {
        let (yt, xt) = if logit_p < F::zero() {
            (y, logit_p)
        } else {
            (F::one() - y, -logit_p)
        };
        yt * xt - num_traits::Float::exp(xt).ln_1p()
    }

    /// The saturated likelihood is zero for logistic regression when y = 0 or 1 but is greater
    /// than zero for 0 < y < 1.
    fn log_like_sat<F: Float>(y: F) -> F {
        prod_log(y) + prod_log(F::one() - y)
    }
}

pub mod link {
    //! Link functions for logistic regression
    use super::*;
    use crate::link::{Canonical, Link, Transform};
    use crate::num::Float;

    /// The canonical link function for logistic regression is the logit function g(p) =
    /// log(p/(1-p)).
    pub struct Logit {}
    impl Canonical for Logit {}
    impl Link<Logistic<Logit>> for Logit {
        fn func<F: Float>(y: F) -> F {
            num_traits::Float::ln(y / (F::one() - y))
        }
        fn func_inv<F: Float>(lin_pred: F) -> F {
            (F::one() + num_traits::Float::exp(-lin_pred)).recip()
        }
    }

    /// The complementary log-log link g(p) = log(-log(1-p)) is appropriate when
    /// modeling the probability of non-zero counts when the counts are
    /// Poisson-distributed with mean lambda = exp(lin_pred).
    pub struct Cloglog {}
    impl Link<Logistic<Cloglog>> for Cloglog {
        fn func<F: Float>(y: F) -> F {
            num_traits::Float::ln(-F::ln_1p(-y))
        }
        // This quickly underflows to zero for inputs greater than ~2.
        fn func_inv<F: Float>(lin_pred: F) -> F {
            -F::exp_m1(-num_traits::Float::exp(lin_pred))
        }
    }
    impl Transform for Cloglog {
        fn nat_param<F: Float>(lin_pred: Array1<F>) -> Array1<F> {
            lin_pred.mapv(|x| num_traits::Float::ln(num_traits::Float::exp(x).exp_m1()))
        }
        fn d_nat_param<F: Float>(lin_pred: &Array1<F>) -> Array1<F> {
            let neg_exp_lin = -lin_pred.mapv(num_traits::Float::exp);
            &neg_exp_lin / &neg_exp_lin.mapv(F::exp_m1)
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
    #[test]
    fn log_reg() -> RegressionResult<()> {
        let beta = array![0., 1.0];
        let ln2 = f64::ln(2.);
        let data_x = array![[0.], [0.], [ln2], [ln2], [ln2]];
        let data_y = array![true, false, true, true, false];
        let model = ModelBuilder::<Logistic>::data(&data_y, &data_x).build()?;
        let fit = model.fit()?;
        // dbg!(fit.n_iter);
        // NOTE: This tolerance must be higher than it would ideally be.
        // Only 2 iterations are completed, so more accuracy could presumably be achieved with a
        // lower tolerance.
        assert_abs_diff_eq!(beta, fit.result, epsilon = 0.5 * f32::EPSILON as f64);
        // let lr = fit.lr_test();
        Ok(())
    }

    // verify that the link and inverse are indeed inverses.
    #[test]
    fn cloglog_closure() {
        use link::Cloglog;
        let mu_test_vals = array![1e-8, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.9999999];
        assert_abs_diff_eq!(
            mu_test_vals,
            mu_test_vals.mapv(|mu| Cloglog::func_inv(Cloglog::func(mu)))
        );
        let lin_test_vals = array![-10., -2., -0.1, 0.0, 0.1, 1., 2.];
        assert_abs_diff_eq!(
            lin_test_vals,
            lin_test_vals.mapv(|lin| Cloglog::func(Cloglog::func_inv(lin))),
            epsilon = 1e-3 * f32::EPSILON as f64
        );
    }
}
