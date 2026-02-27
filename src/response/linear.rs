//! Functions for solving linear regression

#[cfg(feature = "stats")]
use crate::response::Response;
use crate::{
    error::{RegressionError, RegressionResult},
    glm::{DispersionType, Glm},
    link::Link,
    num::Float,
    response::Yval,
};
use num_traits::ToPrimitive;
#[cfg(feature = "stats")]
use statrs::distribution::Normal;
use std::marker::PhantomData;

/// Linear regression with constant variance (Ordinary least squares).
pub struct Linear<L = link::Id>
where
    L: Link<Linear<L>>,
{
    _link: PhantomData<L>,
}

/// Allow all floating point types in the linear model.
impl<Y, L> Yval<Linear<L>> for Y
where
    Y: Float + ToPrimitive + ToString,
    L: Link<Linear<L>>,
{
    fn into_float<F: Float>(self) -> RegressionResult<F, F> {
        F::from(self).ok_or_else(|| RegressionError::InvalidY(self.to_string()))
    }
}

#[cfg(feature = "stats")]
impl<L> Response for Linear<L>
where
    L: Link<Linear<L>>,
{
    type DistributionType = Normal;

    fn get_distribution(mu: f64, phi: f64) -> Self::DistributionType {
        let sigma = phi.sqrt();
        // We might want to return the error, but we should be able to assume that neither are NaN
        // and sigma > 0.
        // TODO: We should probably return an error instead of unwrap()-ing each of these
        // distributions, because a sigma of zero is possible (e.g. in an underspecified model).
        // The statrs errors aren't unified so we can't implement a simple #[from] for our error
        // enum and will need to map_err in each implementation.
        Normal::new(mu, sigma).unwrap()
    }
}

impl<L> Glm for Linear<L>
where
    L: Link<Linear<L>>,
{
    type Link = L;
    const DISPERSED: DispersionType = DispersionType::FreeDispersion;

    /// Logarithm of the partition function in terms of the natural parameter,
    /// which is mu for OLS.
    fn log_partition<F: Float>(nat_par: F) -> F {
        let half = F::from(0.5).unwrap();
        half * nat_par * nat_par
    }

    /// variance is not a function of the mean in OLS regression.
    fn variance<F: Float>(_mean: F) -> F {
        F::one()
    }

    /// The saturated model likelihood is 0.5*y^2 for each observation. Note
    /// that if a sum of squares were used for the log-likelihood, this would be
    /// zero.
    fn log_like_sat<F: Float>(y: F) -> F {
        // Only for linear regression does this identity hold.
        Self::log_partition(y)
    }
}

pub mod link {
    //! Link functions for linear regression.
    use super::*;
    use crate::link::{Canonical, Link};

    /// The identity link function, which is canonical for linear regression.
    pub struct Id;
    /// The identity is the canonical link function.
    impl Canonical for Id {}
    impl Link<Linear> for Id {
        #[inline]
        fn func<F: Float>(y: F) -> F {
            y
        }
        #[inline]
        fn func_inv<F: Float>(lin_pred: F) -> F {
            lin_pred
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Linear;
    use crate::{error::RegressionResult, model::ModelBuilder};
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    // Check closure, which should be trivial for linear as the link and natural parameters are
    // just identity.
    fn id_closure() {
        use crate::link::TestLink;
        // Define an assorted array of values over several orders of magnitude without any
        // obvious exploitable patterns
        let x = crate::array![
            -1e5, -100., -13., -2.0, -1.0, -0.025, -0.001, 0., 0.001, 0.04, 1.0, 2.5, 17., 128.,
            1e5
        ];
        super::link::Id::check_closure(&x);
        super::link::Id::check_closure_y(&x);
    }

    #[test]
    fn lin_reg() -> RegressionResult<(), f64> {
        let beta = array![0.3, 1.2, -0.5];
        let data_x = array![[-0.1, 0.2], [0.7, 0.5], [3.2, 0.1]];
        // let data_x = array![[-0.1, 0.1], [0.7, -0.7], [3.2, -3.2]];
        let data_y = array![
            beta[0] + beta[1] * data_x[[0, 0]] + beta[2] * data_x[[0, 1]],
            beta[0] + beta[1] * data_x[[1, 0]] + beta[2] * data_x[[1, 1]],
            beta[0] + beta[1] * data_x[[2, 0]] + beta[2] * data_x[[2, 1]],
        ];
        let model = ModelBuilder::<Linear>::data(&data_y, &data_x).build()?;
        let fit = model.fit_options().max_iter(10).fit()?;
        // This is failing within the default tolerance
        assert_abs_diff_eq!(beta, fit.result, epsilon = 64.0 * f64::EPSILON);
        let _lr: f64 = fit.lr_test();
        Ok(())
    }
}
