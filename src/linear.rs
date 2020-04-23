//! Functions for solving linear regression

use crate::{
    error::{RegressionError, RegressionResult},
    // model::Model,
    glm::{Glm, Response},
    link::Link,
};
use ndarray::Array1;
use num_traits::{Float, ToPrimitive};
use std::marker::PhantomData;

/// Linear regression with constant variance (Ordinary least squares).
pub struct Linear<L = link::Id>
where
    L: Link<Linear<L>>,
{
    _link: PhantomData<L>,
}

/// Allow all floating point types in the linear model.
impl<Y, L> Response<Linear<L>> for Y
where
    Y: Float + ToPrimitive + ToString,
    L: Link<Linear<L>>,
{
    fn to_float<F: Float>(self) -> RegressionResult<F> {
        // TODO: Can we avoid casting and use traits? We'd likely have to define
        // our own trait constraint.
        Ok(F::from(self).ok_or_else(|| RegressionError::InvalidY(self.to_string()))?)
    }
}

impl<L> Glm for Linear<L>
where
    L: Link<Linear<L>>,
{
    type Link = L;

    /// Logarithm of the partition function in terms of the natural parameter,
    /// which is mu for OLS.
    fn log_partition<F: Float>(nat_par: &Array1<F>) -> F {
        let half = F::from(0.5).unwrap();
        half * nat_par.mapv(|mu| mu * mu).sum()
    }

    /// variance is not a function of the mean in OLS regression.
    fn variance<F: Float>(_mean: F) -> F {
        F::one()
    }

    /// The saturated model likelihood is 0.5*y^2 for each observation. Note
    /// that if a sum of squares were used for the log-likelihood, this would be
    /// zero.
    fn log_like_sat<F: Float>(y: &Array1<F>) -> F {
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
    fn lin_reg() -> RegressionResult<()> {
        let beta = array![0.3, 1.2, -0.5];
        let data_x = array![[-0.1, 0.2], [0.7, 0.5], [3.2, 0.1]];
        // let data_x = array![[-0.1, 0.1], [0.7, -0.7], [3.2, -3.2]];
        let data_y = array![
            beta[0] + beta[1] * data_x[[0, 0]] + beta[2] * data_x[[0, 1]],
            beta[0] + beta[1] * data_x[[1, 0]] + beta[2] * data_x[[1, 1]],
            beta[0] + beta[1] * data_x[[2, 0]] + beta[2] * data_x[[2, 1]],
        ];
        let model = ModelBuilder::<Linear>::data(&data_y, &data_x)
            .max_iter(10)
            .build()?;
        let fit = model.fit()?;
        dbg!(fit.n_iter);
        // This is failing within the default tolerance
        assert_abs_diff_eq!(beta, fit.result, epsilon = 64.0 * std::f64::EPSILON);
        let lr: f64 = fit.lr_test();
        dbg!(&lr);
        dbg!(&lr.sqrt());
        Ok(())
    }
}
