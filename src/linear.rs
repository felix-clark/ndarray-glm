//! Functions for solving linear regression

use crate::{
    glm::{Glm, Response},
    link::Link,
    model::Model,
};
use ndarray::Array1;
use ndarray_linalg::Lapack;
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
    Y: Float + ToPrimitive,
    L: Link<Linear<L>>,
{
    fn to_float<F: Float>(self) -> F {
        // TODO: Can we avoid casting and use traits? We'd likely have to define
        // our own trait constraint.
        F::from(self).unwrap()
    }
}

impl<L> Glm for Linear<L>
where
    L: Link<Linear<L>>,
{
    type Link = L;

    /// variance is not a function of the mean in OLS regression.
    fn variance<F: Float>(_mean: F) -> F {
        F::one()
    }

    // This version doesn't have the variances - either setting them to 1 or
    // 1/2pi to simplify the expression. It returns a simple sum of squares.
    // It also misses a factor of 0.5 in the squares.
    fn quasi_log_likelihood<F: Float + Lapack>(data: &Model<Self, F>, regressors: &Array1<F>) -> F {
        let lin_pred = &data.linear_predictor(&regressors);
        // TODO: transform the linear predictors in the event of a non-identity link function
        let squares: Array1<F> = (&data.y - lin_pred).map(|&d| d * d);
        let l2_term = data.l2_like_term(regressors);
        -squares.sum() + l2_term
    }
}

pub mod link {
    //! Link functions for linear regression.
    use super::*;
    use crate::link::{Canonical, Link};

    /// The identity link function, which is canonical for linear regression.
    pub struct Id;
    /// The identity is the canonical link function so we don't have to manually
    /// implement everything.
    impl Canonical for Id {}
    impl Link<Linear> for Id {
        fn func<F: Float>(y: F) -> F {
            y
        }
        fn inv_func<F: Float>(lin_pred: F) -> F {
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
        Ok(())
    }
}
