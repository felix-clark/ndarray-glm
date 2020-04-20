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

    /// The likelihood is -1/2 times the sum of squared deviations.
    fn log_like_params<F: Float + Lapack>(data: &Model<Self, F>, regressors: &Array1<F>) -> F {
        let lin_pred: Array1<F> = data.linear_predictor(&regressors);
        // Usually OLS will just use the canonical link function. It would be
        // nice to specialize the canonical case to avoid mapping the identify
        // function over the whole array. TODO: consider changing link and
        // mean functions to act on arrays.
        // let mu = lin_pred.mapv(Self::mean);
        let mu = L::nat_param(lin_pred);
        let squares: Array1<F> = (&data.y - &mu).map(|&d| d * d);
        -F::from(0.5).unwrap() * squares.sum()
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
        fn func<F: Float>(y: Array1<F>) -> Array1<F> {
            y
        }
        #[inline]
        fn func_inv<F: Float>(lin_pred: Array1<F>) -> Array1<F> {
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
