//! Functions for solving linear regression

use crate::{glm::Glm, model::Model};
use ndarray::Array1;
use ndarray_linalg::Lapack;
use num_traits::Float;

pub struct Linear;

impl<F> Glm<F> for Linear
where
    F: Float + Lapack,
{
    type Domain = F;

    fn y_float(y: Self::Domain) -> F {
        y
    }

    // the link function, identity
    fn link(y: F) -> F {
        y
    }

    // inverse link function, identity
    fn mean(lin_pred: F) -> F {
        lin_pred
    }

    // variance is not a function of the mean
    fn variance(_mean: F) -> F {
        F::one()
    }

    // This version doesn't have the variances - either setting them to 1 or
    // 1/2pi to simplify the expression. It returns a simple sum of squares.
    // It also misses a factor of 0.5 in the squares.
    fn quasi_log_likelihood(data: &Model<Self, F>, regressors: &Array1<F>) -> F {
        let lin_pred = &data.linear_predictor(&regressors);
        let squares: Array1<F> = (&data.y - lin_pred).map(|&d| d * d);
        let l2_term = data.l2_like_term(regressors);
        -squares.sum() + l2_term
    }
}
