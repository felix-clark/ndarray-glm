//! Model for Poisson regression

use crate::{
    glm::{Glm, Likelihood},
    model::Model,
};
use ndarray::Array1;
use ndarray_linalg::Lapack;
use num_traits::{Float, ToPrimitive, Unsigned};
use std::marker::PhantomData;

/// trait-based implementation to work towards generalization
pub struct Poisson<D>
where
    D: Unsigned,
{
    unsigned: PhantomData<D>,
}

impl<D, F> Glm<F> for Poisson<D>
where
    D: Unsigned + ToPrimitive,
    F: Float + Lapack,
{
    // TODO: this could be relaxed to a float with only mild changes, although
    // it would require checking that 0 <= y <= 1.
    // There should be a domain and a function that maps domain to a float.
    // Should this be a generic unsigned integer type?
    type Domain = D;

    fn y_float(y: Self::Domain) -> F {
        F::from(y).unwrap()
    }

    // the link function, the logarithm
    fn link(y: F) -> F {
        Float::ln(y)
    }

    // inverse link function, exponential
    fn mean(lin_pred: F) -> F {
        lin_pred.exp()
    }

    // var = mu
    fn variance(mean: F) -> F {
        mean
    }

    fn quasi_log_likelihood(data: &Model<Self, F>, regressors: &Array1<F>) -> F {
        Self::log_likelihood(data, regressors)
    }
}

// The true Poisson likelihood includes a factorial of y, which does not contribute to significance calculations.
// This may technically be a quasi-likelihood, and perhaps the concepts should not be distinguished.
impl<D, F> Likelihood<Self, F> for Poisson<D>
where
    D: Unsigned + ToPrimitive,
    F: Float + Lapack,
{
    // specialize LL for logistic regression
    fn log_likelihood(data: &Model<Self, F>, regressors: &Array1<F>) -> F {
        // TODO: this assertion should be a result, or these references should
        // be stored in Fit so they can be checked ahead of time.
        assert_eq!(
            data.x.ncols(),
            regressors.len(),
            "must have same number of explanatory variables as regressors"
        );

        let linear_predictor = data.linear_predictor(regressors);

        let log_like_terms: Array1<F> =
            &data.y * &linear_predictor - linear_predictor.map(|tx| tx.exp());
        let l2_term = data.l2_term(regressors);
        log_like_terms.sum() + l2_term
    }
}
