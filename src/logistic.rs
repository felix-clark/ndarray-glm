//! functions for solving logistic regression

use crate::{
    glm::{Glm, Likelihood},
    model::Model,
};
use ndarray::{Array1, Zip};
use ndarray_linalg::Lapack;
use num_traits::float::Float;

/// trait-based implementation to work towards generalization
pub struct Logistic;

impl<F> Glm<F> for Logistic
where
    F: Float + Lapack,
{
    // TODO: this could be relaxed to a float with only mild changes, although
    // it would require checking that 0 <= y <= 1.
    // There should be a domain and a function that maps domain to a float.
    type Domain = bool;

    fn y_float(y: Self::Domain) -> F {
        if y {
            F::one()
        } else {
            F::zero()
        }
    }

    // the link function, logit
    fn link(y: F) -> F {
        Float::ln(y / (F::one() - y))
    }

    // inverse link function, expit
    fn mean(lin_pred: F) -> F {
        (F::one() + (-lin_pred).exp()).recip()
    }

    // var = mu*(1-mu)
    fn variance(mean: F) -> F {
        mean * (F::one() - mean)
    }

    fn quasi_log_likelihood(data: &Model<Self, F>, regressors: &Array1<F>) -> F {
        Self::log_likelihood(data, regressors)
    }
}

impl<F> Likelihood<Self, F> for Logistic
where
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

        // initialize the log likelihood terms
        let mut log_like_terms: Array1<F> = Array1::zeros(data.y.len());
        Zip::from(&mut log_like_terms)
            .and(&data.y)
            .and(&linear_predictor)
            .apply(|l, &y, &wx| {
                // Both of these expressions are mathematically identical.
                // The distinction is made to avoid under/overflow.
                let (yt, xt) = if wx < F::zero() {
                    (y, wx)
                } else {
                    (F::one() - y, -wx)
                };
                *l = yt * xt - xt.exp().ln_1p()
            });
        let l2_term = data.l2_term(regressors);
        log_like_terms.sum() + l2_term
    }
}
