//! functions for solving logistic regression

use crate::{
    data::DataConfig,
    glm::{Glm, Likelihood},
};
use ndarray::{Array1, Zip};
use num_traits::float::Float;

/// trait-based implementation to work towards generalization
pub struct Logistic;

impl Glm for Logistic {
    // TODO: this could be relaxed to a float with only mild changes, although
    // it would require checking that 0 <= y <= 1.
    // type Domain = bool;

    // the link function, logit
    fn link<F: Float>(y: F) -> F {
        F::ln(y / (F::one() - y))
    }

    // inverse link function, expit
    fn mean<F: Float>(lin_pred: F) -> F {
        (F::one() + (-lin_pred).exp()).recip()
    }

    // var = mu*(1-mu)
    fn variance<F: Float>(mean: F) -> F {
        mean * (F::one() - mean)
    }
}

impl Likelihood for Logistic {
    // specialize LL for logistic regression
    fn log_likelihood<F: 'static + Float>(data: &DataConfig<F>, regressors: &Array1<F>) -> F {
        // TODO: this assertion should be a result, or these references should
        // be stored in Fit so they can be checked ahead of time.
        assert_eq!(
            data.x.ncols(),
            regressors.len(),
            "must have same number of explanatory variables as regressors"
        );
        // FIXME: add linear offsets here
        let linear_predictor: Array1<F> = data.x.dot(regressors);
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
        log_like_terms.sum()
    }
}
