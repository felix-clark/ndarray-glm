//! functions for solving logistic regression

use crate::glm::Glm;
use ndarray::{Array1, Array2, Zip};
use num_traits::float::Float;

/// trait-based implementation to work towards generalization
pub struct Logistic;

impl Glm for Logistic {
    // TODO: this could be relaxed to a float with only mild changes, although
    // it would require checking that 0 <= y <= 1.
    type Domain = bool;

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

    // specialize LL for logistic regression
    fn log_likelihood<F: 'static + Float>(
        data_y: &Array1<bool>,
        data_x: &Array2<F>,
        regressors: &Array1<F>,
    ) -> F {
        // TODO: this assertion should be a result, or these references should
        // be stored in Fit so they can be checked ahead of time.
        assert_eq!(
            data_y.len(),
            data_x.nrows(),
            "must have same number of data points in X and Y"
        );
        assert_eq!(
            data_x.ncols(),
            regressors.len(),
            "must have same number of explanatory variables as regressors"
        );
        // convert y data to floats, although this may not be needed in the
        // future if we change it to use floats.
        let data_y: Array1<F> = data_y.map(|&y| if y { F::one() } else { F::zero() });
        let linear_predictor: Array1<F> = data_x.dot(regressors);
        // initialize the log likelihood terms
        let mut log_like_terms: Array1<F> = Array1::zeros(data_y.len());
        Zip::from(&mut log_like_terms)
            .and(&data_y)
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
