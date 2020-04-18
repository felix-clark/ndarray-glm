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

impl Glm for Logistic {
    // TODO: this could be relaxed to a float with only mild changes, although
    // it would require checking that 0 <= y <= 1.
    // There should be a domain and a function that maps domain to a float.
    type Domain = bool;

    fn y_float<F: Float>(y: Self::Domain) -> F {
        if y {
            F::one()
        } else {
            F::zero()
        }
    }

    // the link function, logit
    fn link<F: Float>(y: F) -> F {
        Float::ln(y / (F::one() - y))
    }

    // inverse link function, expit
    fn mean<F: Float>(lin_pred: F) -> F {
        (F::one() + (-lin_pred).exp()).recip()
    }

    // var = mu*(1-mu)
    fn variance<F: Float>(mean: F) -> F {
        mean * (F::one() - mean)
    }

    fn quasi_log_likelihood<F>(data: &Model<Self, F>, regressors: &Array1<F>) -> F
    where
        F: Float + Lapack,
    {
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
        let l2_term = data.l2_like_term(regressors);
        log_like_terms.sum() + l2_term
    }
}

#[cfg(test)]
mod tests {
    use crate::{error::RegressionResult, logistic::Logistic, model::ModelBuilder};
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn log_reg() -> RegressionResult<()> {
        let beta = array![0., 1.0];
        let ln2 = f64::ln(2.);
        let data_x = array![[0.], [0.], [ln2], [ln2], [ln2]];
        let data_y = array![true, false, true, true, false];
        let model = ModelBuilder::<Logistic, _>::new(&data_y, &data_x).build()?;
        let fit = model.fit()?;
        dbg!(fit.n_iter);
        assert_abs_diff_eq!(beta, fit.result, epsilon = 0.05 * std::f32::EPSILON as f64);
        // test the significance function
        let significance = fit.z_scores(&model);
        dbg!(significance);
        Ok(())
    }
}
