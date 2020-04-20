//! functions for solving logistic regression

use crate::{
    glm::{Glm, Likelihood, Response},
    link::Link,
    model::Model,
};
use ndarray::{Array1, Zip};
use ndarray_linalg::Lapack;
use num_traits::float::Float;
use std::marker::PhantomData;

/// Logistic regression
pub struct Logistic<L = link::Logit>
where
    L: Link<Logistic<L>>,
{
    _link: PhantomData<L>,
}

/// The logistic response variable must be boolean (at least for now).
impl<L> Response<Logistic<L>> for bool
where
    L: Link<Logistic<L>>,
{
    fn to_float<F: Float>(self) -> F {
        if self {
            F::one()
        } else {
            F::zero()
        }
    }
}
// TODO: We could also allow floats as the domain, however the interface should
// be changed to return an error in case of a failed check for 0 <= y <= 1.

/// Implementation of GLM functionality for logistic regression.
impl<L> Glm for Logistic<L>
where
    L: Link<Logistic<L>>,
{
    type Link = L;

    /// var = mu*(1-mu)
    fn variance<F: Float>(mean: F) -> F {
        mean * (F::one() - mean)
    }

    /// Logistic regression has no additional terms to throw away so this just
    /// uses the full likelihood.
    fn log_like_params<F>(data: &Model<Self, F>, regressors: &Array1<F>) -> F
    where
        F: Float + Lapack,
    {
        Self::log_likelihood(data, regressors)
    }
}

impl<F, L> Likelihood<Self, F> for Logistic<L>
where
    F: Float + Lapack,
    L: Link<Logistic<L>>,
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
        let eta = L::nat_param(linear_predictor);

        // initialize the log likelihood terms
        let mut log_like_terms: Array1<F> = Array1::zeros(data.y.len());
        Zip::from(&mut log_like_terms)
            .and(&data.y)
            .and(&eta)
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

pub mod link {
    //! Link functions for logistic regression
    use super::*;
    use crate::link::{Canonical, Link};
    use num_traits::Float;

    pub struct Logit {}
    impl Canonical for Logit {}
    impl Link<Logistic<Logit>> for Logit {
        fn func<F: Float>(y: Array1<F>) -> Array1<F> {
            y.mapv_into(|y| F::ln(y / (F::one() - y)))
        }
        fn func_inv<F: Float>(lin_pred: Array1<F>) -> Array1<F> {
            lin_pred.mapv_into(|xb| (F::one() + (-xb).exp()).recip())
        }
    }

    // TODO: CLogLog link function. Possibly probit as well although we'd need inverse CDF of normal.
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
        let model = ModelBuilder::<Logistic>::data(&data_y, &data_x).build()?;
        let fit = model.fit()?;
        dbg!(fit.n_iter);
        assert_abs_diff_eq!(beta, fit.result, epsilon = 0.05 * std::f32::EPSILON as f64);
        // test the significance function
        let significance = fit.z_scores(&model);
        dbg!(significance);
        Ok(())
    }
}
