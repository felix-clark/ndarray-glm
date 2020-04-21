//! functions for solving logistic regression

use crate::{
    glm::{Glm, Response},
    link::Link,
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

    /// The log of the partition function for logistic regression. The natural
    /// parameter is the logit of p.
    fn log_partition<F: Float>(nat_par: &Array1<F>) -> F {
        nat_par.mapv(|lp| lp.exp().ln_1p()).sum()
    }

    /// var = mu*(1-mu)
    fn variance<F: Float>(mean: F) -> F {
        mean * (F::one() - mean)
    }

    /// This function is specialized over the default provided by Glm in order
    /// to handle over/underflow issues more precisely.
    fn log_like_natural<F>(y: &Array1<F>, logit_p: &Array1<F>) -> F
    where
        F: Float + Lapack,
    {
        // initialize the log likelihood terms
        let mut log_like_terms: Array1<F> = Array1::zeros(y.len());
        // TODO: This can probably be re-written more elegantly now. We shouldn't need to pre-initialize the result.
        Zip::from(&mut log_like_terms)
            .and(y)
            .and(logit_p)
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
        fn func<F: Float>(y: F) -> F {
            F::ln(y / (F::one() - y))
        }
        fn func_inv<F: Float>(lin_pred: F) -> F {
            (F::one() + (-lin_pred).exp()).recip()
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
        let significance = fit.z_scores();
        dbg!(significance);
        let (lr, _) = fit.lr_test();
        dbg!(&lr);
        dbg!(&lr.sqrt());
        Ok(())
    }
}
