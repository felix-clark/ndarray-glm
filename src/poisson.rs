//! Model for Poisson regression

use crate::{
    glm::{Glm, Likelihood, Response},
    link::Link,
    model::Model,
};
use ndarray::Array1;
use ndarray_linalg::Lapack;
use num_traits::{Float, ToPrimitive, Unsigned};
use std::marker::PhantomData;

/// Poisson regression over an unsigned integer type.
pub struct Poisson<L = link::Log>
where
    L: Link<Poisson<L>>,
{
    _link: PhantomData<L>,
}

/// Poisson variables can be any unsigned integer.
impl<U, L> Response<Poisson<L>> for U
where
    U: Unsigned + ToPrimitive,
    L: Link<Poisson<L>>,
{
    fn to_float<F: Float>(self) -> F {
        F::from(self).unwrap()
    }
}
// TODO: A floating point response for Poisson might also be do-able.

impl<L> Glm for Poisson<L>
where
    L: Link<Poisson<L>>,
{
    type Link = L;

    /// The variance of a Poisson variable is equal to its mean.
    fn variance<F: Float>(mean: F) -> F {
        mean
    }

    fn log_like_natural<F>(y: &Array1<F>, log_lambda: &Array1<F>) -> F
    where
        F: Float + Lapack,
    {
        let log_like_terms: Array1<F> = y * log_lambda - log_lambda.mapv(|tx| tx.exp());
        log_like_terms.sum()
    }
}

// The true Poisson likelihood includes a factorial of y, which does not contribute to significance calculations.
// This may technically be a quasi-likelihood, and perhaps the concepts should not be distinguished.
impl<F, L> Likelihood<Self, F> for Poisson<L>
where
    F: Float + Lapack,
    L: Link<Poisson<L>>,
{
    // specialize LL for poisson regression
    // TODO: Phase this trait out entirely.
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
        Self::log_like_natural(&data.y, &eta)
    }
}

pub mod link {
    //! Link functions for Poisson regression
    use super::Poisson;
    use crate::link::{Canonical, Link};
    use num_traits::Float;

    /// The canonical link function of the Poisson response is the logarithm.
    pub struct Log {}
    impl Canonical for Log {}
    impl Link<Poisson<Log>> for Log {
        fn func<F: Float>(y: F) -> F {
            y.ln()
        }
        fn func_inv<F: Float>(lin_pred: F) -> F {
            lin_pred.exp()
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{error::RegressionResult, model::ModelBuilder, poisson::Poisson};
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array1};

    #[test]
    fn poisson_reg() -> RegressionResult<()> {
        let ln2 = f64::ln(2.);
        let beta = array![0., ln2, -ln2];
        let data_x = array![[1., 0.], [1., 1.], [0., 1.], [0., 1.]];
        let data_y: Array1<u32> = array![2, 1, 0, 1];
        let model = ModelBuilder::<Poisson>::data(&data_y, &data_x)
            .max_iter(10)
            .build()?;
        let fit = model.fit()?;
        dbg!(fit.n_iter);
        assert_abs_diff_eq!(beta, fit.result, epsilon = std::f32::EPSILON as f64);
        Ok(())
    }
}
