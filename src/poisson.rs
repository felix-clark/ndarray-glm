//! Model for Poisson regression

use crate::{
    glm::{Glm, Likelihood, Response},
    model::Model,
};
use ndarray::Array1;
use ndarray_linalg::Lapack;
use num_traits::{Float, ToPrimitive, Unsigned};
use std::marker::PhantomData;

/// Poisson regression over an unsigned integer type.
pub struct Poisson {}

/// Poisson variables can be any unsigned integer.
impl<U> Response<Poisson> for U
where
    U: Unsigned + ToPrimitive,
{
    fn to_float<F: Float>(self) -> F {
        F::from(self).unwrap()
    }
}
// TODO: A floating point response for Poisson might also be do-able.

// impl<L> Glm for Poisson<L>
impl Glm for Poisson
// where
//     D: Unsigned + ToPrimitive,
{
    /// the link function, canonically the logarithm.
    fn link<F: Float>(y: F) -> F {
        Float::ln(y)
    }

    /// inverse link function, canonically exponential.
    fn mean<F: Float>(lin_pred: F) -> F {
        lin_pred.exp()
    }

    /// The variance of a Poisson variable is equal to its mean.
    fn variance<F: Float>(mean: F) -> F {
        mean
    }

    fn quasi_log_likelihood<F>(data: &Model<Self, F>, regressors: &Array1<F>) -> F
    where
        F: Float + Lapack,
    {
        Self::log_likelihood(data, regressors)
    }
}

// The true Poisson likelihood includes a factorial of y, which does not contribute to significance calculations.
// This may technically be a quasi-likelihood, and perhaps the concepts should not be distinguished.
impl<F> Likelihood<Self, F> for Poisson
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

        let log_like_terms: Array1<F> =
            &data.y * &linear_predictor - linear_predictor.map(|tx| tx.exp());
        let l2_term = data.l2_like_term(regressors);
        log_like_terms.sum() + l2_term
    }
}

#[cfg(test)]
mod tests {
    use crate::{error::RegressionResult, model::ModelBuilder, poisson::Poisson};
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn poisson_reg() -> RegressionResult<()> {
        let ln2 = f64::ln(2.);
        let beta = array![0., ln2, -ln2];
        let data_x = array![[1., 0.], [1., 1.], [0., 1.], [0., 1.]];
        let data_y = array![2, 1, 0, 1];
        let model = ModelBuilder::<Poisson, _>::new(&data_y, &data_x)
            .max_iter(10)
            .build()?;
        let fit = model.fit()?;
        dbg!(fit.n_iter);
        assert_abs_diff_eq!(beta, fit.result, epsilon = std::f32::EPSILON as f64);
        Ok(())
    }
}
