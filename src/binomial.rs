//! Regression with a binomial response function. The N parameter must be known ahead of time.
use crate::{glm::Glm, model::Model};
use ndarray::Array1;
use ndarray_linalg::Lapack;
use num_traits::{Float, ToPrimitive, Unsigned};

/// Binomial regression with a fixed N.
pub struct Binomial<const N: u32> {
    _n: std::marker::PhantomData<u32>,
    // N: usize,
}

impl<U, F, const N: u32> Glm<F> for Binomial<N>
where
    U: Unsigned + ToPrimitive,
    F: Float + Lapack,
{
    type Domain = U;

    fn y_float(y: Self::Domain) -> F {
        F::from(y).unwrap()
    }

    fn link(y: F) -> F {
        Float::ln(y / (F::from(N).unwrap() - y))
    }

    fn mean(lin_pred: F) -> F {
        F::from(N).unwrap() * lin_pred
    }

    fn quasi_log_likelihood(data: &Model<Self, F>, regressors: &Array1<F>) -> F {
        todo!("implement binomial likelihood");
    }
}
