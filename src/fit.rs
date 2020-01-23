//! struct holding the fit result of a regression

use crate::{
    data::DataConfig,
    glm::{Glm, Likelihood},
};

use ndarray::Array1;
use num_traits::Float;
use std::marker::PhantomData;

/// the result of a successful GLM fit (logistic for now)
/// TODO: finish generalizing, take ownership of Y and X data?
#[derive(Debug)]
pub struct Fit<M, F>
where
    M: Glm,
    F: Float,
{
    // we aren't now storing any type that uses the model type
    pub model: PhantomData<M>,
    // the parameter values that maximize the likelihood
    pub result: Array1<F>,
    // number of data points minus number of free parameters
    pub ndf: usize,
    // the number of iterations taken
    pub n_iter: usize,
}

impl<M, F> Fit<M, F>
where
    M: Likelihood,
    F: 'static + Float,
{
    /// return the signed Z-score for each regression parameter.
    pub fn z_scores(&self, data: &DataConfig<F>) -> Array1<F> {
        let model_like = M::log_likelihood(&data, &self.result);
        // -2 likelihood deviation is asymptotically chi^2 with ndf degrees of freedom.
        let mut chi_sqs: Array1<F> = Array1::zeros(self.result.len());
        // TODO (style): move to (enumerated?) iterator
        for i_like in 0..self.result.len() {
            let mut adjusted = self.result.clone();
            adjusted[i_like] = F::zero();
            let null_like = M::log_likelihood(&data, &adjusted);
            let mut chi_sq = F::from(2.).unwrap() * (model_like - null_like);
            // This can happen due to FPE
            if chi_sq < F::zero() {
                assert!(
                    chi_sq.abs()
                        <= (if model_like.abs() > F::one() {
                            model_like.abs()
                        } else {
                            F::one()
                        }) * F::epsilon(),
                    "negative chi-squared outside of tolerance"
                );
                chi_sq = F::zero();
            }
            chi_sqs[i_like] = chi_sq;
        }
        let signs = self.result.mapv(F::signum);
        let chis = chi_sqs.mapv_into(F::sqrt);
        // return the Z-scores
        signs * chis
    }
}
