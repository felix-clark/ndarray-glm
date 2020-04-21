//! struct holding the fit result of a regression

use crate::{
    glm::{Glm, Likelihood},
    link::Link,
    model::Model,
};
use ndarray::{Array1, Array2};
use num_traits::Float;
use std::marker::PhantomData;

/// the result of a successful GLM fit
/// TODO: finish generalizing, take ownership of model?
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
    // TODO: M should only need Glm when we have general testing.
    M: Likelihood<M, F>,
    F: 'static + Float,
    F: std::fmt::Debug,
{
    /// Returns the expected value of Y given the input data X. This data need
    /// not be the training data, so an option for linear offsets is provided.
    pub fn expectation(&self, data_x: &Array2<F>, lin_off: Option<&Array1<F>>) -> Array1<F> {
        let lin_pred: Array1<F> = data_x.dot(&self.result);
        let lin_pred: Array1<F> = if let Some(off) = &lin_off {
            lin_pred + *off
        } else {
            lin_pred
        };
        lin_pred.mapv_into(M::Link::func_inv)
    }

    /// Returns the errors in the response variables given the model.
    pub fn errors(&self, data: &Model<M, F>) -> Array1<F> {
        &data.y - &self.expectation(&data.x, data.linear_offset.as_ref())
    }

    /// return the signed Z-score for each regression parameter.
    // TODO: phase this out in terms of more general tests.
    pub fn z_scores(&self, data: &Model<M, F>) -> Array1<F> {
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
                // this tolerance could need adjusting.
                let tol = F::from(8.).unwrap()
                    * (if model_like.abs() > F::one() {
                        model_like.abs()
                    } else {
                        F::one()
                    })
                    * F::epsilon();
                if chi_sq.abs() > tol {
                    eprintln!(
                        "negative chi-squared ({:?}) outside of tolerance ({:?}) for element {}",
                        chi_sq, tol, i_like
                    );
                }
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
