//! Stores the fit results of the IRLS regression and provides functions that
//! depend on the MLE estimate. These include statistical tests for goodness-of-fit.

use crate::{glm::Glm, link::Link, model::Model};
use ndarray::{Array1, Array2};
use ndarray_linalg::{Lapack, Scalar};
use num_traits::Float;

/// the result of a successful GLM fit
pub struct Fit<M, F>
where
    M: Glm,
    F: Float,
{
    /// The data and model specification used in the fit.
    // TODO: This field could likely be made private if Fit had a constructor
    // for Glm::regression() to use.
    pub data: Model<M, F>,
    /// The parameter values that maximize the likelihood as given by the IRLS regression.
    pub result: Array1<F>,
    /// The value of the likelihood function for the fit result.
    pub model_like: F,
    /// The number of overall iterations taken in the IRLS.
    pub n_iter: usize,
    /// The number of steps taken in the algorithm, which includes step halving.
    pub n_steps: usize,
}

impl<M, F> Fit<M, F>
where
    M: Glm,
    F: 'static + Float + Lapack,
    F: std::fmt::Debug,
{
    /// Returns the number of degrees of freedom in the model, i.e. the number
    /// of data points minus the number of parameters.
    pub fn ndf(&self) -> usize {
        self.data.y.len() - self.result.len()
    }

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

    /// Perform a likelihood-ratio test, returning the statistic -2*ln(L_0/L)
    /// where L_0 is the likelihood of the best-fit null model (with no
    /// parameters but the intercept) and L is the likelihood of the fit result.
    /// The number of degrees of freedom of this statistic, equal to the number
    /// of parameters fixed to zero to form the null model, is also returned. By
    /// Wilks' theorem this statistic is asymptotically chi-squared distributed
    /// with this number of degrees of freedom.
    // TODO: Should the effective number of degrees of freedom due to
    // regularization be taken into account? Should the degrees of freedom be a
    // float?
    pub fn lr_test(&self) -> (F, usize) {
        // TODO: The calculation could be made simpler and faster if it were
        // guaranteed that each likelihood function was in the natural
        // exponential form. However, that condition hasn't been enforced -- for
        // instance, the OLS likelihood is the sum of squares.
        let model_like = M::log_like_reg(&self.data, &self.result);
        // This is the beta that optimizes the null model. Assuming the
        // intercept is included, it is set to the link function of the mean y.
        // This can be checked by minimizing the likelihood for the null model.
        // The log-likelihood should be the same as the sum of the likelihood
        // using the average of y if L is in the natural exponential form. This
        // could be used to optimize this in the future, if all likelihoods are
        // in the natural exponential form as stated above.
        let (null_beta, ndf): (Array1<F>, usize) = {
            let mut beta = Array1::<F>::zeros(self.result.len());
            let mut ndf = beta.len();
            if self.data.use_intercept {
                beta[0] = M::Link::func(
                    self.data
                        .y
                        .mean()
                        .expect("Should be able to take average of y values"),
                );
                ndf -= 1;
            }
            (beta, ndf)
        };
        let null_like = M::log_like_reg(&self.data, &null_beta);
        let lr = F::from(-2.).unwrap() * (null_like - model_like);
        (lr, ndf)
    }

    /// Returns the errors in the response variables given the model.
    pub fn errors(&self, data: &Model<M, F>) -> Array1<F> {
        &data.y - &self.expectation(&data.x, data.linear_offset.as_ref())
    }

    /// return the signed Z-score for each regression parameter.
    // TODO: phase this out in terms of more general tests.
    pub fn z_scores(&self) -> Array1<F> {
        let model_like = M::log_like_reg(&self.data, &self.result);
        // -2 likelihood deviation is asymptotically chi^2 with ndf degrees of freedom.
        let mut chi_sqs: Array1<F> = Array1::zeros(self.result.len());
        // TODO (style): move to (enumerated?) iterator
        for i_like in 0..self.result.len() {
            let mut adjusted = self.result.clone();
            adjusted[i_like] = F::zero();
            let null_like = M::log_like_reg(&self.data, &adjusted);
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
        let chis = chi_sqs.map(Scalar::sqrt);
        // return the Z-scores
        signs * chis
    }
}
