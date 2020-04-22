//! Stores the fit results of the IRLS regression and provides functions that
//! depend on the MLE estimate. These include statistical tests for goodness-of-fit.

use crate::{
    glm::Glm,
    link::{Link, Transform},
    model::Model,
};
use ndarray::{array, Array1, Array2};
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
    data: Model<M, F>,
    /// The parameter values that maximize the likelihood as given by the IRLS regression.
    pub result: Array1<F>,
    /// The value of the likelihood function for the fit result.
    pub model_like: F,
    /// The number of overall iterations taken in the IRLS.
    pub n_iter: usize,
    /// The number of steps taken in the algorithm, which includes step halving.
    pub n_steps: usize,
    /// The number of data points
    n_data: usize,
    /// The number of parameters
    n_par: usize,
}

impl<M, F> Fit<M, F>
where
    M: Glm,
    F: 'static + Float + Lapack,
    F: std::fmt::Debug,
{
    pub fn new(
        data: Model<M, F>,
        result: Array1<F>,
        model_like: F,
        n_iter: usize,
        n_steps: usize,
    ) -> Self {
        // Cache some of these variables that will be used often.
        let n_par = result.len();
        let n_data = data.y.len();
        Self {
            data,
            result,
            model_like,
            n_iter,
            n_steps,
            n_data,
            n_par,
        }
    }

    /// Returns the number of degrees of freedom in the model, i.e. the number
    /// of data points minus the number of parameters.
    pub fn ndf(&self) -> usize {
        self.n_data - self.n_par
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
        let (null_like, ndf) = self.null_like();
        let lr: F = F::from(-2.).unwrap() * (null_like - self.model_like);
        (lr, ndf)
    }

    /// Returns the likelihood given the null model, which fixes all parameters
    /// to zero except the intercept (if it is used). Also returns the
    /// additional degrees of freedom discarded in the null model.
    fn null_like(&self) -> (F, usize) {
        // The average y
        let y_bar: F = self
            .data
            .y
            .mean()
            .expect("Should be able to take average of y values");
        // This is the beta that optimizes the null model. Assuming the
        // intercept is included, it is set to the link function of the mean y.
        // This can be checked by minimizing the likelihood for the null model.
        // The log-likelihood should be the same as the sum of the likelihood
        // using the average of y if L is in the natural exponential form. This
        // could be used to optimize this in the future, if all likelihoods are
        // in the natural exponential form as stated above.
        // let (null_beta, ndf): (Array1<F>, usize) = {
        //     let mut beta = Array1::<F>::zeros(self.result.len());
        //     let mut ndf = beta.len();
        //     if self.data.use_intercept {
        //         beta[0] = M::Link::func(y_bar);
        //         ndf -= 1;
        //     }
        //     (beta, ndf)
        // };
        // let null_like = M::log_like_reg(&self.data, &null_beta);

        // This approach assumes that the likelihood is in the natural
        // exponential form as calculated by Glm::log_like_natural(). If that
        // function is overridden and the values differ significantly, this
        // approach will give incorrect results. If the likelihood has terms
        // non-linear in y, then the likelihood must be calculated for every
        // point rather than averaged.
        let (null_beta0, ndf): (F, usize) = if self.data.use_intercept {
            (M::Link::func(y_bar), self.n_par - 1)
        } else {
            (F::zero(), self.n_par)
        };
        // the natural parameter for a given beta0 = g(y_bar)
        let eta_beta0 = M::Link::nat_param(array![null_beta0]);
        // The null likelihood per observation
        let null_like_one = M::log_like_natural(&array![y_bar], &eta_beta0);
        let null_like_total = F::from(self.data.y.len()).unwrap() * null_like_one;
        (null_like_total, ndf)
    }

    /// Returns the errors in the response variables given the model.
    pub fn errors(&self, data: &Model<M, F>) -> Array1<F> {
        &data.y - &self.expectation(&data.x, data.linear_offset.as_ref())
    }

    /// return the signed Z-score for each regression parameter. This is not a
    /// particularly robust statistic, as it is sensitive to scaling and offsets
    /// of the covariates.
    #[deprecated(
        since = "0.3.0",
        note = "This statistic is not a robust one. To get an analogous
        statistic of the entire fit, take the square root of the likelihood
        ratio test with `lr_test()`."
    )]
    pub fn z_scores(&self) -> Array1<F> {
        // -2 likelihood deviation is asymptotically chi^2 with ndf degrees of freedom.
        let mut chi_sqs: Array1<F> = Array1::zeros(self.result.len());
        // TODO (style): move to (enumerated?) iterator
        for i_like in 0..self.result.len() {
            let mut adjusted = self.result.clone();
            adjusted[i_like] = F::zero();
            let null_like = M::log_like_reg(&self.data, &adjusted);
            let mut chi_sq = F::from(2.).unwrap() * (self.model_like - null_like);
            // This can happen due to FPE
            if chi_sq < F::zero() {
                // this tolerance could need adjusting.
                let tol = F::from(8.).unwrap()
                    * (if self.model_like.abs() > F::one() {
                        self.model_like.abs()
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

    // TODO: score test using Fisher score and information matrix.

    /// Returns the Akaike information criterion for the model fit.
    // TODO: Should an effective number of parameters that takes regularization
    // into acount be considered?
    pub fn aic(&self) -> F {
        F::from(2 * self.result.len()).unwrap() - F::from(2.).unwrap() * self.model_like
    }

    /// Returns the Bayesian information criterion for the model fit.
    // TODO: Also consider the effect of regularization on this statistic.
    // TODO: Wikipedia suggests that the variance should included in the number
    // of parameters for multiple linear regression. Should an additional
    // parameter be included for the dispersion parameter? This question does
    // not affect the difference between two models fit with the methodology in
    // this package.
    pub fn bic(&self) -> F {
        let logn = F::from(self.data.y.len()).unwrap().ln();
        logn * F::from(self.result.len()).unwrap() - F::from(2.).unwrap() * self.model_like
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{logistic::Logistic, model::ModelBuilder, standardize::standardize};
    use anyhow::Result;
    use approx::assert_abs_diff_eq;
    use ndarray::Axis;

    /// Checks if the test statistics are invariant based upon whether the data is standardized.
    #[test]
    fn standardization_invariance() -> Result<()> {
        let data_y = array![true, false, false, true, true, true, true, false, true];
        let data_x = array![-0.5, 0.3, -0.6, 0.2, 0.3, 1.2, 0.8, 0.6, -0.2].insert_axis(Axis(1));
        let data_x_std = standardize(data_x.clone());
        let model = ModelBuilder::<Logistic>::data(&data_y, &data_x).build()?;
        let fit = model.fit()?;
        let model_std = ModelBuilder::<Logistic>::data(&data_y, &data_x_std).build()?;
        let fit_std = model_std.fit()?;
        let (lr, _) = fit.lr_test();
        let (lr_std, _) = fit_std.lr_test();
        assert_abs_diff_eq!(lr, lr_std);
        assert_abs_diff_eq!(fit.aic(), fit_std.aic());
        assert_abs_diff_eq!(fit.bic(), fit_std.bic());
        // These Z-scores are not invariant under data standardization.
        // assert_abs_diff_eq!(fit.z_scores(), fit_std.z_scores());
        Ok(())
    }

    #[test]
    fn null_model() -> Result<()> {
        let data_y = array![true, false, false, true, true];
        let data_x: Array2<f64> = array![[], [], [], [], []];
        let model = ModelBuilder::<Logistic>::data(&data_y, &data_x).build()?;
        let fit = model.fit()?;
        dbg!(fit.n_iter);
        dbg!(&fit.result);
        let (empty_null_like, empty_null_ndf) = fit.null_like();
        assert_eq!(empty_null_ndf, 0);
        dbg!(&fit.model_like);
        let (lr, lr_ndf) = fit.lr_test();
        dbg!(lr, lr_ndf);
        assert_abs_diff_eq!(lr, 0.);

        let data_x = array![[0.5], [-0.2], [0.3], [0.4], [-0.1]];
        let model = ModelBuilder::<Logistic>::data(&data_y, &data_x).build()?;
        let fit_with = model.fit()?;
        dbg!(&fit_with.result);
        assert_abs_diff_eq!(empty_null_like, fit_with.null_like().0);

        Ok(())
    }

    // TODO: Test that the statistics behave sensibly under regularization. The
    // likelihood ratio test should yield a smaller value.
}
