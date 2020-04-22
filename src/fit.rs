//! Stores the fit results of the IRLS regression and provides functions that
//! depend on the MLE estimate. These include statistical tests for goodness-of-fit.

use crate::{
    error::RegressionResult,
    glm::Glm,
    link::{Link, Transform},
    model::Model,
};
use ndarray::{array, Array1, Array2, ArrayView1};
use ndarray_linalg::{InverseHInto, Lapack, Scalar};
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
    // /// The estimated covariance matrix of the parameters. Since the calculation
    // /// requires a matrix inversion, it is computed only when needed and the
    // /// value is cached. Access through the `covariance()` function. This is not
    // /// yet implemented.
    // cov: RefCell<Option<Array2<F>>>,
}

impl<M, F> Fit<M, F>
where
    M: Glm,
    F: 'static + Float + Lapack + Scalar,
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
        // The model likelihood should include regularization terms and there
        // shouldn't be any in the null model with all non-intercept parameters
        // set to zero.
        let (null_like, ndf) = self.null_like();
        eprintln!("null like in LR test: {}", null_like);
        eprintln!("model like in LR test: {}", self.model_like);
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
        // let (null_beta_slow, _ndf): (Array1<F>, usize) = {
        //     let mut beta = Array1::<F>::zeros(self.result.len());
        //     let mut ndf = beta.len();
        //     if self.data.use_intercept {
        //         beta[0] = M::Link::func(y_bar);
        //         ndf -= 1;
        //     }
        //     (beta, ndf)
        // };
        // let null_like_slow = M::log_like_reg(&self.data, &null_beta_slow);

        // This approach assumes that the likelihood is in the natural
        // exponential form as calculated by Glm::log_like_natural(). If that
        // function is overridden and the values differ significantly, this
        // approach will give incorrect results. If the likelihood has terms
        // non-linear in y, then the likelihood must be calculated for every
        // point rather than averaged.
        // If the intercept is allowed to maximize the likelihood, the natural
        // parameter is equal to the link of the expectation. Otherwise it is
        // the transformation function of zero.
        let (nat_par, ndf) = if self.data.use_intercept {
            (array![M::Link::func(y_bar)], self.n_par - 1)
        } else {
            (M::Link::nat_param(array![F::zero()]), self.n_par)
        };
        // The null likelihood per observation
        let null_like_one = M::log_like_natural(&array![y_bar], &nat_par);
        let null_like_total = F::from(self.n_data).unwrap() * null_like_one;
        (null_like_total, ndf)
    }

    /// The covariance matrix estimated by the Fisher information and the
    /// dispersion parameter. The value will be cached to avoid repeating the
    /// potentially expensive matrix inversion, but this is not yet implemented.
    // TODO: This will also need to be fixed up for the weighted case.
    pub fn covariance(&self) -> RegressionResult<Array2<F>> {
        let lin_pred: Array1<F> = self.data.linear_predictor(&self.result);
        // let mu: Array1<F> = lin_pred.mapv(M::Link::func_inv);
        let mu: Array1<F> = M::mean(&lin_pred);
        // let mu: Array1<F> = self.expectation(&self.data.x, self.data.linear_offset.as_ref());

        let var_diag: Array1<F> = mu.mapv_into(M::variance);
        // adjust the variance for non-canonical link functions
        let eta_d = M::Link::d_nat_param(&lin_pred);
        let adj_var: Array1<F> = &eta_d * &var_diag * eta_d;
        // calculate the fisher matrix
        let fisher: Array2<F> = (&self.data.x.t() * &adj_var).dot(&self.data.x);
        // Regularize the fisher matrix
        let fisher_reg: Array2<F> = (*self.data.reg).irls_mat(fisher, &self.result);
        let cov = fisher_reg.invh_into()?;
        // the covariance must be multiplied by the dispersion parameter.
        let phi = self.dispersion();
        Ok(cov.mapv_into(|c| phi * c))
    }

    /// Returns the deviance of the fit: twice the difference between the
    /// saturated likelihood and the model likelihood. Asymptotically this fits
    /// a chi-squared distribution with `self.ndf()` degrees of freedom.
    // TODO: This is likely sensitive to regularization because the saturated
    // model is not regularized but the model likelihood is. Perhaps this can be
    // accounted for with an effective number of degrees of freedom.
    // TODO: Should this include a term for the dispersion parameter? Probably,
    // as these likelihoods do not include it.
    pub fn deviance(&self) -> F {
        F::from(2.).unwrap() * (M::log_like_sat(&self.data.y) - self.model_like)
    }

    /// Estimate the dispersion parameter through the method of moments.
    // NOTE: This appears to be quite similar to the score test.
    // TODO: This will need to be fixed up for weighted regression, including the weights in the covariance matrix.
    pub fn dispersion(&self) -> F {
        let ndf: F = F::from(self.ndf()).unwrap();
        let mu: Array1<F> = self.expectation(&self.data.x, self.data.linear_offset.as_ref());
        let errors: Array1<F> = &self.data.y - &mu;
        let var_diag: Array1<F> = mu.mapv(M::variance);
        (&errors * &var_diag.mapv_into(|v| (v + F::epsilon()).recip()) * errors).sum() / ndf
    }

    /// Returns the errors in the response variables given the model.
    pub fn errors(&self, data: &Model<M, F>) -> Array1<F> {
        &data.y - &self.expectation(&data.x, data.linear_offset.as_ref())
    }

    /// Returns the signed square root of the Wald test statistic for each parameter.
    pub fn wald_z(&self) -> RegressionResult<Array1<F>> {
        let par_cov = self.covariance()?;
        let par_variances: ArrayView1<F> = par_cov.diag();
        Ok(&self.result / &par_variances.mapv(Float::sqrt))
    }

    /// return the signed Z-score for each regression parameter. This is not a
    /// particularly robust statistic, as it is sensitive to scaling and offsets
    /// of the covariates.
    #[deprecated(
        since = "0.3.0",
        note = "This statistic is not a robust one. To get an analogous
        statistic use `wald_z()`."
    )]
    pub fn z_scores(&self) -> Array1<F> {
        // -2 likelihood deviation is asymptotically chi^2 with ndf degrees of freedom.
        let mut chi_sqs: Array1<F> = Array1::zeros(self.result.len());
        // TODO (style): move to (enumerated?) iterator
        for i_like in 0..self.result.len() {
            let mut adjusted = self.result.clone();
            adjusted[i_like] = F::zero();
            let null_like = M::log_like_reg(&self.data, &adjusted);
            eprintln!("Null like in Z-scores: {}", null_like);
            eprintln!("Model like in Z-scores: {}", self.model_like);
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
    use crate::{
        linear::Linear, logistic::Logistic, model::ModelBuilder, standardize::standardize,
    };
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
        assert_abs_diff_eq!(fit.deviance(), fit_std.deviance());
        // The Wald statistic of the intercept term is not invariant under a
        // linear transformation of the data, but the parameter part seems to
        // be, at least for single-component data.
        assert_abs_diff_eq!(
            fit.wald_z()[1],
            fit_std.wald_z()[1],
            epsilon = 0.01 * std::f32::EPSILON as f64
        );

        dbg!(fit.deviance());
        dbg!(fit.deviance() / fit.ndf() as f64);
        dbg!(lr);
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

    #[test]
    fn null_like_logistic() -> Result<()> {
        // 6 true and 4 false for y_bar = 0.6.
        let data_y = array![true, true, true, true, true, true, false, false, false, false];
        let ybar: f64 = 0.6;
        let data_x = array![0.4, 0.2, 0.5, 0.1, 0.6, 0.7, 0.3, 0.8, -0.1, 0.1].insert_axis(Axis(1));
        let model = ModelBuilder::<Logistic>::data(&data_y, &data_x).build()?;
        let fit = model.fit()?;
        let target_null_like = fit
            .data
            .y
            .mapv(|y| {
                let eta = (ybar / (1. - ybar)).ln();
                y * eta - eta.exp().ln_1p()
            })
            .sum();
        assert_abs_diff_eq!(fit.null_like().0, target_null_like);
        Ok(())
    }

    // check the null likelihood for the case where it can be counted exactly.
    #[test]
    fn null_like_linear() -> Result<()> {
        let data_y = array![0.3, -0.1, 0.5, 0.7, 0.2, 1.3, 1.1, 0.2];
        let data_x = array![0.6, 2.1, 0.4, -3.2, 0.7, 0.1, -0.3, 0.5].insert_axis(Axis(1));
        let ybar: f64 = data_x.mean().unwrap();
        let model = ModelBuilder::<Linear>::data(&data_y, &data_x).build()?;
        let fit = model.fit()?;
        let target_null_like = data_y.mapv(|y| y * ybar - 0.5 * ybar * ybar).sum();
        let fit_null_like = fit.null_like();
        assert_eq!(2. * (&fit.model_like - fit_null_like.0), fit.lr_test().0);
        dbg!(fit.lr_test().0);
        dbg!(2. * (&fit.model_like - target_null_like));
        assert_eq!(fit_null_like.1, 1);
        assert_abs_diff_eq!(fit_null_like.0, target_null_like);
        Ok(())
    }

    // check the null likelihood where there is no dependence on the X data.
    #[test]
    fn null_like_logistic_nodep() -> Result<()> {
        let data_y = array![true, true, false, false, true, false, false, true];
        let data_x = array![0.4, 0.2, 0.4, 0.2, 0.7, 0.7, -0.1, -0.1].insert_axis(Axis(1));
        let model = ModelBuilder::<Logistic>::data(&data_y, &data_x).build()?;
        let fit = model.fit()?;
        let (lr, _) = fit.lr_test();
        assert_abs_diff_eq!(lr, 0.);
        Ok(())
    }
    // TODO: Test that the statistics behave sensibly under regularization. The
    // likelihood ratio test should yield a smaller value.
}
