//! Stores the fit results of the IRLS regression and provides functions that
//! depend on the MLE estimate. These include statistical tests for goodness-of-fit.

pub mod options;
use crate::{
    error::RegressionResult,
    glm::Glm,
    link::{Link, Transform},
    model::{Dataset, Model},
    num::Float,
    Linear,
};
use ndarray::{array, Array1, Array2, ArrayBase, ArrayView1, Data, Ix2};
use ndarray_linalg::InverseHInto;
use options::FitOptions;
use std::{cell::{Ref, RefCell}, marker::PhantomData};

/// the result of a successful GLM fit
pub struct Fit<'a, M, F>
where
    M: Glm,
    F: Float,
{
    model: PhantomData<M>,
    /// The data and model specification used in the fit.
    data: &'a Dataset<F>,
    /// Whether the intercept covariate is used
    use_intercept: bool,
    /// The parameter values that maximize the likelihood as given by the IRLS regression.
    pub result: Array1<F>,
    /// The options used for this fit.
    pub options: FitOptions<F>,
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
    /// The estimated covariance matrix of the parameters. Since the calculation
    /// requires a matrix inversion, it is computed only when needed and the
    /// value is cached. Access through the `covariance()` function.
    cov: RefCell<Option<Array2<F>>>,
    /// The likelihood and parameters for the null model.
    null_model: RefCell<Option<(F, Array1<F>)>>,
}

impl<'a, M, F> Fit<'a, M, F>
where
    M: Glm,
    F: 'static + Float,
{
    pub(crate) fn new(
        data: &'a Dataset<F>,
        use_intercept: bool,
        result: Array1<F>,
        options: FitOptions<F>,
        model_like: F,
        n_iter: usize,
        n_steps: usize,
    ) -> Self {
        if !model_like.is_nan()
            && model_like != M::log_like_reg(data, &result, options.reg.as_ref())
        {
            eprintln!("Model likelihood does not match result! There is an error in the GLM fitting code.");
            dbg!(&result);
            dbg!(model_like);
            dbg!(n_iter);
            dbg!(n_steps);
        }
        // Cache some of these variables that will be used often.
        let n_par = result.len();
        let n_data = data.y.len();
        Self {
            model: PhantomData,
            data,
            use_intercept,
            result,
            options,
            model_like,
            n_iter,
            n_steps,
            n_data,
            n_par,
            cov: RefCell::new(None),
            null_model: RefCell::new(None),
        }
    }

    /// Returns the number of degrees of freedom in the model, i.e. the number
    /// of data points minus the number of parameters. Not to be confused with
    /// `test_ndf()`, the degrees of freedom in the statistical tests of the
    /// fit.
    pub fn ndf(&self) -> usize {
        self.n_data - self.n_par
    }

    /// Returns the expected value of Y given the input data X. This data need
    /// not be the training data, so an option for linear offsets is provided.
    /// Panics if the number of covariates in the data matrix is not consistent
    /// with the training set. The data matrix may need to be padded by ones if
    /// it is not part of a Model. The `utility::one_pad()` function facilitates
    /// this.
    pub fn expectation<S>(
        &self,
        data_x: &ArrayBase<S, Ix2>,
        lin_off: Option<&Array1<F>>,
    ) -> Array1<F>
    where
        S: Data<Elem = F>,
    {
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
    /// of parameters fixed to zero to form the null model, is `test_ndf()`. By
    /// Wilks' theorem this statistic is asymptotically chi-squared distributed
    /// with this number of degrees of freedom.
    // TODO: Should the effective number of degrees of freedom due to
    // regularization be taken into account? Should the degrees of freedom be a
    // float?
    pub fn lr_test(&self) -> F {
        // The model likelihood should include regularization terms and there
        // shouldn't be any in the null model with all non-intercept parameters
        // set to zero.
        let null_like = self.null_like();
        F::from(-2.).unwrap() * (null_like - self.model_like)
    }

    /// Perform a likelihood-ratio test against a general alternative model, not
    /// necessarily a null model. The alternative model is regularized the same
    /// way that the regression resulting in this fit was. The degrees of
    /// freedom cannot be generally inferred.
    pub fn lr_test_against(&self, alternative: &Array1<F>) -> F {
        let alt_like = M::log_like_reg(&self.data, &alternative, self.options.reg.as_ref());
        F::from(2.).unwrap() * (self.model_like - alt_like)
    }

    /// Return the likelihood and intercept for the null model. Since this can
    /// require an additional regression, the values are cached.
    fn null_model_fit(&self) -> (F, Array1<F>) {
        // TODO: make a result instead of allowing a potential panic in the borrow.
        if self.null_model.borrow().is_none() {
            let (null_like, null_intercept): (F, Array1<F>) = match &self.data.linear_offset {
                None => {
                    // If there is no linear offset, the natural parameter is
                    // identical for all observations so it is sufficient to
                    // calculate the null likelihood for a single point with y equal
                    // to the average.
                    // The average y
                    let y_bar: F = self
                        .data
                        .y
                        .mean()
                        .expect("Should be able to take average of y values");
                    // This approach assumes that the likelihood is in the natural
                    // exponential form as calculated by Glm::log_like_natural(). If that
                    // function is overridden and the values differ significantly, this
                    // approach will give incorrect results. If the likelihood has terms
                    // non-linear in y, then the likelihood must be calculated for every
                    // point rather than averaged.
                    // If the intercept is allowed to maximize the likelihood, the natural
                    // parameter is equal to the link of the expectation. Otherwise it is
                    // the transformation function of zero.
                    let intercept: F = if self.use_intercept {
                        M::Link::func(y_bar)
                    } else {
                        F::zero()
                    };
                    // this is a length-one array. This works because the
                    // likelihood contribution is the same for all observations.
                    let nat_par = M::Link::nat_param(array![intercept]);
                    // The null likelihood per observation
                    let null_like_one: F = M::log_like_natural(y_bar, nat_par[0]);
                    // just multiply the average likelihood by the number of data points, since every term is the same.
                    let null_like_total = F::from(self.n_data).unwrap() * null_like_one;
                    let null_params: Array1<F> = {
                        let mut par = Array1::<F>::zeros(self.n_par);
                        par[0] = intercept;
                        par
                    };
                    (null_like_total, null_params)
                }
                Some(off) => {
                    if self.use_intercept {
                        // If there are linear offsets and the intercept is allowed
                        // to be free, there is not a major simplification and the
                        // model needs to be re-fit.
                        // the X data is a single column of ones. Since this model
                        // isn't being created by the ModelBuilder, the X data
                        // has to be automatically padded with ones.
                        let data_x_null = Array2::<F>::ones((self.n_data, 1));
                        let null_model = Model {
                            model: std::marker::PhantomData::<M>,
                            data: Dataset::<F> {
                                y: self.data.y.clone(),
                                x: data_x_null,
                                linear_offset: Some(off.clone()),
                                weights: self.data.weights.clone(),
                                // If we are here it is because an intercept is needed.
                                hat: RefCell::new(None),
                            },
                            use_intercept: true,
                        };
                        // TODO: Make this function return an error, although it's
                        // difficult to imagine this case happening.
                        // TODO: Should the tolerance of this fit be stricter?
                        // The intercept should not be regularized
                        let null_fit = null_model
                            .fit_options()
                            // There shouldn't be too much trouble fitting this
                            // single-parameter fit, but there shouldn't be harm in
                            // using the same maximum as in the original model.
                            .max_iter(self.options.max_iter)
                            .fit()
                            .expect("Could not fit null model!");
                        let null_params: Array1<F> = {
                            let mut par = Array1::<F>::zeros(self.n_par);
                            // there is only one parameter in this fit.
                            par[0] = null_fit.result[0];
                            par
                        };
                        (null_fit.model_like, null_params)
                    } else {
                        // If the intercept is fixed to zero, then no minimization is
                        // required. The natural parameters are directly known in terms
                        // of the linear offset. The likelihood must still be summed
                        // over all observations, since they have different offsets.
                        let nat_par = M::Link::nat_param(off.clone());
                        let null_like = ndarray::Zip::from(&self.data.y)
                            .and(&nat_par)
                            .map_collect(|&y, &eta| M::log_like_natural(y, eta))
                            .sum();
                        let null_params = Array1::<F>::zeros(self.n_par);
                        (null_like, null_params)
                    }
                }
            };
            *self.null_model.borrow_mut() = Some((null_like, null_intercept));
        }
        self.null_model
            .borrow()
            .as_ref()
            .expect("the null model should be cached now")
            .clone()
    }

    /// Returns the likelihood given the null model, which fixes all parameters
    /// to zero except the intercept (if it is used). A total of `test_ndf()`
    /// parameters are constrained.
    fn null_like(&self) -> F {
        let (null_like, _) = self.null_model_fit();
        null_like
    }

    /// The covariance matrix estimated by the Fisher information and the
    /// dispersion parameter. The matrix is cached to avoid repeating the
    /// potentially expensive matrix inversion.
    // TODO: This will also need to be fixed up for the weighted case.
    pub fn covariance(&self) -> RegressionResult<Ref<Array2<F>>> {
        if self.cov.borrow().is_none() {
            let fisher_reg = self.fisher(&self.result);
            // the covariance must be multiplied by the dispersion parameter.
            // However it should be the likelihood dispersion parameter, not the
            // estimated one. In logistic regression, for instance, the dispersion
            // parameter is identically 1.
            // let phi = self.dispersion();
            let cov = fisher_reg.invh_into()?;
            *self.cov.borrow_mut() = Some(cov);
        }
        Ok(Ref::map(self.cov.borrow(), |x| x.as_ref().unwrap()))
    }

    /// Returns the deviance of the fit: twice the difference between the
    /// saturated likelihood and the model likelihood. Asymptotically this fits
    /// a chi-squared distribution with `self.ndf()` degrees of freedom.
    /// Note that the regularized likelihood is used here.
    // TODO: This is likely sensitive to regularization because the saturated
    // model is not regularized but the model likelihood is. Perhaps this can be
    // accounted for with an effective number of degrees of freedom.
    // TODO: Should this include a term for the dispersion parameter? Probably,
    // as these likelihoods do not include it.
    pub fn deviance(&self) -> F {
        // Note that this must change if the GLM likelihood subtracts the
        // saturated one already.
        F::from(2.).unwrap() * (self.data.y.mapv(M::log_like_sat).sum() - self.model_like)
    }

    /// Estimate the dispersion parameter (typically denoted `phi`)  which relates the variance
    /// of the `y` values with the variance of the response distribution: `Var[y] = phi *
    /// Var[mu]`.
    /// The method of moments is used to approximate phi, which is exact in the linear OLS case
    /// but only an approximation in general.
    /// Equal to the sum of `(y_i - mu_i)^2 / V(mu_i)` divided by the degrees of freedom (`n - p`).
    /// For OLS linear regression, this provides an estimate of `sigma^2`; with no covariates
    /// it is equal to the sample variance.
    /// In logistic and Poisson regression `phi > 1` indicates overdispersion; that is, a larger
    /// variance in the `y` data than is accouned for in the response distribution.
    // NOTE: This appears to be quite similar to the score test.
    // TODO: This will need to be fixed up for weighted regression, including
    // the weights in the covariance matrix.
    pub fn dispersion(&self) -> F {
        let ndf: F = F::from(self.ndf()).unwrap();
        let mu: Array1<F> = self.expectation(&self.data.x, self.data.linear_offset.as_ref());
        let errors: Array1<F> = &self.data.y - &mu;
        let var_diag: Array1<F> = mu.mapv(M::variance);
        (&errors * &var_diag.mapv_into(|v| (v + F::epsilon()).recip()) * errors).sum() / ndf
    }

    /// Returns the errors in the response variables for the data passed as an
    /// argument given the current model fit.
    fn errors(&self, data: &Dataset<F>) -> Array1<F> {
        &data.y - &self.expectation(&data.x, data.linear_offset.as_ref())
    }

    /// Returns the fisher information (the negative hessian of the likelihood)
    /// at the parameter values given. The regularization is included.
    pub fn fisher(&self, params: &Array1<F>) -> Array2<F> {
        let lin_pred: Array1<F> = self.data.linear_predictor(params);
        let mu: Array1<F> = M::mean(&lin_pred);
        let var_diag: Array1<F> = mu.mapv_into(M::variance);
        // adjust the variance for non-canonical link functions
        let eta_d = M::Link::d_nat_param(&lin_pred);
        let adj_var: Array1<F> = &eta_d * &var_diag * eta_d;
        // calculate the fisher matrix
        let fisher: Array2<F> = (&self.data.x.t() * &adj_var).dot(&self.data.x);
        // Regularize the fisher matrix
        self.options.reg.as_ref().irls_mat(fisher, params)
    }

    /// Returns the deviance residuals for each point in the training data.
    /// Equal to `sign(y-E[y|x])*sqrt(-2*(L[y|x] - L_sat[y]))`.
    /// This is usually a better choice for non-linear models.
    /// NaNs might be possible if L[y|x] > L_sat[y] due to floating-point operations. These are
    /// not checked or clipped right now.
    pub fn residuals_deviance(&self) -> Array1<F> {
        let signs = self.residuals_response().mapv_into(F::signum);
        let ll_terms: Array1<F> = M::log_like_terms(&self.data, &self.result);
        let ll_sat: Array1<F> = self.data.y.mapv(M::log_like_sat);
        let neg_two = F::from(-2.).unwrap();
        let dev: Array1<F> =
            (ll_terms - ll_sat).mapv_into(|l| num_traits::Float::sqrt(neg_two * l));
        signs * dev
    }

    /// Returns the Pearson or standardized residuals for each point in the training data.
    /// This is equal to `(y - E[y|x])/sqrt(Var[y|x])`, given the fitted model.
    pub fn residuals_pearson(&self) -> Array1<F> {
        let mu: Array1<F> = self.expectation(&self.data.x, self.data.linear_offset.as_ref());
        let residuals = &self.data.y - &mu;
        let var_diag: Array1<F> = mu.mapv_into(M::variance);
        let std: Array1<F> = var_diag.mapv_into(num_traits::Float::sqrt);
        residuals / std
    }

    /// Returns the raw residuals, or fitting deviation, for each data point in the fit; that is,
    /// the difference y - E[y|x] where the expectation value is the y value predicted by the model
    /// given x.
    pub fn residuals_response(&self) -> Array1<F> {
        self.errors(&self.data)
    }

    /// Returns the score function (the gradient of the likelihood) at the
    /// parameter values given. It should be zero within FPE at the minimized
    /// result.
    pub fn score(&self, params: &Array1<F>) -> Array1<F> {
        // This represents the predictions given the input parameters, not the
        // fit parameters.
        let lin_pred: Array1<F> = self.data.linear_predictor(&params);
        let mu: Array1<F> = M::mean(&lin_pred);
        // adjust for non-canonical link functions.
        let eta_d = M::Link::d_nat_param(&lin_pred);
        let score_unreg = self.data.x.t().dot(&(eta_d * (&self.data.y - &mu)));
        self.options.reg.as_ref().gradient(score_unreg, &params)
    }

    /// Returns the score test statistic. This statistic is asymptotically
    /// chi-squared distributed with `test_ndf()` degrees of freedom.
    pub fn score_test(&self) -> RegressionResult<F> {
        let (_, null_params) = self.null_model_fit();
        self.score_test_against(null_params)
    }

    /// Returns the score test statistic compared to another set of model
    /// parameters, not necessarily a null model. The degrees of freedom cannot
    /// be generally inferred.
    pub fn score_test_against(&self, alternative: Array1<F>) -> RegressionResult<F> {
        let score_alt = self.score(&alternative);
        let fisher_alt = self.fisher(&alternative);
        // The is not the same as the cached covariance matrix since it is
        // evaluated at the null parameters.
        let inv_fisher_alt = fisher_alt.invh_into()?;
        Ok(score_alt.t().dot(&inv_fisher_alt.dot(&score_alt)))
    }

    /// The degrees of freedom for the likelihood ratio test, the score test,
    /// and the Wald test. Not to be confused with `ndf()`, the degrees of
    /// freedom in the model fit.
    pub fn test_ndf(&self) -> usize {
        if self.use_intercept {
            self.n_par - 1
        } else {
            self.n_par
        }
    }

    /// Returns the Wald test statistic compared to a null model with only an
    /// intercept (if one is used). This statistic is asymptotically chi-squared
    /// distributed with `test_ndf()` degrees of freedom.
    pub fn wald_test(&self) -> F {
        // The null parameters are all zero except for a possible intercept term
        // which optimizes the null model.
        let (_, null_params) = self.null_model_fit();
        self.wald_test_against(&null_params)
    }

    /// Returns the Wald test statistic compared to another specified model fit
    /// instead of the null model. The degrees of freedom cannot be generally
    /// inferred.
    pub fn wald_test_against(&self, alternative: &Array1<F>) -> F {
        let d_params: Array1<F> = &self.result - alternative;
        let fisher_alt: Array2<F> = self.fisher(&alternative);
        d_params.t().dot(&fisher_alt.dot(&d_params))
    }

    /// Returns the signed square root of the Wald test statistic for each
    /// parameter. Since it does not account for covariance between the
    /// parameters it may not be accurate.
    pub fn wald_z(&self) -> RegressionResult<Array1<F>> {
        let par_cov = self.covariance()?;
        let par_variances: ArrayView1<F> = par_cov.diag();
        Ok(&self.result / &par_variances.mapv(num_traits::Float::sqrt))
    }

    /// Returns the Akaike information criterion for the model fit.
    // TODO: Should an effective number of parameters that takes regularization
    // into acount be considered?
    pub fn aic(&self) -> F {
        F::from(2 * self.n_par).unwrap() - F::from(2.).unwrap() * self.model_like
    }

    /// Returns the Bayesian information criterion for the model fit.
    // TODO: Also consider the effect of regularization on this statistic.
    // TODO: Wikipedia suggests that the variance should included in the number
    // of parameters for multiple linear regression. Should an additional
    // parameter be included for the dispersion parameter? This question does
    // not affect the difference between two models fit with the methodology in
    // this package.
    pub fn bic(&self) -> F {
        let logn = num_traits::Float::ln(F::from(self.data.y.len()).unwrap());
        logn * F::from(self.n_par).unwrap() - F::from(2.).unwrap() * self.model_like
    }
}

/// Specialized functions for OLS.
impl<'a, F> Fit<'a, Linear, F>
where
    F: 'static + Float,
{
    /// Returns the coefficient of multiple correlation, R^2.
    pub fn r_sq(&self) -> F {
        let y_avg: F = self.data.y.mean().expect("Data should be non-empty");
        let total_sum_sq: F = self.data.y.mapv(|y| y - y_avg).mapv(|dy| dy * dy).sum();
        (total_sum_sq - self.residual_sum_sq()) / total_sum_sq
    }

    /// Returns the residual sum of squares, i.e. the sum of the squared residuals.
    pub fn residual_sum_sq(&self) -> F {
        self.residuals_response().mapv_into(|r| r * r).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        model::ModelBuilder,
        utility::{one_pad, standardize},
        Linear, Logistic,
    };
    use anyhow::Result;
    use approx::assert_abs_diff_eq;
    use ndarray::Axis;

    /// Checks if the test statistics are invariant based upon whether the data is standardized.
    #[test]
    fn standardization_invariance() -> Result<()> {
        let data_y = array![true, false, false, true, true, true, true, false, true];
        let data_x = array![-0.5, 0.3, -0.6, 0.2, 0.3, 1.2, 0.8, 0.6, -0.2].insert_axis(Axis(1));
        let lin_off = array![0.1, 0.0, -0.1, 0.2, 0.1, 0.3, 0.4, -0.1, 0.1];
        let data_x_std = standardize(data_x.clone());
        let model = ModelBuilder::<Logistic>::data(&data_y, &data_x)
            .linear_offset(lin_off.clone())
            .build()?;
        let fit = model.fit()?;
        let model_std = ModelBuilder::<Logistic>::data(&data_y, &data_x_std)
            .linear_offset(lin_off)
            .build()?;
        let fit_std = model_std.fit()?;
        let lr = fit.lr_test();
        let lr_std = fit_std.lr_test();
        assert_abs_diff_eq!(lr, lr_std);
        eprintln!("about to try score test");
        assert_abs_diff_eq!(
            fit.score_test()?,
            fit_std.score_test()?,
            epsilon = f32::EPSILON as f64
        );
        eprintln!("about to try wald test");
        assert_abs_diff_eq!(
            fit.wald_test(),
            fit_std.wald_test(),
            epsilon = 4.0 * f64::EPSILON
        );
        assert_abs_diff_eq!(fit.aic(), fit_std.aic());
        assert_abs_diff_eq!(fit.bic(), fit_std.bic());
        eprintln!("about to try deviance");
        assert_abs_diff_eq!(fit.deviance(), fit_std.deviance());
        // The Wald Z-score of the intercept term is not invariant under a
        // linear transformation of the data, but the parameter part seems to
        // be, at least for single-component data.
        assert_abs_diff_eq!(
            fit.wald_z()?[1],
            fit_std.wald_z()?[1],
            epsilon = 0.01 * f32::EPSILON as f64
        );

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
        // with no offsets, the result should be the link function of the mean.
        assert_abs_diff_eq!(
            fit.result[0],
            <Logistic as Glm>::Link::func(0.6),
            epsilon = 4.0 * f64::EPSILON
        );
        let empty_null_like = fit.null_like();
        assert_eq!(fit.test_ndf(), 0);
        dbg!(&fit.model_like);
        let lr = fit.lr_test();
        // Since there is no data, the null likelihood should be identical to
        // the fit likelihood, so the likelihood ratio test should yield zero.
        assert_abs_diff_eq!(lr, 0.);

        // Check that the assertions still hold if linear offsets are included.
        let lin_off: Array1<f64> = array![0.2, -0.1, 0.1, 0.0, 0.1];
        let model = ModelBuilder::<Logistic>::data(&data_y, &data_x)
            .linear_offset(lin_off)
            .build()?;
        let fit_off = model.fit()?;
        let empty_model_like_off = fit_off.model_like;
        let empty_null_like_off = fit_off.null_like();
        // these two assertions should be equivalent
        assert_abs_diff_eq!(fit_off.lr_test(), 0.);
        assert_abs_diff_eq!(empty_model_like_off, empty_null_like_off);

        // check consistency with data provided
        let data_x_with = array![[0.5], [-0.2], [0.3], [0.4], [-0.1]];
        let model = ModelBuilder::<Logistic>::data(&data_y, &data_x_with).build()?;
        let fit_with = model.fit()?;
        dbg!(&fit_with.result);
        // The null likelihood of the model with parameters should be the same
        // as the likelihood of the model with only the intercept.
        assert_abs_diff_eq!(empty_null_like, fit_with.null_like());

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
        assert_abs_diff_eq!(fit.null_like(), target_null_like);
        Ok(())
    }

    // Check that the deviance is equal to the sum of square deviations for a linear model
    #[test]
    fn deviance_linear() -> Result<()> {
        let data_y = array![0.3, -0.2, 0.5, 0.7, 0.2, 1.4, 1.1, 0.2];
        let data_x = array![0.6, 2.1, 0.4, -3.2, 0.7, 0.1, -0.3, 0.5].insert_axis(Axis(1));
        let model = ModelBuilder::<Linear>::data(&data_y, &data_x).build()?;
        let fit = model.fit()?;
        // The predicted values of Y given the model.
        let pred_y = fit.expectation(&one_pad(data_x.view()), None);
        let target_dev = (data_y - pred_y).mapv(|dy| dy * dy).sum();
        assert_abs_diff_eq!(fit.deviance(), target_dev,);
        Ok(())
    }

    // Check that the residuals for a linear model are all consistent.
    #[test]
    fn residuals_linear() -> Result<()> {
        let data_y = array![0.1, -0.3, 0.7, 0.2, 1.2, -0.4];
        let data_x = array![0.4, 0.1, 0.3, -0.1, 0.5, 0.6].insert_axis(Axis(1));
        let model = ModelBuilder::<Linear>::data(&data_y, &data_x).build()?;
        let fit = model.fit()?;
        let response = fit.residuals_response();
        let pearson = fit.residuals_pearson();
        let deviance = fit.residuals_deviance();
        assert_abs_diff_eq!(response, pearson);
        assert_abs_diff_eq!(response, deviance);
        Ok(())
    }

    // check the null likelihood for the case where it can be counted exactly.
    #[test]
    fn null_like_linear() -> Result<()> {
        let data_y = array![0.3, -0.2, 0.5, 0.7, 0.2, 1.4, 1.1, 0.2];
        let data_x = array![0.6, 2.1, 0.4, -3.2, 0.7, 0.1, -0.3, 0.5].insert_axis(Axis(1));
        let ybar: f64 = data_y.mean().unwrap();
        let model = ModelBuilder::<Linear>::data(&data_y, &data_x).build()?;
        let fit = model.fit()?;
        // let target_null_like = data_y.mapv(|y| -0.5 * (y - ybar) * (y - ybar)).sum();
        let target_null_like = data_y.mapv(|y| y * ybar - 0.5 * ybar * ybar).sum();
        // With the saturated likelihood subtracted the null likelihood should
        // just be the sum of squared differences from the mean.
        // let target_null_like = 0.;
        // dbg!(target_null_like);
        let fit_null_like = fit.null_like();
        assert_abs_diff_eq!(2. * (fit.model_like - fit_null_like), fit.lr_test());
        assert_eq!(fit.test_ndf(), 1);
        assert_abs_diff_eq!(
            fit_null_like,
            target_null_like,
            epsilon = 4.0 * f64::EPSILON
        );
        Ok(())
    }

    // check the null likelihood where there is no dependence on the X data.
    #[test]
    fn null_like_logistic_nodep() -> Result<()> {
        let data_y = array![true, true, false, false, true, false, false, true];
        let data_x = array![0.4, 0.2, 0.4, 0.2, 0.7, 0.7, -0.1, -0.1].insert_axis(Axis(1));
        let model = ModelBuilder::<Logistic>::data(&data_y, &data_x).build()?;
        let fit = model.fit()?;
        let lr = fit.lr_test();
        assert_abs_diff_eq!(lr, 0.);
        Ok(())
    }
    // TODO: Test that the statistics behave sensibly under regularization. The
    // likelihood ratio test should yield a smaller value.

    // Test the basic caching funcions.
    #[test]
    fn cached_computations() -> Result<()> {
        let data_y = array![true, true, false, true, true, false, false, false, true];
        let data_x = array![0.4, 0.1, -0.3, 0.7, -0.5, -0.1, 0.8, 1.0, 0.4].insert_axis(Axis(1));
        let model = ModelBuilder::<Logistic>::data(&data_y, &data_x).build()?;
        let fit = model.fit()?;
        let _null_like = fit.null_like();
        let _null_like = fit.null_like();
        let _cov = fit.covariance()?;
        let _wald = fit.wald_z();
        Ok(())
    }

    // Check the consistency of the various statistical tests for linear
    // regression, where they should be the most comparable.
    #[test]
    fn linear_stat_tests() -> Result<()> {
        let data_y = array![-0.3, -0.1, 0.0, 0.2, 0.4, 0.5, 0.8, 0.8, 1.1];
        let data_x = array![-0.5, -0.2, 0.1, 0.2, 0.5, 0.6, 0.7, 0.9, 1.3].insert_axis(Axis(1));
        let model = ModelBuilder::<Linear>::data(&data_y, &data_x).build()?;
        let fit = model.fit()?;
        let lr = fit.lr_test();
        let wald = fit.wald_test();
        let score = fit.score_test()?;
        assert_abs_diff_eq!(lr, wald, epsilon = 32.0 * f64::EPSILON);
        assert_abs_diff_eq!(lr, score, epsilon = 32.0 * f64::EPSILON);
        Ok(())
    }
}
