//! Stores the fit results of the IRLS regression and provides functions that
//! depend on the MLE estimate. These include statistical tests for goodness-of-fit.

pub mod options;
use crate::{
    error::RegressionResult,
    glm::{DispersionType, Glm},
    irls::Irls,
    link::{Link, Transform},
    model::{Dataset, Model},
    num::Float,
    regularization::IrlsReg,
    Linear,
};
use ndarray::{array, Array1, Array2, ArrayBase, ArrayView1, Axis, Data, Ix2};
use ndarray_linalg::InverseInto;
use options::FitOptions;
use std::{
    cell::{Ref, RefCell},
    marker::PhantomData,
};

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
    /// The regularizer of the fit
    reg: Box<dyn IrlsReg<F>>,
    /// The number of overall iterations taken in the IRLS.
    pub n_iter: usize,
    /// The number of parameters
    n_par: usize,
    /// The unscaled covariance matrix of the parameters, otherwise known as the Fisher
    /// information. Since the calculation requires a matrix inversion, it is computed only when
    /// needed and the value is cached.
    cov_unscaled: RefCell<Option<Array2<F>>>,
    /// The hat matrix of the data and fit. Since the calculation requires a matrix inversion of
    /// the fisher information, it is computed only when needed and the value is cached. Access
    /// through the `hat()` function.
    hat: RefCell<Option<Array2<F>>>,
    /// The likelihood and parameters for the null model.
    null_model: RefCell<Option<(F, Array1<F>)>>,
}

impl<'a, M, F> Fit<'a, M, F>
where
    M: Glm,
    F: 'static + Float,
{
    /// Returns the Akaike information criterion for the model fit. It is unique only to an
    /// additive constant, so only differences in AIC are meaningful.
    // TODO: Should an effective number of parameters that takes regularization
    // into acount be considered?
    pub fn aic(&self) -> F {
        F::from(2 * self.n_par).unwrap() - F::two() * self.model_like
    }

    /// Returns the Bayesian information criterion for the model fit.
    // TODO: Also consider the effect of regularization on this statistic.
    // TODO: Wikipedia suggests that the variance should included in the number
    // of parameters for multiple linear regression. Should an additional
    // parameter be included for the dispersion parameter? This question does
    // not affect the difference between two models fit with the methodology in
    // this package.
    pub fn bic(&self) -> F {
        let logn = num_traits::Float::ln(self.data.n_obs());
        logn * F::from(self.n_par).unwrap() - F::two() * self.model_like
    }

    /// The covariance matrix estimated by the Fisher information and the dispersion parameter (for
    /// families with a free scale). The Fisher matrix is cached to avoid repeating the potentially
    /// expensive matrix inversion.
    pub fn covariance(&self) -> RegressionResult<Array2<F>> {
        // The covariance must be multiplied by the dispersion parameter.
        // For logistic/poisson regression, this is identically 1.
        // For linear/gamma regression it is estimated from the data.
        let phi: F = self.dispersion();
        // NOTE: invh/invh_into() are bugged and incorrect!
        let unscaled_cov: Array2<F> = self.fisher_inv()?.to_owned();
        let cov = unscaled_cov * phi;
        Ok(cov)
    }

    /// Returns the deviance of the fit: twice the difference between the
    /// saturated likelihood and the model likelihood. Asymptotically this fits
    /// a chi-squared distribution with `self.ndf()` degrees of freedom.
    /// Note that the regularized likelihood is used here.
    // TODO: This is likely sensitive to regularization because the saturated
    // model is not regularized but the model likelihood is. Perhaps this can be
    // accounted for with an effective number of degrees of freedom.
    pub fn deviance(&self) -> F {
        // Note that this must change if the GLM likelihood subtracts the
        // saturated one already.
        let sat_like = self
            .data
            .apply_total_weights(self.data.y.mapv(M::log_like_sat))
            .sum();
        F::two() * (sat_like - self.model_like)
    }

    /// The dispersion parameter(typically denoted `phi`)  which relates the variance of the `y`
    /// values with the variance of the response distribution: `Var[y] = phi * Var[mu]`.
    /// Identically one for logistic, binomial, and Poisson regression.
    /// For others (linear, gamma) the dispersion parameter is estimated from the data.
    /// This is equal to the total deviance divided by the degrees of freedom.  For OLS linear
    /// regression this is equal to the sum of `(y_i - mu_i)^2 / (n-p)`, an estimate of `sigma^2`;
    /// with no covariates it is equal to the sample variance.
    pub fn dispersion(&self) -> F {
        use DispersionType::*;
        match M::DISPERSED {
            FreeDispersion => {
                let dev = self.deviance();
                dev / self.ndf()
            }
            NoDispersion => F::one(),
        }
    }

    /// Returns the errors in the response variables for the data passed as an
    /// argument given the current model fit.
    fn errors(&self, data: &Dataset<F>) -> Array1<F> {
        &data.y - &self.predict(&data.x, data.linear_offset.as_ref())
    }

    #[deprecated(since = "0.0.10", note = "use predict() instead")]
    pub fn expectation<S>(
        &self,
        data_x: &ArrayBase<S, Ix2>,
        lin_off: Option<&Array1<F>>,
    ) -> Array1<F>
    where
        S: Data<Elem = F>,
    {
        self.predict(data_x, lin_off)
    }

    /// Returns the fisher information (the negative hessian of the likelihood)
    /// at the parameter values given. The regularization is included.
    pub fn fisher(&self, params: &Array1<F>) -> Array2<F> {
        let lin_pred: Array1<F> = self.data.linear_predictor(params);
        let adj_var: Array1<F> = M::adjusted_variance_diag(&lin_pred);
        // calculate the fisher matrix
        let fisher: Array2<F> = (self.data.x_conj() * &adj_var).dot(&self.data.x);
        // Regularize the fisher matrix
        self.reg.as_ref().irls_mat(fisher, params)
    }

    /// The inverse of the (regularized) fisher information matrix. This is used in some other
    /// calculations (like the covariance and hat matrices) so it is cached.
    fn fisher_inv(&self) -> RegressionResult<Ref<Array2<F>>> {
        if self.cov_unscaled.borrow().is_none() {
            let fisher_reg = self.fisher(&self.result);
            // NOTE: invh/invh_into() are bugged and incorrect!
            let unscaled_cov: Array2<F> = fisher_reg.inv_into()?;
            *self.cov_unscaled.borrow_mut() = Some(unscaled_cov);
        }
        Ok(Ref::map(self.cov_unscaled.borrow(), |x| x.as_ref().unwrap()))
    }


    /// Returns the hat matrix of fit, also known as the "projection" or "influence" matrix.
    /// The convention used corresponds to H = dE[y]/dy and is orthogonal to the response
    /// residuals. This version is not symmetric.
    pub fn hat(&self) -> RegressionResult<Ref<Array2<F>>> {
        if self.hat.borrow().is_none() {
            let lin_pred = self.data.linear_predictor(&self.result);
            // Apply the eta' terms manually instead of calling adjusted_variance_diag, because the
            // adjusted variance method applies 2 powers to the variance, while we want one power
            // to the variance and one to the weights.
            // let adj_var = M::adjusted_variance_diag(&lin_pred);

            let mu = M::mean(&lin_pred);
            let var = mu.mapv_into(M::variance);
            let eta_d = M::Link::d_nat_param(&lin_pred);

            let fisher_inv = self.fisher_inv()?;

            // the GLM variance and the data weights are put on different sides in this convention
            let left = (var * &eta_d).insert_axis(Axis(1)) * &self.data.x;
            let right = self.data.x_conj() * &eta_d;
            let result = left.dot(&fisher_inv.dot(&right));

            *self.hat.borrow_mut() = Some(result);
        }
        let borrowed: Ref<Option<Array2<F>>> = self.hat.borrow();
        Ok(Ref::map(borrowed, |x| x.as_ref().unwrap()))
    }

    /// A matrix where each row corresponds to the contribution to the coefficients incurred by
    /// including the observation in that row. This is inexact for nonlinear models, as a one-step
    /// approximation is used.
    /// To approximate the coeficients that would result from excluding the ith observation, the
    /// ith row of this matrix should be subtracted from the fit result.
    pub fn infl_coef(&self) -> RegressionResult<Array2<F>> {
        let lin_pred = self.data.linear_predictor(&self.result);
        let resid_resp = self.resid_resp();
        let omh = - self.leverage()? + F::one();
        let resid_adj = M::Link::adjust_errors(resid_resp, &lin_pred) / omh;
        let xte = self.data.x_conj() * resid_adj;
        let fisher_inv = self.fisher_inv()?;
        let delta_b = xte.t().dot(&*fisher_inv);
        Ok(delta_b)
    }

    /// Returns the leverage for each observation. This is given by the diagonal of the projection
    /// matrix and indicates the sensitivity of each prediction to its corresponding observation.
    pub fn leverage(&self) -> RegressionResult<Array1<F>> {
        let hat = self.hat()?;
        Ok(hat.diag().to_owned())
    }

    /// Perform a likelihood-ratio test, returning the statistic -2*ln(L_0/L)
    /// where L_0 is the likelihood of the best-fit null model (with no
    /// parameters but the intercept) and L is the likelihood of the fit result.
    /// The number of degrees of freedom of this statistic, equal to the number
    /// of parameters fixed to zero to form the null model, is `test_ndf()`. By
    /// Wilks' theorem this statistic is asymptotically chi-squared distributed
    /// with this number of degrees of freedom.
    // TODO: Should the effective number of degrees of freedom due to
    // regularization be taken into account?
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
        let alt_like = M::log_like(self.data, alternative);
        let alt_like_reg = alt_like + self.reg.likelihood(alternative);
        F::two() * (self.model_like - alt_like_reg)
    }

    /// Returns the residual degrees of freedom in the model, i.e. the number
    /// of data points minus the number of parameters. Not to be confused with
    /// `test_ndf()`, the degrees of freedom in the statistical tests of the
    /// fit.
    pub fn ndf(&self) -> F {
        self.data.n_obs() - F::from(self.n_par).unwrap()
    }

    pub(crate) fn new(data: &'a Dataset<F>, use_intercept: bool, irls: Irls<M, F>) -> Self {
        let Irls {
            guess: result,
            options,
            reg,
            n_iter,
            last_like_data: data_like,
            ..
        } = irls;
        assert_eq!(
            data_like,
            M::log_like(data, &result),
            "Unregularized likelihoods should match exactly."
        );
        // Cache some of these variables that will be used often.
        let n_par = result.len();
        let model_like = data_like + reg.likelihood(&result);
        Self {
            model: PhantomData,
            data,
            use_intercept,
            result,
            options,
            model_like,
            reg,
            n_iter,
            n_par,
            cov_unscaled: RefCell::new(None),
            hat: RefCell::new(None),
            null_model: RefCell::new(None),
        }
    }

    /// Returns the likelihood given the null model, which fixes all parameters
    /// to zero except the intercept (if it is used). A total of `test_ndf()`
    /// parameters are constrained.
    pub fn null_like(&self) -> F {
        let (null_like, _) = self.null_model_fit();
        null_like
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
                    let y_bar: F = self.data.apply_total_weights(self.data.y.clone()).sum()
                        / self.data.sum_weights();
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
                    let null_like_total = self.data.sum_weights() * null_like_one;
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
                        let data_x_null = Array2::<F>::ones((self.data.y.len(), 1));
                        let null_model = Model {
                            model: std::marker::PhantomData::<M>,
                            data: Dataset::<F> {
                                y: self.data.y.clone(),
                                x: data_x_null,
                                linear_offset: Some(off.clone()),
                                weights: self.data.weights.clone(),
                                freqs: self.data.freqs.clone(),
                            },
                            // If we are in this branch it is because an intercept is needed.
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
                        let null_like_terms = ndarray::Zip::from(&self.data.y)
                            .and(&nat_par)
                            .map_collect(|&y, &eta| M::log_like_natural(y, eta));
                        let null_like = self.data.apply_total_weights(null_like_terms).sum()
                            / self.data.sum_weights();
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

    /// Returns the expected value of Y given the input data X. This data need
    /// not be the training data, so an option for linear offsets is provided.
    /// Panics if the number of covariates in the data matrix is not consistent
    /// with the training set. The data matrix may need to be padded by ones if
    /// it is not part of a Model. The `utility::one_pad()` function facilitates
    /// this.
    pub fn predict<S>(&self, data_x: &ArrayBase<S, Ix2>, lin_off: Option<&Array1<F>>) -> Array1<F>
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

    /// Return the deviance residuals for each point in the training data.
    /// Equal to `sign(y-E[y|x])*sqrt(-2*(L[y|x] - L_sat[y]))`.
    /// This is usually a better choice for non-linear models.
    /// NaNs might be possible if L[y|x] > L_sat[y] due to floating-point operations. These are
    /// not checked or clipped right now.
    pub fn resid_dev(&self) -> Array1<F> {
        let signs = self.resid_resp().mapv_into(F::signum);
        let ll_terms: Array1<F> = M::log_like_terms(self.data, &self.result);
        let ll_sat: Array1<F> = self.data.y.mapv(M::log_like_sat);
        let neg_two = F::from(-2.).unwrap();
        let ll_diff = (ll_terms - ll_sat) * neg_two;

        let ll_diff = match &self.data.weights {
            None => ll_diff,
            Some(w) => ll_diff * w,
        };

        let dev: Array1<F> = ll_diff.mapv_into(num_traits::Float::sqrt);
        signs * dev
    }

    /// Return the standardized deviance residuals, also known as the "internally studentized
    /// deviance residuals". This is generally applicable for outlier detection, although the
    /// influence of each point on the fit is only approximately accounted for.
    /// `d / sqrt(phi * (1 - h))` where `d` is the deviance residual, phi is the dispersion (e.g.
    /// sigma^2 for linear regression, 1 for logistic regression), and h is the leverage.
    pub fn resid_dev_std(&self) -> RegressionResult<Array1<F>> {
        let dev = self.resid_dev();
        let phi = self.dispersion();
        let hat: Array1<F> = self.leverage()?;
        let omh: Array1<F> = -hat + F::one();
        let denom: Array1<F> = (omh * phi).mapv_into(num_traits::Float::sqrt);
        Ok(dev / denom)
    }

    /// Return the partial residuals.
    pub fn resid_part(&self) -> Array1<F> {
        let x_mean = self.data.x.mean_axis(Axis(0)).expect("empty dataset");
        let x_centered = &self.data.x - x_mean.insert_axis(Axis(0));
        self.resid_work() + x_centered.dot(&self.result)
    }

    /// Return the Pearson residuals for each point in the training data.
    /// This is equal to `(y - E[y])/sqrt(V(E[y]))`, where V is the variance function.
    /// These are not scaled by the sample standard deviation for families with a free dispersion
    /// parameter like linear regression.
    pub fn resid_pear(&self) -> Array1<F> {
        let mu: Array1<F> = self.predict(&self.data.x, self.data.linear_offset.as_ref());
        let residuals = &self.data.y - &mu;
        let var_diag: Array1<F> = mu.mapv_into(M::variance);
        let var_diag = match &self.data.weights {
            None => var_diag,
            Some(w) => var_diag / w,
        };
        let std: Array1<F> = var_diag.mapv_into(num_traits::Float::sqrt);
        residuals / std
    }

    /// Return the standardized Pearson residuals for every observation.
    /// Also known as the "internally studentized Pearson residuals".
    /// (y - E[y]) / (sqrt(Var[y] * (1 - h))) where h is a vector representing the leverage for
    /// each observation.
    pub fn resid_pear_std(&self) -> RegressionResult<Array1<F>> {
        let pearson = self.resid_pear();
        let phi = self.dispersion();
        let hat = self.leverage()?;
        let omh = -hat + F::one();
        let denom: Array1<F> = (omh * phi).mapv_into(num_traits::Float::sqrt);
        Ok(pearson / denom)
    }

    /// Return the response residuals, or fitting deviation, for each data point in the fit; that
    /// is, the difference y - E[y|x] where the expectation value is the y value predicted by the
    /// model given x.
    pub fn resid_resp(&self) -> Array1<F> {
        self.errors(self.data)
    }

    /// Return the studentized residuals, which are the changes in the fit likelihood resulting
    /// from leaving each observation out. This is a robust and general method for outlier
    /// detection, although a one-step approximation is used to avoid re-fitting the model
    /// completely for each observation.
    /// If the linear errors are standard normally distributed then this statistic should follow a
    /// t-distribution with `self.ndf() - 1` degrees of freedom.
    pub fn resid_student(&self) -> RegressionResult<Array1<F>> {
        let r_dev = self.resid_dev();
        let r_pear = self.resid_pear();
        let signs = r_pear.mapv(F::signum);
        let r_dev_sq = r_dev.mapv_into(|x| x * x);
        let r_pear_sq = r_pear.mapv_into(|x| x * x);
        let hat = self.leverage()?;
        let omh = -hat.clone() + F::one();
        let sum_quad = &r_dev_sq + hat * r_pear_sq / &omh;
        let sum_quad_scaled = match M::DISPERSED {
            // The dispersion is corrected for the contribution from each current point.
            // This is an approximation; the exact solution would perform a fit at each point.
            DispersionType::FreeDispersion => {
                let dev = self.deviance();
                let dof = self.ndf() - F::one();
                let phi_i: Array1<F> = (-r_dev_sq / &omh + dev) / dof;
                sum_quad / phi_i
            }
            DispersionType::NoDispersion => sum_quad,
        };
        Ok(signs * sum_quad_scaled.mapv_into(num_traits::Float::sqrt))
    }

    /// Returns the working residuals `dg(\mu)/d\mu * (y - E{y|x})`.
    /// This should be equal to the response residuals divided by the variance function (as
    /// opposed to the square root of the variance as in the Pearson residuals).
    pub fn resid_work(&self) -> Array1<F> {
        let lin_pred: Array1<F> = self.data.linear_predictor(&self.result);
        let mu: Array1<F> = lin_pred.mapv(M::Link::func_inv);
        let resid_response: Array1<F> = &self.data.y - &mu;
        let var: Array1<F> = mu.mapv(M::variance);
        // adjust for non-canonical link functions; we want a total factor of 1/eta'
        let (adj_response, adj_var) = M::Link::adjust_errors_variance(resid_response, var, &lin_pred);
        adj_response / adj_var
    }

    /// Returns the score function (the gradient of the likelihood) at the
    /// parameter values given. It should be zero within FPE at the minimized
    /// result.
    pub fn score(&self, params: &Array1<F>) -> Array1<F> {
        // This represents the predictions given the input parameters, not the
        // fit parameters.
        let lin_pred: Array1<F> = self.data.linear_predictor(params);
        let mu: Array1<F> = M::mean(&lin_pred);
        let resid_response = &self.data.y - mu;
        let resid_working = M::Link::adjust_errors(resid_response, &lin_pred);
        let score_unreg = self.data.x_conj().dot(&resid_working);
        self.reg.as_ref().gradient(score_unreg, params)
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
        // NOTE: invh/invh_into() are bugged and incorrect!
        let inv_fisher_alt = fisher_alt.inv_into()?;
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
        let fisher_alt: Array2<F> = self.fisher(alternative);
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
        (total_sum_sq - self.resid_sum_sq()) / total_sum_sq
    }

    /// Returns the residual sum of squares, i.e. the sum of the squared residuals.
    pub fn resid_sum_sq(&self) -> F {
        self.resid_resp().mapv_into(|r| r * r).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{s, concatenate};
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
        assert_abs_diff_eq!(lr, 0., epsilon = 4. * f64::EPSILON);

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
        let pred_y = fit.predict(&one_pad(data_x.view()), None);
        let target_dev = (data_y - pred_y).mapv(|dy| dy * dy).sum();
        assert_abs_diff_eq!(fit.deviance(), target_dev,);
        Ok(())
    }

    // Check that the deviance and dispersion parameter are equal up to the number of degrees of
    // freedom for a linea model.
    #[test]
    fn deviance_dispersion_eq_linear() -> Result<()> {
        let data_y = array![0.2, -0.1, 0.4, 1.3, 0.2, -0.6, 0.9];
        let data_x = array![
            [0.4, 0.2],
            [0.1, 0.4],
            [-0.1, 0.3],
            [0.5, 0.7],
            [0.4, 0.1],
            [-0.2, -0.3],
            [0.4, -0.1]
        ];
        let model = ModelBuilder::<Linear>::data(&data_y, &data_x).build()?;
        let fit = model.fit()?;
        let dev = fit.deviance();
        let disp = fit.dispersion();
        let ndf = fit.ndf();
        assert_abs_diff_eq!(dev, disp * ndf, epsilon = 4. * f64::EPSILON);
        Ok(())
    }

    // Check that the residuals for a linear model are all consistent.
    #[test]
    fn residuals_linear() -> Result<()> {
        let data_y = array![0.1, -0.3, 0.7, 0.2, 1.2, -0.4];
        let data_x = array![0.4, 0.1, 0.3, -0.1, 0.5, 0.6].insert_axis(Axis(1));
        let model = ModelBuilder::<Linear>::data(&data_y, &data_x).build()?;
        let fit = model.fit()?;
        let response = fit.resid_resp();
        let pearson = fit.resid_pear();
        let deviance = fit.resid_dev();
        assert_abs_diff_eq!(response, pearson);
        assert_abs_diff_eq!(response, deviance);
        let pearson_std = fit.resid_pear_std()?;
        let deviance_std = fit.resid_dev_std()?;
        let _student = fit.resid_student()?;
        assert_abs_diff_eq!(pearson_std, deviance_std, epsilon = 8. * f64::EPSILON);

        // // NOTE: Studentization can't be checked directly because the method used is an
        // approximation. Another approach will be needed to give exact values.
        // let orig_dev = fit.deviance();
        // let n_data = data_y.len();
        // // Check that the leave-one-out stats hold literally
        // let mut loo_dev: Vec<f64> = Vec::new();
        // for i in 0..n_data {
        //     let ya = data_y.slice(s![0..i]);
        //     let yb = data_y.slice(s![i + 1..]);
        //     let xa = data_x.slice(s![0..i, ..]);
        //     let xb = data_x.slice(s![i + 1.., ..]);
        //     let y_loo = concatenate![Axis(0), ya, yb];
        //     let x_loo = concatenate![Axis(0), xa, xb];
        //     let model_i = ModelBuilder::<Linear>::data(&y_loo, &x_loo).build()?;
        //     let fit_i = model_i.fit()?;
        //     let yi = data_y[i];
        //     let xi = data_x.slice(s![i..i + 1, ..]);
        //     let xi = crate::utility::one_pad(xi);
        //     let yi_pred: f64 = fit_i.predict(&xi, None)[0];
        //     let disp_i = fit_i.dispersion();
        //     let pear_loo = (yi - yi_pred) / disp_i.sqrt();
        //     let dev_i = fit_i.deviance();
        //     let d_dev = 2. * (orig_dev - dev_i);
        //     loo_dev.push(d_dev.sqrt() * (yi - yi_pred).signum());
        // }
        // let loo_dev: Array1<f64> = loo_dev.into();
        // This is off from 1 by a constant factor that depends on the data
        // This is only approximately true
        // assert_abs_diff_eq!(student, loo_dev);
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

    // check the leave-one-out one-step for the linear model
    #[test]
    fn loo_linear() -> Result<()> {
        let data_y = array![0.1, -0.3, 0.7, 0.2, 1.2, -0.4];
        let data_x = array![0.4, 0.1, 0.3, -0.1, 0.5, 0.6].insert_axis(Axis(1));
        let weights = array![1.0, 1.2, 0.8, 1.1, 1.0, 0.7];
        let model = ModelBuilder::<Linear>::data(&data_y, &data_x).var_weights(weights.clone()).build()?;
        let fit = model.fit()?;

        let loo_coef: Array2<f64> = fit.infl_coef()?;
        let loo_results = &fit.result - loo_coef;
        let n_data = data_y.len();
        for i in 0..n_data {
            let ya = data_y.slice(s![0..i]);
            let yb = data_y.slice(s![i + 1..]);
            let xa = data_x.slice(s![0..i, ..]);
            let xb = data_x.slice(s![i + 1.., ..]);
            let wa = weights.slice(s![0..i]);
            let wb = weights.slice(s![i+1..]);
            let y_loo = concatenate![Axis(0), ya, yb];
            let x_loo = concatenate![Axis(0), xa, xb];
            let w_loo = concatenate![Axis(0), wa, wb];
            let model_i = ModelBuilder::<Linear>::data(&y_loo, &x_loo).var_weights(w_loo).build()?;
            let fit_i = model_i.fit()?;
            assert_abs_diff_eq!(loo_results.row(i), &fit_i.result, epsilon = f32::EPSILON as f64);
        }
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
