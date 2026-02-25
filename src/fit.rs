//! Stores the fit results of the IRLS regression and provides functions that
//! depend on the MLE estimate. These include statistical tests for goodness-of-fit.

pub mod options;
use crate::{
    Linear,
    data::{Dataset, one_pad},
    error::RegressionResult,
    glm::{DispersionType, Glm},
    irls::{Irls, IrlsStep},
    link::{Link, Transform},
    model::Model,
    num::Float,
    regularization::IrlsReg,
};
use ndarray::{Array1, Array2, ArrayBase, ArrayView1, Axis, Data, Ix2, array, s};
use ndarray_linalg::InverseInto;
use once_cell::unsync::OnceCell; // can be replaced by std::cell::OnceCell upon stabilization
use options::FitOptions;
use std::marker::PhantomData;

/// the result of a successful GLM fit
pub struct Fit<'a, M, F>
where
    M: Glm,
    F: Float,
{
    model: PhantomData<M>,
    /// The data and model specification used in the fit.
    data: &'a Dataset<F>,
    /// The parameter values that maximize the likelihood as given by the IRLS regression. If the
    /// dataset was internally standardized, this is transformed back
    pub result: Array1<F>,
    /// The parameter values used internally for the standardized data.
    result_std: Array1<F>,
    /// The predicted y-values for the training data.
    y_hat: Array1<F>,
    /// The options used for this fit.
    pub options: FitOptions<F>,
    /// The value of the likelihood function for the fit result.
    pub model_like: F,
    /// The regularizer of the fit
    reg: Box<dyn IrlsReg<F>>,
    /// The number of overall iterations taken in the IRLS.
    pub n_iter: usize,
    /// The history of guesses and likelihoods over the IRLS iterations.
    pub history: Vec<IrlsStep<F>>,
    /// The number of parameters
    n_par: usize,
    // The remaining variables hold cached results and matrices
    /// The estimated dispersion parameter, which is called in many places. For some families this
    /// is just one.
    phi: OnceCell<F>,
    /// The Pearson residuals, a common statistic that is re-used in several other quantities.
    resid_pear: OnceCell<Array1<F>>,
    /// The covariance matrix, using the sandwich approach.
    covariance: OnceCell<Array2<F>>,
    /// The inverse of the regularized Fisher information matrix in the external non-standardized
    /// parameter basis. Used as a component of the sandwich covariance and for influence
    /// calculations. Cached on first access.
    fisher_inv: OnceCell<Array2<F>>,
    /// The inverse of the fisher matrix in standardized internal parameters.
    fisher_std_inv: OnceCell<Array2<F>>,
    /// The hat matrix of the data and fit. Since the calculation requires a matrix inversion of
    /// the fisher information, it is computed only when needed and the value is cached. Access
    /// through the `hat()` function.
    hat: OnceCell<Array2<F>>,
    /// The likelihood and parameters for the null model.
    null_model: OnceCell<(F, Array1<F>)>,
}

impl<'a, M, F> Fit<'a, M, F>
where
    M: Glm,
    F: 'static + Float,
{
    /// Returns the Akaike information criterion for the model fit.
    ///
    /// ```math
    /// \text{AIC} = D + 2K - 2\sum_{i} \ln w_{i}
    /// ```
    ///
    /// where $`D`$ is the deviance, $`K`$ is the number of parameters, and $`w_i`$ are the
    /// variance weights.
    /// This is unique only to an additive constant, so only differences in AIC are meaningful.
    pub fn aic(&self) -> F {
        let log_weights = self.data.get_variance_weights().mapv(num_traits::Float::ln);
        let sum_log_weights = self.data.freq_sum(&log_weights);
        // NOTE: This is now the unregularized deviance.
        self.deviance() + F::two() * self.rank() - F::two() * sum_log_weights
    }

    /// Returns the Bayesian information criterion for the model fit.
    ///
    /// ```math
    /// \text{BIC} = K \ln(n) - 2l - 2\sum_{i} \ln w_{i}
    /// ```
    ///
    /// where $`K`$ is the number of parameters, $`n`$ is the number of observations, and $`l`$ is
    /// the log-likelihood (including the variance weight normalization terms).
    // TODO: Also consider the effect of regularization on this statistic.
    // TODO: Wikipedia suggests that the variance should included in the number
    // of parameters for multiple linear regression. Should an additional
    // parameter be included for the dispersion parameter? This question does
    // not affect the difference between two models fit with the methodology in
    // this package.
    pub fn bic(&self) -> F {
        let log_weights = self.data.get_variance_weights().mapv(num_traits::Float::ln);
        let sum_log_weights = self.data.freq_sum(&log_weights);
        let logn = num_traits::Float::ln(self.data.n_obs());
        logn * self.rank() - F::two() * self.model_like - F::two() * sum_log_weights
    }

    /// The Cook's distance for each observation, which measures how much the predicted values
    /// change when leaving out each observation.
    ///
    /// ```math
    /// C_i = \frac{r_i^2 \, h_i}{K \, \hat\phi \, (1 - h_i)^2}
    /// ```
    ///
    /// where $`r_i`$ is the Pearson residual, $`h_i`$ is the leverage, $`K`$ is the rank
    /// (number of parameters), and $`\hat\phi`$ is the estimated dispersion.
    pub fn cooks(&self) -> RegressionResult<Array1<F>, F> {
        let hat = self.leverage()?;
        let pear_sq = self.resid_pear().mapv(|r| r * r);
        let h_terms: Array1<F> = hat.mapv_into(|h| {
            let omh = F::one() - h;
            h / (omh * omh)
        });
        let denom: F = self.rank() * self.dispersion();
        Ok(pear_sq * h_terms / denom)
    }

    /// The covariance matrix of the parameter estimates. When no regularization is used, this is:
    ///
    /// ```math
    /// \text{Cov}[\hat{\boldsymbol\beta}] = \hat\phi \, (\mathbf{X}^\mathsf{T}\mathbf{WSX})^{-1}
    /// ```
    ///
    /// When regularization is active, the sandwich form is used to correctly account for the bias
    /// introduced by the penalty:
    ///
    /// ```math
    /// \text{Cov}[\hat{\boldsymbol\beta}] = \hat\phi \, \mathcal{I}_\text{reg}^{-1} \, \mathcal{I}_\text{data} \, \mathcal{I}_\text{reg}^{-1}
    /// ```
    ///
    /// where $`\mathcal{I}_\text{reg}`$ is the regularized Fisher information and
    /// $`\mathcal{I}_\text{data}`$ is the unregularized (data-only) Fisher information.
    /// When unregularized, $`\mathcal{I}_\text{reg} = \mathcal{I}_\text{data}`$ and this reduces
    /// to the standard form. The result is cached on first access.
    pub fn covariance(&self) -> RegressionResult<&Array2<F>, F> {
        self.covariance.get_or_try_init(|| {
            // The covariance must be multiplied by the dispersion parameter.
            // For logistic/poisson regression, this is identically 1.
            // For linear/gamma regression it is estimated from the data.
            let phi: F = self.dispersion();
            // NOTE: invh()/invh_into() are bugged and give incorrect values!
            let f_reg_inv: Array2<F> = self.fisher_inv()?.to_owned();
            // Use the sandwich form so that regularization doesn't artificially deflate uncertainty.
            // When unregularized, F_reg = F_data and this reduces to F_data^{-1}.
            // The unregularized fisher matrix is most easily acquired in terms of the standardized
            // variables, so apply the external transformation to it.
            let f_data = self
                .data
                .inverse_transform_fisher(self.fisher_data_std(&self.result_std));
            Ok(f_reg_inv.dot(&f_data).dot(&f_reg_inv) * phi)
        })
    }

    /// Returns the deviance of the fit:
    ///
    /// ```math
    /// D = -2 \left[ l(\hat{\boldsymbol\beta}) - l_\text{sat} \right]
    /// ```
    ///
    /// Asymptotically $`\chi^2`$-distributed with [`ndf()`](Self::ndf) degrees of freedom.
    /// The unregularized likelihood is used.
    pub fn deviance(&self) -> F {
        let terms = self.deviance_terms();
        self.data.freq_sum(&terms)
    }

    /// Returns the contribution to the deviance from each observation. The total deviance should
    /// be the sum of all of these. Variance weights are already included, but not frequency
    /// weights.
    fn deviance_terms(&self) -> Array1<F> {
        let ll_terms: Array1<F> = M::log_like_terms(self.data, &self.result_std);
        let ll_sat: Array1<F> = self.data.y.mapv(M::log_like_sat);
        let terms = (ll_sat - ll_terms) * F::two();
        self.data.apply_var_weights(terms)
    }

    /// Returns the self-excluded deviance terms, i.e. the deviance of an observation as if the
    /// model was fit without it. This is a one-step approximation.
    fn deviance_terms_loo(&self) -> RegressionResult<Array1<F>, F> {
        let dev_terms = self.deviance_terms();
        let pear_sq = self.resid_pear().mapv(|r| r * r);
        let hat_rat = self.leverage()?.mapv(|h| h / (F::one() - h));
        let result = dev_terms + &hat_rat * (&hat_rat + F::two()) * pear_sq;
        Ok(result)
    }

    /// The dispersion parameter $`\hat\phi`$ relating the variance to the variance function:
    /// $`\text{Var}[y] = \phi \, V(\mu)`$.
    ///
    /// Identically one for logistic, binomial, and Poisson regression.
    /// For families with a free dispersion (linear, gamma), estimated as:
    ///
    /// ```math
    /// \hat\phi = \frac{D}{\left(1 - \frac{K}{n_\text{eff}}\right) \sum_i w_i}
    /// ```
    ///
    /// which reduces to $`D / (N - K)`$ without variance weights.
    pub fn dispersion(&self) -> F {
        *self.phi.get_or_init(|| {
            use DispersionType::*;
            match M::DISPERSED {
                FreeDispersion => {
                    let dev = self.deviance();
                    let p = self.rank();
                    let n_eff = self.data.n_eff();
                    let scaling = if p >= n_eff {
                        // This is the overparameterized regime, which is checked directly instead of
                        // allowing negative values. It's not clear what conditions result in this when
                        // p < N.
                        F::zero()
                    } else {
                        (F::one() - p / n_eff) * self.data.sum_weights()
                    };
                    dev / scaling
                }
                NoDispersion => F::one(),
            }
        })
    }

    /// Return the dispersion terms with the observation(s) at each point excluded from the fit.
    fn dispersion_loo(&self) -> RegressionResult<Array1<F>, F> {
        use DispersionType::*;
        match M::DISPERSED {
            FreeDispersion => {
                let pear_sq = self.resid_pear().mapv(|r| r * r);
                let hat_rat = self.leverage()?.mapv(|h| h / (F::one() - h));
                let terms = self.deviance_terms() + hat_rat * pear_sq;
                // Don't apply total weights since the variance weights are already
                // included in the residual terms. However, we do need the frequency weights.
                let terms = self.data.apply_freq_weights(terms);
                let total: Array1<F> = -terms + self.deviance();
                let scaled_total: Array1<F> = match &self.data.weights {
                    None => match &self.data.freqs {
                        Some(f) => total / -(f - self.ndf()),
                        None => total / (self.ndf() - F::one()),
                    },
                    Some(w) => {
                        let v1 = self.data.freq_sum(w);
                        let w2 = w * w;
                        let v2 = self.data.freq_sum(&w2);
                        // The subtracted out terms need the frequency terms as well
                        let f_w = self.data.apply_freq_weights(w.clone());
                        let f_w2 = self.data.apply_freq_weights(w2);
                        // the modifed sums from leaving out the ith observation
                        let v1p = -f_w + v1;
                        let v2p = -f_w2 + v2;
                        let p = self.rank();
                        let scale = &v1p - v2p / &v1p * p;
                        total / scale
                    }
                };
                Ok(scaled_total)
            }
            NoDispersion => Ok(Array1::<F>::ones(self.data.y.len())),
        }
    }

    /// Returns the Fisher information (the negative Hessian of the log-likelihood) at the
    /// parameter values given:
    ///
    /// ```math
    /// \mathcal{I}(\boldsymbol\beta) = \mathbf{X}^\mathsf{T}\mathbf{W}\eta'^2\mathbf{S}\mathbf{X}
    /// ```
    ///
    /// where $`\mathbf{S} = \text{diag}(V(\mu_i))`$ and $`\eta'`$ is the derivative of the natural
    /// parameter in terms of the linear predictor ($`\eta(\omega) = g_0(g^{-1}(\omega))`$ where
    /// $`g_0`$ is the canonical link function).
    /// The regularization is included.
    pub fn fisher(&self, params: &Array1<F>) -> Array2<F> {
        // Note that fisher() is a public function so it should take the external parameters.
        let params = self.data.transform_beta(params.clone());
        // We actually need to futher apply the transformation beyond this, so that *arrays
        // multiplying this resulting matrix* are hit the right way.
        let fish_std = self.fisher_std(&params);
        self.data.inverse_transform_fisher(fish_std)
    }

    /// The inverse of the (regularized) fisher information matrix, in the external parameter
    /// basis. Used for the influence calculations, and as a component of the sandwich covariance.
    /// Cached on first access.
    fn fisher_inv(&self) -> RegressionResult<&Array2<F>, F> {
        self.fisher_inv.get_or_try_init(|| {
            let fisher_reg = self.fisher(&self.result);
            // NOTE: invh/invh_into() are bugged and incorrect!
            let fish_inv = fisher_reg.inv_into()?;
            Ok(fish_inv)
        })
    }

    /// Compute the data-only (unregularized) Fisher information in the internal standardized basis.
    fn fisher_data_std(&self, params: &Array1<F>) -> Array2<F> {
        let lin_pred: Array1<F> = self.data.linear_predictor_std(params);
        let adj_var: Array1<F> = M::adjusted_variance_diag(&lin_pred);
        (self.data.x_conj() * &adj_var).dot(&self.data.x)
    }

    /// Compute the fisher information in terms of the internal (likely standardized)
    /// coefficients. The regularization is included here.
    fn fisher_std(&self, params: &Array1<F>) -> Array2<F> {
        let fisher = self.fisher_data_std(params);
        self.reg.as_ref().irls_mat(fisher, params)
    }

    /// The inverse of the (regularized) fisher information matrix, in the internal standardized
    /// parameter basis.
    fn fisher_std_inv(&self) -> RegressionResult<&Array2<F>, F> {
        self.fisher_std_inv.get_or_try_init(|| {
            let fisher_reg = self.fisher_std(&self.result_std);
            // NOTE: invh/invh_into() are bugged and incorrect!
            let fish_inv = fisher_reg.inv_into()?;
            Ok(fish_inv)
        })
    }

    /// Returns the hat matrix, also known as the "projection" or "influence" matrix:
    ///
    /// ```math
    /// P_{ij} = \frac{\partial \hat{y}_i}{\partial y_j}
    /// ```
    ///
    /// Orthogonal to the response residuals at the fit result: $`\mathbf{P}(\mathbf{y} -
    /// \hat{\mathbf{y}}) = 0`$.
    /// This version is not symmetric, but the diagonal is invariant to this choice of convention.
    pub fn hat(&self) -> RegressionResult<&Array2<F>, F> {
        self.hat.get_or_try_init(|| {
            // Do the full computation in terms of the internal parameters, since this observable
            // is not sensitive to the choice of basis.
            let lin_pred = self.data.linear_predictor_std(&self.result_std);
            // Apply the eta' terms manually instead of calling adjusted_variance_diag, because the
            // adjusted variance method applies 2 powers to the variance, while we want one power
            // to the variance and one to the weights.
            // let adj_var = M::adjusted_variance_diag(&lin_pred);

            let mu = M::mean(&lin_pred);
            let var = mu.mapv_into(M::variance);
            let eta_d = M::Link::d_nat_param(&lin_pred);

            let fisher_inv = self.fisher_std_inv()?;

            // the GLM variance and the data weights are put on different sides in this convention
            let left = (var * &eta_d).insert_axis(Axis(1)) * &self.data.x;
            let right = self.data.x_conj() * &eta_d;
            Ok(left.dot(&fisher_inv.dot(&right)))
        })
    }

    /// The one-step approximation to the change in coefficients from excluding each observation:
    ///
    /// ```math
    /// \Delta\boldsymbol\beta^{(-i)} \approx \frac{1}{1-h_i}\,\mathcal{I}^{-1}\mathbf{x}^{(i)} w_i \eta'_i e_i
    /// ```
    ///
    /// Each row $`i`$ should be subtracted from $`\hat{\boldsymbol\beta}`$ to approximate
    /// the coefficients that would result from excluding observation $`i`$.
    /// Exact for linear models; a one-step approximation for nonlinear models.
    pub fn infl_coef(&self) -> RegressionResult<Array2<F>, F> {
        // The linear predictor can be acquired in terms of the standardized parameters, but the
        // rest of the computation should use external.
        let lin_pred = self.data.linear_predictor_std(&self.result_std);
        let resid_resp = self.resid_resp();
        let omh = -self.leverage()? + F::one();
        let resid_adj = M::Link::adjust_errors(resid_resp, &lin_pred) / omh;
        let xte = self.data.x_conj_ext() * resid_adj;
        let fisher_inv = self.fisher_inv()?;
        let delta_b = xte.t().dot(fisher_inv);
        Ok(delta_b)
    }

    /// Returns the leverage $`h_i = P_{ii}`$ for each observation: the diagonal of the hat matrix.
    /// Indicates the sensitivity of each prediction to its corresponding observation.
    pub fn leverage(&self) -> RegressionResult<Array1<F>, F> {
        let hat = self.hat()?;
        Ok(hat.diag().to_owned())
    }

    /// Returns exact coefficients from leaving each observation out, one-at-a-time.
    /// This is a much more expensive operation than the original regression because a new one is
    /// performed for each observation.
    pub fn loo_exact(&self) -> RegressionResult<Array2<F>, F> {
        // Use the one-step approximation to get good starting points for each exclusion.
        let loo_coef: Array2<F> = self.infl_coef()?;
        // NOTE: These coefficients need to be in terms of external parameters
        let loo_initial = &self.result - loo_coef;
        let mut loo_result = loo_initial.clone();
        let n_obs = self.data.y.len();
        for i in 0..n_obs {
            let data_i: Dataset<F> = {
                // Get the proper X data matrix as it existed before standardization and
                // intercept-padding.
                let x_i = {
                    let x_i = if self.data.has_intercept {
                        self.data.x.slice(s![.., 1..]).to_owned()
                    } else {
                        self.data.x.clone()
                    };
                    match &self.data.standardizer {
                        Some(std) => std.inverse_transform(x_i),
                        None => x_i,
                    }
                };
                // Leave the observation out by setting the frequency weight to zero.
                let mut freqs_i: Array1<F> =
                    self.data.freqs.clone().unwrap_or(Array1::<F>::ones(n_obs));
                freqs_i[i] = F::zero();
                let mut data_i = Dataset {
                    y: self.data.y.clone(),
                    x: x_i,
                    linear_offset: self.data.linear_offset.clone(),
                    weights: self.data.weights.clone(),
                    freqs: Some(freqs_i),
                    // These fields must be set this way, as they are in the ModelBuilder, before
                    // finalize_design_matrix() is called.
                    has_intercept: false,
                    standardizer: None,
                };
                data_i.finalize_design_matrix(
                    self.data.standardizer.is_some(),
                    self.data.has_intercept,
                );
                data_i
            };
            let model_i = Model {
                model: PhantomData::<M>,
                data: data_i,
            };
            let options = {
                let mut options = self.options.clone();
                // The one-step approximation should be a good starting point.
                // Use the external result as it should be re-standardized to the new dataset
                // internally before being passed to IRLS.
                options.init_guess = Some(loo_initial.row(i).to_owned());
                options
            };
            let fit_i = model_i.with_options(options).fit()?;
            // The internal re-fit transforms back to the external scale.
            loo_result.row_mut(i).assign(&fit_i.result);
        }
        Ok(loo_result)
    }

    /// Perform a likelihood-ratio test, returning the statistic:
    ///
    /// ```math
    /// \Lambda = -2 \ln \frac{L_0}{L} = -2(l_0 - l)
    /// ```
    ///
    /// where $`L_0`$ is the null model likelihood (intercept only) and $`L`$ is the fit likelihood.
    /// By Wilks' theorem, asymptotically $`\chi^2`$-distributed with
    /// [`test_ndf()`](Self::test_ndf) degrees of freedom.
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
        let alt_std = self.data.transform_beta(alternative.clone());
        let alt_like = M::log_like(self.data, &alt_std);
        let alt_like_reg = alt_like + self.reg.likelihood(&alt_std);
        F::two() * (self.model_like - alt_like_reg)
    }

    /// Returns the residual degrees of freedom in the model, i.e. the number
    /// of data points minus the number of parameters. Not to be confused with
    /// `test_ndf()`, the degrees of freedom in the statistical tests of the
    /// fit parameters.
    pub fn ndf(&self) -> F {
        self.data.n_obs() - self.rank()
    }

    pub(crate) fn new(data: &'a Dataset<F>, optimum: IrlsStep<F>, irls: Irls<M, F>) -> Self {
        let IrlsStep {
            guess: result_std,
            like: model_like,
        } = optimum;
        let Irls {
            options,
            reg,
            n_iter,
            history,
            ..
        } = irls;
        // Cache some of these variables that will be used often.
        let n_par = result_std.len();
        // NOTE: This necessarily uses the coefficients directly from the standardized data.
        // Store these predictions as they are commonly used.
        let y_hat = M::mean(&data.linear_predictor_std(&result_std));
        // The public result must be transformed back to the external scale for compatability with
        // the input data.
        let result_ext = data.inverse_transform_beta(result_std.clone());
        // The history also needs to be transformed back to the external scale, since it is public.
        // It shouldn't be used directly by the Fit's methods.
        let history = history
            .into_iter()
            .map(|IrlsStep { guess, like }| IrlsStep {
                guess: data.inverse_transform_beta(guess),
                like,
            })
            .collect();
        Self {
            model: PhantomData,
            data,
            result: result_ext,
            result_std,
            y_hat,
            options,
            model_like,
            reg,
            n_iter,
            history,
            n_par,
            phi: OnceCell::new(),
            resid_pear: OnceCell::new(),
            covariance: OnceCell::new(),
            fisher_inv: OnceCell::new(),
            fisher_std_inv: OnceCell::new(),
            hat: OnceCell::new(),
            null_model: OnceCell::new(),
        }
    }

    /// Returns the likelihood given the null model, which fixes all parameters
    /// to zero except the intercept (if it is used). A total of `test_ndf()`
    /// parameters are constrained.
    pub fn null_like(&self) -> F {
        let (null_like, _) = self.null_model_fit();
        *null_like
    }

    /// Return the likelihood and null model parameters, which will be zero with the possible
    /// exception of the intercept term. Since this can require an additional regression, the
    /// values are cached.
    fn null_model_fit(&self) -> &(F, Array1<F>) {
        self.null_model
            .get_or_init(|| match &self.data.linear_offset {
                None => {
                    // If there is no linear offset, the natural parameter is
                    // identical for all observations so it is sufficient to
                    // calculate the null likelihood for a single point with y equal
                    // to the average.
                    // The average y
                    let y_bar: F = self.data.weighted_sum(&self.data.y) / self.data.sum_weights();
                    // This approach assumes that the likelihood is in the natural
                    // exponential form as calculated by Glm::log_like_natural(). If that
                    // function is overridden and the values differ significantly, this
                    // approach will give incorrect results. If the likelihood has terms
                    // non-linear in y, then the likelihood must be calculated for every
                    // point rather than averaged.
                    // If the intercept is allowed to maximize the likelihood, the natural
                    // parameter is equal to the link of the expectation. Otherwise it is
                    // the transformation function of zero.
                    let intercept: F = if self.data.has_intercept {
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
                    if self.data.has_intercept {
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
                                // If we are in this branch it is because an intercept is needed.
                                has_intercept: true,
                                // We don't use standardization for the null model.
                                standardizer: None,
                            },
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
                            // there is only one parameter in this fit. It should be the same as
                            // the external result since this model doesn't have standardization.
                            par[0] = null_fit.result_std[0];
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
                        let null_like = self.data.weighted_sum(&null_like_terms);
                        let null_params = Array1::<F>::zeros(self.n_par);
                        (null_like, null_params)
                    }
                }
            })
    }

    /// Returns $`\hat{\mathbf{y}} = g^{-1}(\mathbf{X}\hat{\boldsymbol\beta} + \boldsymbol\omega_0)`$
    /// given input data $`\mathbf{X}`$ and an optional linear offset
    /// $`\boldsymbol\omega_0`$. The data need not be the training data.
    pub fn predict<S>(&self, data_x: &ArrayBase<S, Ix2>, lin_off: Option<&Array1<F>>) -> Array1<F>
    where
        S: Data<Elem = F>,
    {
        let lin_pred = if self.data.has_intercept {
            one_pad(data_x.view())
        } else {
            data_x.to_owned()
        }
        .dot(&self.result);
        let lin_pred: Array1<F> = if let Some(off) = &lin_off {
            lin_pred + *off
        } else {
            lin_pred
        };
        M::mean(&lin_pred)
    }

    /// Returns the rank $`K`$ of the model (i.e. the number of parameters)
    fn rank(&self) -> F {
        F::from(self.n_par).unwrap()
    }

    /// Return the deviance residuals for each point in the training data:
    ///
    /// ```math
    /// d_i = \text{sign}(y_i - \hat\mu_i)\sqrt{D_i}
    /// ```
    ///
    /// where $`D_i`$ is the per-observation deviance contribution.
    /// This is usually a better choice than Pearson residuals for non-linear models.
    pub fn resid_dev(&self) -> Array1<F> {
        let signs = self.resid_resp().mapv_into(F::signum);
        let ll_diff = self.deviance_terms();
        let dev: Array1<F> = ll_diff.mapv_into(num_traits::Float::sqrt);
        signs * dev
    }

    /// Return the standardized deviance residuals (internally studentized):
    ///
    /// ```math
    /// d_i^* = \frac{d_i}{\sqrt{\hat\phi(1 - h_i)}}
    /// ```
    ///
    /// where $`d_i`$ is the deviance residual, $`\hat\phi`$ is the dispersion, and $`h_i`$ is
    /// the leverage. Generally applicable for outlier detection.
    pub fn resid_dev_std(&self) -> RegressionResult<Array1<F>, F> {
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
        self.resid_work() + x_centered.dot(&self.result_std)
    }

    /// Return the Pearson residuals for each point in the training data:
    ///
    /// ```math
    /// r_i = \sqrt{w_i} \, \frac{y_i - \hat\mu_i}{\sqrt{V(\hat\mu_i)}}
    /// ```
    ///
    /// where $`V`$ is the variance function and $`w_i`$ are the variance weights.
    /// Not scaled by the dispersion for families with a free dispersion parameter.
    pub fn resid_pear(&self) -> &Array1<F> {
        self.resid_pear.get_or_init(|| {
            let residuals = self.resid_resp();
            let inv_var_diag: Array1<F> = self
                .y_hat
                .clone()
                .mapv_into(M::variance)
                .mapv_into(F::recip);
            // the variance weights are the reciprocal of the corresponding variance
            let scales = self
                .data
                .apply_var_weights(inv_var_diag)
                .mapv_into(num_traits::Float::sqrt);
            scales * residuals
        })
    }

    /// Return the standardized Pearson residuals (internally studentized):
    ///
    /// ```math
    /// r_i^* = \frac{r_i}{\sqrt{\hat\phi(1 - h_i)}}
    /// ```
    ///
    /// where $`r_i`$ is the Pearson residual and $`h_i`$ is the leverage. These are expected
    /// to have unit variance.
    pub fn resid_pear_std(&self) -> RegressionResult<Array1<F>, F> {
        let pearson = self.resid_pear();
        let phi = self.dispersion();
        let hat = self.leverage()?;
        let omh = -hat + F::one();
        let denom: Array1<F> = (omh * phi).mapv_into(num_traits::Float::sqrt);
        Ok(pearson / denom)
    }

    /// Return the response residuals: $`e_i^\text{resp} = y_i - \hat\mu_i`$.
    pub fn resid_resp(&self) -> Array1<F> {
        &self.data.y - &self.y_hat
    }

    /// Return the externally studentized residuals:
    ///
    /// ```math
    /// \tilde{t}_i = \text{sign}(e_i) \sqrt{\frac{D_i^{(-i)}}{\hat\phi^{(-i)}}}
    /// ```
    ///
    /// where $`D_i^{(-i)}`$ and $`\hat\phi^{(-i)}`$ are the LOO deviance and dispersion
    /// approximated via one-step deletion. Under normality, $`t`$-distributed with
    /// $`N - K - 1`$ degrees of freedom. This is a robust and general method for outlier
    /// detection.
    pub fn resid_student(&self) -> RegressionResult<Array1<F>, F> {
        let signs = self.resid_resp().mapv(F::signum);
        let dev_terms_loo: Array1<F> = self.deviance_terms_loo()?;
        // NOTE: This match could also be handled internally in dispersion_loo()
        let dev_terms_scaled = match M::DISPERSED {
            // The dispersion is corrected for the contribution from each current point.
            // This is an approximation; the exact solution would perform a fit at each point.
            DispersionType::FreeDispersion => dev_terms_loo / self.dispersion_loo()?,
            DispersionType::NoDispersion => dev_terms_loo,
        };
        Ok(signs * dev_terms_scaled.mapv_into(num_traits::Float::sqrt))
    }

    /// Returns the working residuals:
    ///
    /// ```math
    /// e_i^\text{work} = g'(\hat\mu_i)\,(y_i - \hat\mu_i) = \frac{y_i - \hat\mu_i}{\eta'(\omega_i)\,V(\hat\mu_i)}
    /// ```
    ///
    /// where $`g'(\mu)`$ is the derivative of the link function and $`\eta'(\omega)`$ is the
    /// derivative of the natural parameter with respect to the linear predictor. For canonical
    /// links $`\eta'(\omega) = 1`$, reducing this to $`(y_i - \hat\mu_i)/V(\hat\mu_i)`$.
    ///
    /// These can be interpreted as the residual differences mapped into the linear predictor space
    /// of $`\omega = \mathbf{x}\cdot\boldsymbol{\beta}`$.
    pub fn resid_work(&self) -> Array1<F> {
        let lin_pred: Array1<F> = self.data.linear_predictor_std(&self.result_std);
        let mu: Array1<F> = self.y_hat.clone();
        let resid_response: Array1<F> = &self.data.y - &mu;
        let var: Array1<F> = mu.mapv(M::variance);
        // adjust for non-canonical link functions; we want a total factor of 1/eta'
        let (adj_response, adj_var) =
            M::Link::adjust_errors_variance(resid_response, var, &lin_pred);
        adj_response / adj_var
    }

    /// Returns the score function $`\nabla_{\boldsymbol\beta} l`$ (the gradient of the
    /// regularized log-likelihood) at the parameter values given. Should be zero at the MLE.
    /// The input and output are in the external (unstandardized) parameter space.
    pub fn score(&self, params: Array1<F>) -> Array1<F> {
        // Compute in the internal (standardized) basis so the regularization gradient is applied
        // to the correct parameters, then transform the result back to external coordinates via
        // score_ext = J^T score_int where J = d(beta_std)/d(beta_ext).
        let params_std = self.data.transform_beta(params);
        let lin_pred: Array1<F> = self.data.linear_predictor_std(&params_std);
        let mu: Array1<F> = M::mean(&lin_pred);
        let resid_response = &self.data.y - mu;
        let resid_working = M::Link::adjust_errors(resid_response, &lin_pred);
        let score_int = self.data.x_conj().dot(&resid_working);
        let score_int_reg = self.reg.as_ref().gradient(score_int, &params_std);
        self.data.inverse_transform_score(score_int_reg)
    }

    /// Returns the score test statistic:
    ///
    /// ```math
    /// S = \mathbf{J}(\boldsymbol\beta_0)^\mathsf{T} \, \mathcal{I}(\boldsymbol\beta_0)^{-1} \, \mathbf{J}(\boldsymbol\beta_0)
    /// ```
    ///
    /// where $`\mathbf{J}`$ is the score and $`\mathcal{I}`$ is the Fisher information, both
    /// evaluated at the null parameters. Asymptotically $`\chi^2`$-distributed with
    /// [`test_ndf()`](Self::test_ndf) degrees of freedom.
    pub fn score_test(&self) -> RegressionResult<F, F> {
        let (_, null_params) = self.null_model_fit();
        self.score_test_against(null_params.clone())
    }

    /// Returns the score test statistic compared to another set of model
    /// parameters, not necessarily a null model. The degrees of freedom cannot
    /// be generally inferred.
    pub fn score_test_against(&self, alternative: Array1<F>) -> RegressionResult<F, F> {
        let fisher_alt = self.fisher(&alternative);
        let score_alt = self.score(alternative);
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
        if self.data.has_intercept {
            self.n_par - 1
        } else {
            self.n_par
        }
    }

    /// Returns the Wald test statistic:
    ///
    /// ```math
    /// W = (\hat{\boldsymbol\beta} - \boldsymbol\beta_0)^\mathsf{T} \, \mathcal{I}(\boldsymbol\beta_0) \, (\hat{\boldsymbol\beta} - \boldsymbol\beta_0)
    /// ```
    ///
    /// Compared to a null model with only an intercept (if one is used). Asymptotically
    /// $`\chi^2`$-distributed with [`test_ndf()`](Self::test_ndf) degrees of freedom.
    pub fn wald_test(&self) -> F {
        // The null parameters are all zero except for a possible intercept term
        // which optimizes the null model.
        let (_, null_params) = self.null_model_fit();
        // NOTE: The null model is agnostic to standardization, since it discards all features
        // except perhaps the intercept.
        self.wald_test_against(null_params)
    }

    /// Returns the Wald test statistic compared to another specified model fit
    /// instead of the null model. The degrees of freedom cannot be generally
    /// inferred.
    pub fn wald_test_against(&self, alternative: &Array1<F>) -> F {
        // This could be computed in either the internal or external basis. This implementation
        // will use internal, so first we just need to transform the input.
        let alt_std = self.data.transform_beta(alternative.clone());
        let d_params_std = &self.result_std - &alt_std;
        // Get the fisher matrix at the *other* result, since this is a test of this fit's
        // parameters under a model given by the other.
        let fisher_std: Array2<F> = self.fisher_std(&alt_std);
        d_params_std.t().dot(&fisher_std.dot(&d_params_std))
    }

    /// Returns the per-parameter Wald $`z`$-statistic:
    ///
    /// ```math
    /// z_k = \frac{\hat\beta_k}{\sqrt{\text{Cov}[\hat{\boldsymbol\beta}]_{kk}}}
    /// ```
    ///
    /// Since it does not account for covariance between parameters it may not be accurate.
    pub fn wald_z(&self) -> RegressionResult<Array1<F>, F> {
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
    /// Returns the coefficient of determination $`R^2`$:
    ///
    /// ```math
    /// R^2 = 1 - \frac{\text{RSS}}{\text{TSS}}
    /// ```
    ///
    /// where RSS is the residual sum of squares and TSS is the total sum of squares.
    pub fn r_sq(&self) -> F {
        let y_avg: F = self.data.y.mean().expect("Data should be non-empty");
        let total_sum_sq: F = self.data.y.mapv(|y| y - y_avg).mapv(|dy| dy * dy).sum();
        (total_sum_sq - self.resid_sum_sq()) / total_sum_sq
    }

    /// Returns the residual sum of squares:
    /// $`\text{RSS} = \sum_i (y_i - \hat\mu_i)^2`$.
    pub fn resid_sum_sq(&self) -> F {
        self.resid_resp().mapv_into(|r| r * r).sum()
    }
}

/// Methods that require the `stats` feature for distribution CDF evaluation.
#[cfg(feature = "stats")]
impl<'a, M, F> Fit<'a, M, F>
where
    M: Glm,
    F: 'static + Float,
{
    /// Perform a full re-fit of the model excluding the ith data column.
    /// NOTE: Ideally we could return a full Fit object, but as it stands the source data must
    /// outlive the Fit result. It would be nice if we could get around this with some COW
    /// shenanigans but this will probably require a big change. To get around this for now, just
    /// return the deviance, which is all we're using this function for at the moment.
    fn dev_without_covariate(&self, i: usize) -> RegressionResult<F, F> {
        use ndarray::{concatenate, s};

        let is_intercept = self.data.has_intercept && (i == 0);
        let x_reduced: Array2<F> = if is_intercept {
            // If this is the intercept term, we need to use the original unscaled data to avoid
            // mixing
            self.data.x_ext().slice(s![.., 1..]).to_owned()
        } else {
            concatenate![
                Axis(1),
                self.data.x.slice(s![.., ..i]),
                self.data.x.slice(s![.., i + 1..])
            ]
        };
        // NOTE: This data has been standardized already. Since we are fully dropping a covariate,
        // each data column should still be standardized if the full external set is.
        let data_reduced = Dataset::<F> {
            y: self.data.y.clone(),
            x: x_reduced,
            linear_offset: self.data.linear_offset.clone(),
            weights: self.data.weights.clone(),
            freqs: self.data.freqs.clone(),
            has_intercept: !is_intercept,
            // It shouldn't be necessary to standardize again. Each individual column should
            // already be standardized, if the full dataset was.
            standardizer: None,
        };
        let model_reduced = Model {
            model: PhantomData::<M>,
            data: data_reduced,
        };
        // Start with the non-excluded parameters at the values from the main fit.
        let init_guess = if is_intercept {
            // For the intercept term, we need to use the original unscaled data, or else the
            // transformation mixes the intercept with the other terms. This number is supposed to
            // represent the model deviance from excluding the *external* intercept term.
            self.result.slice(s![1..]).to_owned()
        } else {
            // The data is already standardized in the non-intercept case, so we should start the
            // the standardized parameters, removing the covariate of interest.
            concatenate![
                Axis(0),
                self.result_std.slice(s![..i]),
                self.result_std.slice(s![i + 1..])
            ]
        };
        let fit_options = {
            let mut fit_options = self.options.clone();
            fit_options.init_guess = Some(init_guess);
            fit_options
        };
        let fit_reduced = model_reduced.with_options(fit_options).fit()?;
        Ok(fit_reduced.deviance())
    }

    /// Returns the p-value for the omnibus likelihood-ratio test (full model vs. null/intercept-only
    /// model).
    ///
    /// The LR statistic is asymptotically $`\chi^2`$-distributed with
    /// [`test_ndf()`](Self::test_ndf) degrees of freedom, so the p-value is the upper-tail
    /// probability:
    ///
    /// ```math
    /// p = 1 - F_{\chi^2}(\Lambda;\, \text{test\_ndf})
    /// ```
    pub fn pvalue_lr_test(&self) -> F {
        use statrs::distribution::{ChiSquared, ContinuousCDF};
        let stat = self.lr_test().to_f64().unwrap();
        let ndf = self.test_ndf() as f64;
        if ndf == 0.0 {
            return F::one();
        }
        let chi2 = ChiSquared::new(ndf).unwrap();
        F::from(chi2.sf(stat)).unwrap()
    }

    /// Returns per-parameter p-values from the Wald $`z`$-statistics.
    ///
    /// The reference distribution depends on whether the family has a free dispersion parameter:
    ///
    /// - **No dispersion** (logistic, Poisson): standard normal — two-tailed
    ///   $`p_k = 2\bigl[1 - \Phi(|z_k|)\bigr]`$
    /// - **Free dispersion** (linear): Student-$`t`$ with [`ndf()`](Self::ndf) degrees of
    ///   freedom — two-tailed $`p_k = 2\bigl[1 - F_t(|z_k|;\, \text{ndf})\bigr]`$
    ///
    /// IMPORTANT: Note that this test is an approximation for non-linear models and is known to
    /// sometimes yield misleading values compared to an exact test. It is not hard to find it
    /// give p-values that may imply significantly different conclusions for your analysis (e.g.
    /// p<0.07 vs. p<0.02 in one of our tests).
    pub fn pvalue_wald(&self) -> RegressionResult<Array1<F>, F> {
        use statrs::distribution::ContinuousCDF;
        let z = self.wald_z()?;
        let pvals = match M::DISPERSED {
            DispersionType::NoDispersion => {
                use statrs::distribution::Normal;
                let norm = Normal::standard();
                z.mapv(|zi| {
                    let abs_z = num_traits::Float::abs(zi).to_f64().unwrap();
                    F::from(2.0 * norm.sf(abs_z)).unwrap()
                })
            }
            DispersionType::FreeDispersion => {
                use statrs::distribution::StudentsT;
                let df = self.ndf().to_f64().unwrap();
                let t_dist = StudentsT::new(0.0, 1.0, df).unwrap();
                z.mapv(|zi| {
                    let abs_z = num_traits::Float::abs(zi).to_f64().unwrap();
                    F::from(2.0 * t_dist.sf(abs_z)).unwrap()
                })
            }
        };
        Ok(pvals)
    }

    /// Returns per-parameter p-values from drop-one analysis of deviance (exact, expensive).
    ///
    /// For each parameter $`k`$, a reduced model is fit with that parameter removed and the
    /// deviance difference $`\Delta D = D_{\text{reduced}} - D_{\text{full}}`$ is used:
    ///
    /// - **No dispersion**: $`p = 1 - F_{\chi^2}(\Delta D;\, 1)`$
    /// - **Free dispersion**: $`F = \Delta D / \hat\phi`$, $`p = 1 - F_F(F;\, 1,\,
    ///   \text{ndf})`$
    ///
    /// For the intercept (if present), the reduced model is fit without an intercept. For all
    /// other parameters, the reduced model is fit with that column removed from the design matrix.
    pub fn pvalue_exact(&self) -> RegressionResult<Array1<F>, F> {
        use statrs::distribution::ContinuousCDF;
        let n_par = self.n_par;
        let dev_full = self.deviance();
        let phi_full = self.dispersion();
        let ndf_f64 = self.ndf().to_f64().unwrap();
        let mut pvals = Array1::<F>::zeros(n_par);
        for k in 0..n_par {
            let dev_reduced = self.dev_without_covariate(k)?;
            let delta_d = dev_reduced - dev_full;

            let p = match M::DISPERSED {
                DispersionType::NoDispersion => {
                    use statrs::distribution::ChiSquared;
                    let chi2 = ChiSquared::new(1.0).unwrap();
                    1.0 - chi2.cdf(delta_d.to_f64().unwrap())
                }
                DispersionType::FreeDispersion => {
                    use statrs::distribution::FisherSnedecor;
                    let f_stat = delta_d / phi_full;
                    let f_dist = FisherSnedecor::new(1.0, ndf_f64).unwrap();
                    f_dist.sf(f_stat.to_f64().unwrap())
                }
            };
            pvals[k] = F::from(p).unwrap();
        }

        Ok(pvals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Linear, Logistic, model::ModelBuilder};
    use anyhow::Result;
    use approx::assert_abs_diff_eq;
    use ndarray::Axis;
    use ndarray::{concatenate, s};

    /// Checks if the test statistics are invariant based upon whether the data is standardized.
    #[test]
    fn standardization_invariance() -> Result<()> {
        let data_y = array![true, false, false, true, true, true, true, false, true];
        let data_x = array![-0.5, 0.3, -0.6, 0.2, 0.3, 1.2, 0.8, 0.6, -0.2].insert_axis(Axis(1));
        let lin_off = array![0.1, 0.0, -0.1, 0.2, 0.1, 0.3, 0.4, -0.1, 0.1];
        let model = ModelBuilder::<Logistic>::data(&data_y, &data_x)
            .no_standardize()
            .linear_offset(lin_off.clone())
            .build()?;
        let fit = model.fit()?;
        let model_std = ModelBuilder::<Logistic>::data(&data_y, &data_x)
            .linear_offset(lin_off)
            .build()?;
        let fit_std = model_std.fit()?;
        assert_abs_diff_eq!(&fit.result, &fit_std.result, epsilon = 4.0 * f64::EPSILON);
        assert_abs_diff_eq!(
            fit.covariance()?,
            fit_std.covariance()?,
            epsilon = 0.01 * f32::EPSILON as f64
        );
        let lr = fit.lr_test();
        let lr_std = fit_std.lr_test();
        assert_abs_diff_eq!(lr, lr_std, epsilon = 4.0 * f64::EPSILON);
        assert_abs_diff_eq!(
            fit.score_test()?,
            fit_std.score_test()?,
            epsilon = f32::EPSILON as f64
        );
        assert_abs_diff_eq!(
            fit.wald_test(),
            fit_std.wald_test(),
            epsilon = 4.0 * f64::EPSILON
        );
        assert_abs_diff_eq!(fit.aic(), fit_std.aic(), epsilon = 4.0 * f64::EPSILON);
        assert_abs_diff_eq!(fit.bic(), fit_std.bic(), epsilon = 4.0 * f64::EPSILON);
        assert_abs_diff_eq!(
            fit.deviance(),
            fit_std.deviance(),
            epsilon = 4.0 * f64::EPSILON
        );
        // The Wald Z-score of the intercept term is also invariant, with the new scaling
        // approach, so we can compare the full vectors.
        assert_abs_diff_eq!(
            fit.wald_z()?,
            fit_std.wald_z()?,
            epsilon = 0.01 * f32::EPSILON as f64
        );
        // try p-values
        assert_abs_diff_eq!(
            fit.pvalue_lr_test(),
            fit_std.pvalue_lr_test(),
            epsilon = 0.01 * f32::EPSILON as f64
        );
        assert_abs_diff_eq!(
            fit.pvalue_wald()?,
            fit_std.pvalue_wald()?,
            epsilon = 0.01 * f32::EPSILON as f64
        );
        assert_abs_diff_eq!(
            fit.pvalue_exact()?,
            fit_std.pvalue_exact()?,
            epsilon = 0.01 * f32::EPSILON as f64
        );

        // Ensure that the score and fisher functions are identical even when evaluated at another
        // point. The fit results are near [0.5, 0.5], so pick somewhere not too close.
        let other = array![-0.5, 2.0];
        assert_abs_diff_eq!(fit.score(other.clone()), fit_std.score(other.clone()));
        assert_abs_diff_eq!(fit.fisher(&other), fit_std.fisher(&other.clone()));

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
        let data_y = array![
            true, true, true, true, true, true, false, false, false, false
        ];
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
        let pred_y = fit.predict(&data_x, None);
        let target_dev = (data_y - pred_y).mapv(|dy| dy * dy).sum();
        assert_abs_diff_eq!(fit.deviance(), target_dev, epsilon = 4. * f64::EPSILON);
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
        let weights = array![0.8, 1.2, 0.9, 0.8, 1.1, 0.9];
        // the implied variances from the weights
        let wgt_sigmas = weights.map(|w: &f64| 1. / w.sqrt());
        let model = ModelBuilder::<Linear>::data(&data_y, &data_x)
            .var_weights(weights.clone())
            .build()?;
        let fit = model.fit()?;
        let response = fit.resid_resp();
        let resp_scaled = &response / wgt_sigmas;
        let pearson = fit.resid_pear();
        let deviance = fit.resid_dev();
        assert_abs_diff_eq!(resp_scaled, pearson);
        assert_abs_diff_eq!(resp_scaled, deviance);
        let pearson_std = fit.resid_pear_std()?;
        let deviance_std = fit.resid_dev_std()?;
        assert_abs_diff_eq!(pearson_std, deviance_std, epsilon = 8. * f64::EPSILON);
        // The externally-studentized residuals aren't expected to match the internally-studentized
        // ones.
        let dev_terms_loo = fit.deviance_terms_loo()?;
        let disp_terms_loo = fit.dispersion_loo()?;
        let student = fit.resid_student()?;

        // NOTE: Studentization can't be checked directly in general because the method used is a
        // one-step approximation, however it should be exact in the linear OLS case.
        let n_data = data_y.len();
        // Check that the leave-one-out stats hold literally
        let mut loo_diff: Vec<f64> = Vec::new();
        let mut loo_dev_res: Vec<f64> = Vec::new();
        let mut loo_disp: Vec<f64> = Vec::new();
        for i in 0..n_data {
            let ya = data_y.slice(s![0..i]);
            let yb = data_y.slice(s![i + 1..]);
            let xa = data_x.slice(s![0..i, ..]);
            let xb = data_x.slice(s![i + 1.., ..]);
            let wa = weights.slice(s![0..i]);
            let wb = weights.slice(s![i + 1..]);
            let y_loo = concatenate![Axis(0), ya, yb];
            let x_loo = concatenate![Axis(0), xa, xb];
            let w_loo = concatenate![Axis(0), wa, wb];
            let model_i = ModelBuilder::<Linear>::data(&y_loo, &x_loo)
                .var_weights(w_loo)
                .build()?;
            let fit_i = model_i.fit()?;
            let yi = data_y[i];
            let xi = data_x.slice(s![i..i + 1, ..]);
            let wi = weights[i];
            let yi_pred: f64 = fit_i.predict(&xi, None)[0];
            let disp_i = fit_i.dispersion();
            let var_i = disp_i / wi;
            let diff_i = yi - yi_pred;
            let res_dev_i = diff_i / var_i.sqrt();
            loo_diff.push(wi * diff_i * diff_i);
            loo_disp.push(disp_i);
            loo_dev_res.push(res_dev_i);
        }
        let loo_diff: Array1<f64> = loo_diff.into();
        let loo_disp: Array1<f64> = loo_disp.into();
        let loo_dev_res: Array1<f64> = loo_dev_res.into();
        assert_abs_diff_eq!(dev_terms_loo, loo_diff, epsilon = 8. * f64::EPSILON);
        assert_abs_diff_eq!(disp_terms_loo, loo_disp, epsilon = 8. * f64::EPSILON);
        assert_abs_diff_eq!(student, loo_dev_res, epsilon = 8. * f64::EPSILON);
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
        let model = ModelBuilder::<Linear>::data(&data_y, &data_x)
            .var_weights(weights.clone())
            .build()?;
        let fit = model.fit()?;

        let loo_exact = fit.loo_exact()?;

        let loo_coef: Array2<f64> = fit.infl_coef()?;
        let loo_results = &fit.result - loo_coef;
        let n_data = data_y.len();
        for i in 0..n_data {
            let ya = data_y.slice(s![0..i]);
            let yb = data_y.slice(s![i + 1..]);
            let xa = data_x.slice(s![0..i, ..]);
            let xb = data_x.slice(s![i + 1.., ..]);
            let wa = weights.slice(s![0..i]);
            let wb = weights.slice(s![i + 1..]);
            let y_loo = concatenate![Axis(0), ya, yb];
            let x_loo = concatenate![Axis(0), xa, xb];
            let w_loo = concatenate![Axis(0), wa, wb];
            let model_i = ModelBuilder::<Linear>::data(&y_loo, &x_loo)
                .var_weights(w_loo)
                .build()?;
            let fit_i = model_i.fit()?;
            assert_abs_diff_eq!(
                loo_exact.row(i),
                &fit_i.result,
                epsilon = f32::EPSILON as f64
            );
            assert_abs_diff_eq!(
                loo_results.row(i),
                &fit_i.result,
                epsilon = f32::EPSILON as f64
            );
        }
        Ok(())
    }

    // check the leave-one-out one-step for the logistic model
    #[test]
    fn loo_logistic() -> Result<()> {
        let data_y = array![false, false, true, true, true, false];
        let data_x = array![0.4, 0.1, 0.3, -0.1, 0.5, 0.6].insert_axis(Axis(1));
        let weights = array![1.0, 1.2, 0.8, 1.1, 1.0, 0.7];
        let model = ModelBuilder::<Logistic>::data(&data_y, &data_x)
            .var_weights(weights.clone())
            .build()?;
        let fit = model.fit()?;
        let fit_reg = model.fit_options().l2_reg(0.5).fit()?;

        // NOTE: The one-step approximation fails for non-linear response functions, so we
        // should only test the exact case.
        let loo_exact = fit.loo_exact()?;
        let loo_exact_reg = fit_reg.loo_exact()?;
        let n_data = data_y.len();
        for i in 0..n_data {
            let ya = data_y.slice(s![0..i]);
            let yb = data_y.slice(s![i + 1..]);
            let xa = data_x.slice(s![0..i, ..]);
            let xb = data_x.slice(s![i + 1.., ..]);
            let wa = weights.slice(s![0..i]);
            let wb = weights.slice(s![i + 1..]);
            let y_loo = concatenate![Axis(0), ya, yb];
            let x_loo = concatenate![Axis(0), xa, xb];
            let w_loo = concatenate![Axis(0), wa, wb];
            let model_i = ModelBuilder::<Logistic>::data(&y_loo, &x_loo)
                .var_weights(w_loo)
                .build()?;
            let fit_i = model_i.fit()?;
            let fit_i_reg = model_i.fit_options().l2_reg(0.5).fit()?;
            assert_abs_diff_eq!(
                loo_exact.row(i),
                &fit_i.result,
                epsilon = f32::EPSILON as f64
            );
            assert_abs_diff_eq!(
                loo_exact_reg.row(i),
                &fit_i_reg.result,
                epsilon = f32::EPSILON as f64
            );
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
        let eps = 32.0 * f64::EPSILON;
        assert_abs_diff_eq!(lr, wald, epsilon = eps);
        assert_abs_diff_eq!(lr, score, epsilon = eps);
        // The score vector should be zero at the minimum
        assert_abs_diff_eq!(fit.score(fit.result.clone()), array![0., 0.], epsilon = eps,);
        Ok(())
    }

    // The score should be zero at the MLE even with L2 regularization and internal standardization.
    #[test]
    fn score_zero_at_mle_regularized() -> Result<()> {
        let data_y = array![-0.3, -0.1, 0.0, 0.2, 0.4, 0.5, 0.8, 0.8, 1.1];
        let data_x = array![-0.5, -0.2, 0.1, 0.2, 0.5, 0.6, 0.7, 0.9, 1.3].insert_axis(Axis(1));
        let model = ModelBuilder::<Linear>::data(&data_y, &data_x).build()?;
        let fit = model.fit_options().l2_reg(0.1).fit()?;
        let eps = 1e-8;
        assert_abs_diff_eq!(fit.score(fit.result.clone()), array![0., 0.], epsilon = eps);
        // Also check with standardization disabled
        let model_nostd = ModelBuilder::<Linear>::data(&data_y, &data_x)
            .no_standardize()
            .build()?;
        let fit_nostd = model_nostd.fit_options().l2_reg(0.1).fit()?;
        assert_abs_diff_eq!(
            fit_nostd.score(fit_nostd.result.clone()),
            array![0., 0.],
            epsilon = eps
        );
        // NOTE: Without regularization, the results themselves will not be exactly identical
        // through standardization.
        Ok(())
    }

    /// Test that `pvalue_exact` computes a valid intercept p-value by comparing against R's
    /// `anova(glm(y ~ x - 1), glm(y ~ x), test=...)`.
    ///
    /// R reference (linear):
    /// ```r
    /// y  <- c(0.3, 1.5, 0.8, 2.1, 1.7, 3.2, 2.5, 0.9)
    /// x1 <- c(0.1, 0.5, 0.2, 0.8, 0.6, 1.1, 0.9, 0.3)
    /// x2 <- c(0.4, 0.1, 0.3, 0.7, 0.2, 0.5, 0.8, 0.6)
    /// anova(glm(y ~ x1 + x2 - 1), glm(y ~ x1 + x2), test = "F")
    /// # Pr(>F) = 0.1203156
    /// ```
    ///
    /// R reference (logistic):
    /// ```r
    /// y <- c(1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1)
    /// x <- c(0.5, -0.3, 0.2, 0.8, 0.1, 0.6, -0.1, -0.4, 0.3, 0.4, 0.7, -0.2)
    /// anova(glm(y ~ x - 1, family=binomial()), glm(y ~ x, family=binomial()), test = "Chisq")
    /// # Pr(>Chi) = 0.6042003
    ///
    /// NOTE: This test was generated by claude code, hence the ugly hard-coded values.
    /// ```
    #[cfg(feature = "stats")]
    #[test]
    fn pvalue_exact_intercept() -> Result<()> {
        use crate::Logistic;

        // Linear: intercept F-test should match R's anova() result.
        // For Gaussian the exact (F-test) intercept p-value equals the Wald t-test p-value.
        let y = array![0.3_f64, 1.5, 0.8, 2.1, 1.7, 3.2, 2.5, 0.9];
        let x = array![
            [0.1_f64, 0.4],
            [0.5, 0.1],
            [0.2, 0.3],
            [0.8, 0.7],
            [0.6, 0.2],
            [1.1, 0.5],
            [0.9, 0.8],
            [0.3, 0.6]
        ];
        let model = ModelBuilder::<Linear>::data(&y, &x).build()?;
        let fit = model.fit()?;
        let exact_p = fit.pvalue_exact()?;
        assert!(
            exact_p[0].is_finite() && exact_p[0] >= 0.0 && exact_p[0] <= 1.0,
            "intercept p-value must be in [0, 1]"
        );
        // R: anova(glm(y ~ x1+x2-1), glm(y ~ x1+x2), test="F") Pr(>F) = 0.1203156
        assert_abs_diff_eq!(exact_p[0], 0.1203156, epsilon = 1e-5);

        // Logistic: intercept chi-squared test.
        let y_bin = array![
            true, false, true, true, false, true, false, false, true, false, true, true
        ];
        let x_bin = array![
            0.5_f64, -0.3, 0.2, 0.8, 0.1, 0.6, -0.1, -0.4, 0.3, 0.4, 0.7, -0.2
        ]
        .insert_axis(Axis(1));
        let model_bin = ModelBuilder::<Logistic>::data(&y_bin, &x_bin).build()?;
        let fit_bin = model_bin.fit()?;
        let exact_p_bin = fit_bin.pvalue_exact()?;
        // R: anova(..., test="Chisq") Pr(>Chi) = 0.6042003
        assert_abs_diff_eq!(exact_p_bin[0], 0.6042003, epsilon = 1e-4);

        Ok(())
    }
}
