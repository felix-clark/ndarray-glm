//! Trait defining a generalized linear model for common functionality.
//! Models are fit such that <Y> = g^-1(X*B) where g is the link function.

use crate::irls::IrlsStep;
use crate::link::{Link, Transform};
use crate::{
    error::RegressionResult,
    fit::{options::FitOptions, Fit},
    irls::Irls,
    model::{Dataset, Model},
    num::Float,
};
use ndarray::{Array1, Array2, ArrayRef2};
use ndarray_linalg::SolveH;

/// Whether the model's response has a free dispersion parameter (e.g. linear) or if it is fixed to
/// one (e.g. logistic)
pub enum DispersionType {
    FreeDispersion,
    NoDispersion,
}

/// Trait describing generalized linear model that enables the IRLS algorithm
/// for fitting.
pub trait Glm: Sized {
    /// The link function type of the GLM instantiation. Implementations specify
    /// this manually so that the provided methods can be called in this trait
    /// without necessitating a trait parameter.
    type Link: Link<Self>;

    /// Registers whether the dispersion is fixed at one (e.g. logistic) or free (e.g. linear)
    const DISPERSED: DispersionType;

    /// The link function which maps the expected value of the response variable
    /// to the linear predictor.
    fn link<F: Float>(y: Array1<F>) -> Array1<F> {
        y.mapv(Self::Link::func)
    }

    /// The inverse of the link function which maps the linear predictors to the
    /// expected value of the prediction.
    fn mean<F: Float>(lin_pred: &Array1<F>) -> Array1<F> {
        lin_pred.mapv(Self::Link::func_inv)
    }

    /// The logarithm of the partition function in terms of the natural parameter.
    /// This can be used to calculate the normalized likelihood.
    fn log_partition<F: Float>(nat_par: F) -> F;

    /// The variance as a function of the mean. This should be related to the
    /// Laplacian of the log-partition function, or in other words, the
    /// derivative of the inverse link function mu = g^{-1}(eta). This is unique
    /// to each response function, but should not depend on the link function.
    fn variance<F: Float>(mean: F) -> F;

    /// Get the full adjusted variance diagonal from the linear predictors directly
    fn adjusted_variance_diag<F: Float>(lin_pred: &Array1<F>) -> Array1<F> {
        // The prediction of y given the current model.
        let predictor: Array1<F> = Self::mean(lin_pred);

        // The variances predicted by the model.
        let var_diag: Array1<F> = predictor.mapv(Self::variance);

        Self::Link::adjust_variance(var_diag, lin_pred)
    }

    /// Returns the likelihood function summed over all observations.
    fn log_like<F>(data: &Dataset<F>, regressors: &Array1<F>) -> F
    where
        F: Float,
    {
        // the total likelihood prior to regularization
        let terms = Self::log_like_terms(data, regressors);
        let weighted_terms = data.apply_total_weights(terms);
        weighted_terms.sum()
    }

    /// Returns the likelihood function of the response distribution as a
    /// function of the response variable y and the natural parameters of each
    /// observation. Terms that depend only on the response variable `y` are
    /// dropped. This dispersion parameter is taken to be 1, as it does not
    /// affect the IRLS steps.
    /// The default implementation can be overwritten for performance or numerical
    /// accuracy, but should be mathematically equivalent to the default implementation.
    fn log_like_natural<F>(y: F, nat: F) -> F
    where
        F: Float,
    {
        // subtracting the saturated likelihood to keep the likelihood closer to
        // zero, but this can complicate some fit statistics. In addition to
        // causing some null likelihood tests to fail as written, it would make
        // the current deviance calculation incorrect.
        y * nat - Self::log_partition(nat)
    }

    /// Returns the likelihood of a saturated model where every observation can
    /// be fit exactly.
    fn log_like_sat<F>(y: F) -> F
    where
        F: Float;

    /// Returns the log-likelihood contributions for each observable given the regressor values.
    fn log_like_terms<F>(data: &Dataset<F>, regressors: &Array1<F>) -> Array1<F>
    where
        F: Float,
    {
        let lin_pred = data.linear_predictor(regressors);
        let nat_par = Self::Link::nat_param(lin_pred);
        // the likelihood prior to regularization
        ndarray::Zip::from(&data.y)
            .and(&nat_par)
            .map_collect(|&y, &eta| Self::log_like_natural(y, eta))
    }

    /// Provide an initial guess for the parameters. This can be overridden
    /// but this should provide a decent general starting point. The y data is
    /// averaged with its mean to prevent infinities resulting from application
    /// of the link function:
    /// X * beta_0 ~ g(0.5*(y + y_avg))
    /// This is equivalent to minimizing half the sum of squared differences
    /// between X*beta and g(0.5*(y + y_avg)).
    fn init_guess<F>(data: &Dataset<F>) -> Array1<F>
    where
        F: Float,
        ArrayRef2<F>: SolveH<F>,
    {
        let y_bar: F = data.y.mean().unwrap_or_else(F::zero);
        let mu_y: Array1<F> = data.y.mapv(|y| F::half() * (y + y_bar));
        let link_y = mu_y.mapv(Self::Link::func);
        // Compensate for linear offsets if they are present
        let link_y: Array1<F> = if let Some(off) = &data.linear_offset {
            &link_y - off
        } else {
            link_y
        };
        let x_mat: Array2<F> = data.x_conj().dot(&data.x);
        let init_guess: Array1<F> = x_mat
            .solveh_into(data.x_conj().dot(&link_y))
            .unwrap_or_else(|err| {
                eprintln!("WARNING: failed to get initial guess for IRLS. Will begin at zero.");
                eprintln!("{err}");
                Array1::<F>::zeros(data.x.ncols())
            });
        init_guess
    }

    /// Do the regression and return a result. Returns object holding fit result.
    fn regression<F>(
        model: &Model<Self, F>,
        options: FitOptions<F>,
    ) -> RegressionResult<Fit<'_, Self, F>>
    where
        F: Float,
        Self: Sized,
    {
        let initial: Array1<F> = options
            .init_guess
            .clone()
            .unwrap_or_else(|| Self::init_guess(&model.data));

        let mut irls: Irls<Self, F> = Irls::new(model, initial, options);

        let fit_history: Vec<IrlsStep<F>> = irls.by_ref().collect::<Result<Vec<_>, _>>()?;

        Ok(Fit::new(
            &model.data,
            model.use_intercept,
            irls,
            fit_history,
        ))
    }
}
