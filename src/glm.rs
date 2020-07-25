//! trait defining a generalized linear model and providing common functionality
//! Models are fit such that E[Y] = g^-1(X*B) where g is the link function.

use crate::link::{Link, Transform};
use crate::{error::RegressionResult, fit::Fit, irls::Irls, model::Model, num::Float};
use ndarray::{Array1, Array2};
use ndarray_linalg::SolveH;

/// Trait describing generalized linear model that enables the IRLS algorithm
/// for fitting.
pub trait Glm: Sized {
    /// The link function type of the GLM instantiation. Implementations specify
    /// this manually so that the provided methods can be called in this trait
    /// without necessitating a trait parameter.
    type Link: Link<Self>;

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
    /// This can be used to calculate the likelihood generally. All input terms
    /// are summed over in the result.
    fn log_partition<F: Float>(nat_par: &Array1<F>) -> F;

    /// The variance as a function of the mean. This should be related to the
    /// Laplacian of the log-partition function, or in other words, the
    /// derivative of the inverse link function mu = g^{-1}(eta). This is unique
    /// to each response function, but should not depend on the link function.
    fn variance<F: Float>(mean: F) -> F;

    /// Returns the likelihood function of the response distribution as a
    /// function of the response variable y and the natural parameters of each
    /// observation. Terms that depend only on the response variable `y` are
    /// dropped. This dispersion parameter is taken to be 1, as it does not
    /// affect the IRLS steps.
    // TODO: A default implementation could be written in terms of the log
    // partition function, but in some cases this could be more expensive (?).
    fn log_like_natural<F>(y: &Array1<F>, nat: &Array1<F>) -> F
    where
        F: Float,
    {
        // subtracting the saturated likelihood to keep the likelihood closer to
        // zero, but this can complicate some fit statistics. In addition to
        // causing some null likelihood tests to fail as written, it would make
        // the deviance calculation incorrect.
        (y * nat).sum() - Self::log_partition(nat)
    }

    /// Returns the likelihood of a saturated model where every observation can
    /// be fit exactly.
    fn log_like_sat<F>(y: &Array1<F>) -> F
    where
        F: Float;

    /// Returns the likelihood function including regularization terms.
    fn log_like_reg<F>(data: &Model<Self, F>, regressors: &Array1<F>) -> F
    where
        F: Float,
    {
        let lin_pred = data.linear_predictor(&regressors);
        // the likelihood prior to regularization
        let l_unreg = Self::log_like_natural(&data.y, &Self::Link::nat_param(lin_pred));
        (*data.reg).likelihood(l_unreg, regressors)
    }

    /// Provide an initial guess for the parameters. This can be overridden
    /// but this should provide a decent general starting point. The y data is
    /// averaged with its mean to prevent infinities resulting from application
    /// of the link function:
    /// X * beta_0 ~ g(0.5*(y + y_avg))
    /// This is equivalent to minimizing half the sum of squared differences
    /// between X*beta and g(0.5*(y + y_avg)).
    // TODO: consider incorporating weights and/or correlations.
    fn init_guess<F>(data: &Model<Self, F>) -> Array1<F>
    where
        F: Float,
        Array2<F>: SolveH<F>,
    {
        let y_bar: F = data.y.mean().unwrap_or_else(F::zero);
        let mu_y: Array1<F> = data.y.mapv(|y| F::from(0.5).unwrap() * (y + y_bar));
        let link_y = mu_y.mapv(Self::Link::func);
        // Compensate for linear offsets if they are present
        let link_y: Array1<F> = if let Some(off) = &data.linear_offset {
            &link_y - off
        } else {
            link_y
        };
        let x_mat: Array2<F> = data.x.t().dot(&data.x);
        let init_guess: Array1<F> =
            x_mat
                .solveh_into(data.x.t().dot(&link_y))
                .unwrap_or_else(|err| {
                    eprintln!("WARNING: failed to get initial guess for IRLS. Will begin at zero.");
                    eprintln!("{}", err);
                    Array1::<F>::zeros(data.x.ncols())
                });
        init_guess
    }

    /// Do the regression and return a result. Returns object holding fit result.
    fn regression<F>(data: &Model<Self, F>) -> RegressionResult<Fit<Self, F>>
    where
        F: Float,
        Self: Sized,
    {
        let initial: Array1<F> = Self::init_guess(&data);

        // This represents the number of overall iterations
        let mut n_iter: usize = 0;
        // This is the number of steps tried, which includes those arising from step halving.
        let mut n_steps: usize = 0;
        // initialize the result and likelihood in case no steps are taken.
        let mut result: Array1<F> = initial.clone();
        let mut model_like: F = Self::log_like_reg(&data, &initial);

        let irls: Irls<Self, F> = Irls::new(&data, initial, model_like);

        for iteration in irls {
            let it_result = iteration?;
            result.assign(&it_result.guess);
            model_like = it_result.like;
            // This number of iterations does not include any extras from step halving.
            n_iter += 1;
            n_steps += it_result.steps;
        }

        Ok(Fit::new(data, result, model_like, n_iter, n_steps))
    }
}

/// Describes the domain of the response variable for a GLM, e.g. integer for
/// Poisson, float for Linear, bool for logistic. Implementing this trait for a
/// type Y shows how to convert to a floating point type and allows that type to
/// be used as a response variable.
pub trait Response<M: Glm> {
    /// Converts the domain to a floating-point value for IRLS.
    fn to_float<F: Float>(self) -> RegressionResult<F>;
}
