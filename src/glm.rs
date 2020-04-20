//! trait defining a generalized linear model and providing common functionality
//! Models are fit such that E[Y] = g^-1(X*B) where g is the link function.

use crate::link::{Link, Transform};
use crate::{
    error::{RegressionError, RegressionResult},
    fit::Fit,
    model::Model,
};
use ndarray::{Array1, Array2};
use ndarray_linalg::{lapack::Lapack, SolveH};
use num_traits::Float;
use std::marker::PhantomData;

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
        Self::Link::func(y)
    }

    /// The inverse of the link function which maps the linear predictors to the
    /// expected value of the prediction.
    fn mean<F: Float>(lin_pred: Array1<F>) -> Array1<F> {
        Self::Link::func_inv(lin_pred)
    }

    /// The variance as a function of the mean. This should be related to the
    /// Laplacian of the log-partition function, or in other words, the
    /// derivative of the inverse link function mu = g^{-1}(eta). This is unique
    /// to each response function, but should not depend on the link function.
    fn variance<F: Float>(mean: F) -> F;

    // /// Returns the likelihood function of the response distribution as a
    // /// function of the regression parameters.
    // TODO: A default implementation could be defined in terms of the
    // log-partition function.
    // TODO: change the interface to only take y and the natural parameter eta.
    // This will allow the Glm trait functionality to handle a non-canonical
    // transformation.
    // fn log_like_params<F>(data: &Model<Self, F>, regressors: &Array1<F>) -> F
    // where
    //     F: Float + Lapack;

    /// Returns the likelihood function of the response distribution as a
    /// function of the response variable y and the natural parameters of each
    /// observation. Terms that depend only on the response variable `y` are
    /// dropped. This dispersion parameter is taken to be 1, as it does not
    /// affect the IRLS steps.
    // TODO: A default implementation could be written in terms of the log
    // partition function, but in some cases this could be more expensive (?).
    fn log_like_natural<F>(y: &Array1<F>, nat: &Array1<F>) -> F
    where
        F: Float + Lapack;

    /// Returns the likelihood function including regularization terms.
    fn log_like_reg<F>(data: &Model<Self, F>, regressors: &Array1<F>) -> F
    where
        F: Float + Lapack,
    {
        let lin_pred = data.linear_predictor(&regressors);
        // the likelihood prior to regularization
        let l_unreg = Self::log_like_natural(&data.y, &Self::Link::nat_param(lin_pred));
        (*data.reg).likelihood(l_unreg, regressors)
    }

    /// Do the regression and return a result. Returns object holding fit result.
    fn regression<F>(data: &Model<Self, F>) -> RegressionResult<Fit<Self, F>>
    where
        F: Float + Lapack,
        Self: Sized,
    {
        let n_data = data.y.len();

        // TODO: determine first element based on fraction of cases in sample
        // This is only a possible improvement when the x points are centered
        // around zero, and may introduce more complications than it's worth. It
        // is further complicated by the possibility of linear offsets.
        // For logistic regression, beta = 0 is typically reasonable.
        let initial: Array1<F> = Array1::<F>::zeros(data.x.ncols());
        let mut n_iter: usize = 0;

        let mut result: Array1<F> = Array1::<F>::zeros(initial.len());

        let irls: Irls<Self, F> = Irls::new(&data, initial);

        for iteration in irls {
            // TODO: This assignment at every loop is probably unnecessary since
            // we can simply grab the last guess of the IRLS.
            result.assign(&iteration?);
            n_iter += 1;
        }

        // TODO: Possibly check if the likelihood is improved by setting each
        // parameter to zero, and if so set it to zero. This could be dependent
        // on the order of operations, however.

        // ndf is guaranteed to be > 0 because of the underconstrained check
        let ndf = n_data - result.len();
        Ok(Fit {
            model: PhantomData::<Self>,
            result,
            ndf,
            n_iter,
        })
    }
}

/// Describes the domain of the response variable for a GLM, e.g. integer for
/// Poisson, float for Linear, bool for logistic. Implementing this trait for a
/// type Y shows how to convert to a floating point type and allows that type to
/// be used as a response variable.
pub trait Response<M: Glm> {
    /// Converts the domain to a floating-point value for IRLS.
    fn to_float<F: Float>(self) -> F;

    // TODO: a function to check if a Y-value is valid? This may be useful for
    // some models. Actually changing the signature of to_float() to return a
    // result should serve this purpose.
}

/// A subtrait for GLMs that have an unambiguous likelihood function.
// Not all regression types have a well-defined likelihood. E.g. logistic
// (binomial) and Poisson do; linear (normal) and negative binomial do not due
// to the extra parameter. If the dispersion term can be calculated, this can be
// fixed, although it will be best to separate the true likelihood from an
// effective one for minimization.
// TODO: This trait should be phased out, but it affects the Z-scores. That will
// be changing too.
pub trait Likelihood<M, F>: Glm
where
    M: Glm,
    F: Float,
{
    /// logarithm of the likelihood given the data and fit parameters
    fn log_likelihood(data: &Model<M, F>, regressors: &Array1<F>) -> F;
}

/// Iterate over updates via iteratively re-weighted least-squares until
/// reaching a specified tolerance.
struct Irls<'a, M, F>
where
    M: Glm,
    F: Float + Lapack,
    Array2<F>: SolveH<F>,
{
    data: &'a Model<M, F>,
    guess: Array1<F>,
    max_iter: usize,
    pub n_iter: usize,
    tolerance: F,
    last_like: F,
}

impl<'a, M, F> Irls<'a, M, F>
where
    M: Glm,
    F: Float + Lapack,
    Array2<F>: SolveH<F>,
{
    fn new(data: &'a Model<M, F>, initial: Array1<F>) -> Self {
        let tolerance: F = F::epsilon(); // * F::from(data.y.len()).unwrap();
        let init_like = M::log_like_reg(&data, &initial);
        Self {
            data,
            guess: initial,
            max_iter: data.max_iter.unwrap_or(50),
            n_iter: 0,
            // As a ratio with the variance, this epsilon could be too small.
            tolerance,
            last_like: init_like,
        }
    }

    /// A helper function to step to a new guess, while incrementing the number
    /// of iterations and checking that it is not over the maximum.
    fn step_with(
        &mut self,
        next_guess: Array1<F>,
        next_like: F,
        extra_iter: usize,
    ) -> <Self as Iterator>::Item {
        self.guess.assign(&next_guess);
        self.last_like = next_like;
        self.n_iter += 1 + extra_iter;
        if self.n_iter > self.max_iter {
            return Err(RegressionError::MaxIter(self.max_iter));
        }
        Ok(next_guess)
    }

    /// Returns the (LHS, RHS) of the IRLS update matrix equation.
    fn irls_mat_vec(&self) -> (Array2<F>, Array1<F>) {
        // The linear predictor without control terms
        let linear_predictor_no_control: Array1<F> = self.data.x.dot(&self.guess);
        // the linear predictor given the model, including offsets if present
        let linear_predictor = match &self.data.linear_offset {
            Some(off) => &linear_predictor_no_control + off,
            None => linear_predictor_no_control.clone(),
        };
        // The data.linear_predictor() function is not used above because we will use
        // both versions, with and without the linear offset, and we don't want
        // to repeat the matrix multiplication.

        // The prediction of y given the current model.
        // This does cause an unnecessary clone with an identity link, but we
        // need the linear predictor around for the future.
        let predictor: Array1<F> = M::mean(linear_predictor.clone());

        // The variances predicted by the model. This should have weights with
        // it and must be non-zero.
        // This could become a full covariance with weights.
        // TODO: allow the variance conditioning to be a configurable parameter.
        let var_diag: Array1<F> = predictor.mapv(|mu| M::variance(mu) + F::epsilon());

        // The errors represent the difference between observed and predicted.
        let errors = &self.data.y - &predictor;

        // Adjust the errors and variance using the appropriate derivatives of
        // the link function.
        let (errors, var_diag) =
            M::Link::adjust_errors_variance(errors, var_diag, &linear_predictor);

        // X weighted by the model variance for each observation
        // This is really the negative Hessian of the likelihood.
        // When adding correlations between observations this statement will
        // need to be modified.
        let neg_hessian: Array2<F> = (&self.data.x.t() * &var_diag).dot(&self.data.x);

        // This isn't quite the jacobian because the H*beta_old term is subtracted out.
        let rhs: Array1<F> = {
            // NOTE: This w*X should not include the linear offset, because it
            // comes from the Hessian times the last guess.
            let target: Array1<F> = (var_diag * linear_predictor_no_control) + errors;
            let target: Array1<F> = self.data.x.t().dot(&target);
            target
        };
        // Regularize the matrix and vector terms
        let lhs: Array2<F> = (*self.data.reg).irls_mat(neg_hessian, &self.guess);
        let rhs: Array1<F> = (*self.data.reg).irls_vec(rhs, &self.guess);
        (lhs, rhs)
    }
}

impl<'a, M, F> Iterator for Irls<'a, M, F>
where
    M: Glm,
    F: Float + Lapack,
    Array2<F>: SolveH<F>,
{
    type Item = RegressionResult<Array1<F>>;

    /// Acquire the next IRLS step based on the previous one.
    fn next(&mut self) -> Option<Self::Item> {
        let (irls_mat, irls_vec) = self.irls_mat_vec();
        // let mut next_guess: Array1<F> = match neg_hessian.solveh_into(rhs) {
        let mut next_guess: Array1<F> = match irls_mat.solveh_into(irls_vec) {
            Ok(solution) => solution,
            Err(err) => return Some(Err(err.into())),
        };

        // NOTE: might be optimizable by not checking the likelihood until step
        // = next_guess - &self.guess stops decreasing. There could be edge
        // cases that lead to poor convergence.
        // Ideally we could only check the step difference but that might not be
        // as stable. Some parameters might be at different scales.
        let mut like = M::log_like_reg(&self.data, &next_guess);
        // This should be positive for an improved guess
        let mut rel = (like - self.last_like) / (F::epsilon() + like.abs());
        // Terminate if the difference is close to zero
        if rel.abs() <= self.tolerance {
            return None;
        }

        // apply step halving if rel < 0, which means the likelihood has decreased.
        // Don't terminate if rel gets back to within tolerance as a result of this.
        // TODO: make the maximum step halves customizable
        const MAX_STEP_HALVES: usize = 6;
        let mut step_halves = 0;
        let half: F = F::from(0.5).unwrap();
        let mut step_multiplier = half;
        while rel < -self.tolerance && step_halves < MAX_STEP_HALVES {
            let next_guess_sh = next_guess.map(|&x| x * (step_multiplier))
                + &self.guess.map(|&x| x * (F::one() - step_multiplier));
            like = M::log_like_reg(&self.data, &next_guess);
            let next_rel = (like - self.last_like) / (F::epsilon() + like.abs());
            if next_rel >= rel {
                next_guess = next_guess_sh;
                rel = next_rel;
                step_multiplier = half;
            } else {
                step_multiplier *= half;
            }
            step_halves += 1;
        }

        if rel > F::zero() {
            Some(self.step_with(next_guess, like, step_halves))
        } else {
            // We can end up here if the step direction is a poor one.
            // This signals the end of iteration, but more checks should be done
            // to see how valid the result is.
            None
        }
    }
}
