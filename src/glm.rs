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
        F: Float + Lapack,
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
        F: Float + Lapack,
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
        F: Float + Lapack,
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
    fn regression<F>(data: Model<Self, F>) -> RegressionResult<Fit<Self, F>>
    where
        F: Float + Lapack,
        Self: Sized,
    {
        // TODO: determine first element based on fraction of cases in sample
        // This is only a possible improvement when the x points are centered
        // around zero, and may introduce more complications than it's worth. It
        // is further complicated by the possibility of linear offsets.
        // For logistic regression, beta = 0 is typically reasonable.
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

        // TODO: Possibly check if the likelihood is improved by setting each
        // parameter to zero, and if so set it to zero. This could be dependent
        // on the order of operations, however.

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

    // TODO: a function to check if a Y-value is valid? This may be useful for
    // some models. Actually changing the signature of to_float() to return a
    // result should serve this purpose.
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
    /// The current parameter guess.
    guess: Array1<F>,
    /// The maximum iterations before aborting with error.
    max_iter: usize,
    pub n_iter: usize,
    tolerance: F,
    last_like: F,
    /// Sometimes the next guess is better than the previous but within
    /// tolerance, so we want to return the current guess but exit immediately
    /// in the next iteration.
    done: bool,
}

impl<'a, M, F> Irls<'a, M, F>
where
    M: Glm,
    F: Float + Lapack,
    Array2<F>: SolveH<F>,
{
    fn new(data: &'a Model<M, F>, initial: Array1<F>, initial_like: F) -> Self {
        // This tolerance is rather small, but it is used as a fraction of the likelihood.
        let tolerance: F = F::epsilon();
        Self {
            data,
            guess: initial,
            // TODO: make this maximum configurable
            max_iter: data.max_iter.unwrap_or(50),
            n_iter: 0,
            tolerance,
            last_like: initial_like,
            done: false,
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
        let n_steps: usize = 1 + extra_iter;
        self.guess.assign(&next_guess);
        self.last_like = next_like;
        self.n_iter += n_steps;
        if self.n_iter > self.max_iter {
            return Err(RegressionError::MaxIter(self.max_iter));
        }
        Ok(IrlsStep {
            guess: next_guess,
            like: next_like,
            steps: n_steps,
        })
    }

    /// Returns the (LHS, RHS) of the IRLS update matrix equation. This is a bit
    /// faster than computing the Fisher matrix and the Jacobian separately.
    // TODO: re-factor to have the distributions compute the fisher information,
    // as that is useful in the score test as well.
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
        let predictor: Array1<F> = M::mean(&linear_predictor);

        // The variances predicted by the model. This should have weights with
        // it and must be non-zero.
        // This could become a full covariance with weights.
        // TODO: allow the variance conditioning to be a configurable parameter.
        let var_diag: Array1<F> = predictor.mapv(M::variance);

        // The errors represent the difference between observed and predicted.
        let errors = &self.data.y - &predictor;

        // Adjust the errors and variance using the appropriate derivatives of
        // the link function.
        let (errors, var_diag) =
            M::Link::adjust_errors_variance(errors, var_diag, &linear_predictor);
        // Try adjusting only the variance as if the derivative will cancel.
        // This might not be quite right due to the matrix multiplications.
        // let var_diag = M::Link::d_nat_param(&linear_predictor) * var_diag;

        // condition after the adjustment in case the derivatives are zero. Or
        // should the Hessian itself be conditioned?
        let var_diag: Array1<F> = var_diag.mapv_into(|v| v + F::epsilon());

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

/// Represents a step in the IRLS. Holds the current guess, likelihood, and the
/// number of steps taken this iteration.
struct IrlsStep<F> {
    /// The current parameter guess.
    guess: Array1<F>,
    /// The log-likelihood of the current guess.
    like: F,
    /// The number of steps taken this iteration. Often equal to 1, but step
    /// halving increases it.
    steps: usize,
}

impl<'a, M, F> Iterator for Irls<'a, M, F>
where
    M: Glm,
    F: Float + Lapack,
    Array2<F>: SolveH<F>,
{
    type Item = RegressionResult<IrlsStep<F>>;

    /// Acquire the next IRLS step based on the previous one.
    fn next(&mut self) -> Option<Self::Item> {
        // if the last step was an improvement but within tolerance, this step
        // has been flagged to terminate early.
        if self.done {
            return None;
        }

        let (irls_mat, irls_vec) = self.irls_mat_vec();
        let mut next_guess: Array1<F> = match irls_mat.solveh_into(irls_vec) {
            Ok(solution) => solution,
            Err(err) => return Some(Err(err.into())),
        };

        // NOTE: might be optimizable by not checking the likelihood until step
        // = next_guess - &self.guess stops decreasing. There could be edge
        // cases that lead to poor convergence.
        // Ideally we could only check the step difference but that might not be
        // as stable. Some parameters might be at different scales.
        let mut next_like = M::log_like_reg(&self.data, &next_guess);
        // This should be positive for an improved guess
        let mut rel = (next_like - self.last_like) / (F::epsilon() + next_like.abs());
        // If this guess is a strict improvement, return it immediately.
        if rel > F::zero() {
            return Some(self.step_with(next_guess, next_like, 0));
        }
        // Terminate if the difference is close to zero
        if rel.abs() <= self.tolerance {
            // If this guess is an improvement then go ahead and return it, but
            // quit early on the next iteration. The equivalence with zero is
            // necessary in order to return a value when the iteration starts at
            // the best guess. This comparison includes zero so that the
            // iteration terminates if the likelihood hasn't changed at all.
            if rel >= F::zero() {
                self.done = true;
                return Some(self.step_with(next_guess, next_like, 0));
            }
            return None;
        }

        // apply step halving if rel < 0, which means the likelihood has decreased.
        // Don't terminate if rel gets back to within tolerance as a result of this.
        // TODO: make the maximum step halves customizable
        // TODO: None of the tests result in step-halving, so this part is untested.
        const MAX_STEP_HALVES: usize = 8;
        let mut step_halves = 0;
        let half: F = F::from(0.5).unwrap();
        let mut step_multiplier = half;
        while rel < -self.tolerance && step_halves < MAX_STEP_HALVES {
            // The next guess for the step-halving
            let next_guess_sh = next_guess.map(|&x| x * (step_multiplier))
                + &self.guess.map(|&x| x * (F::one() - step_multiplier));
            let next_like_sh = M::log_like_reg(&self.data, &next_guess_sh);
            let next_rel = (next_like_sh - self.last_like) / (F::epsilon() + next_like_sh.abs());
            if next_rel >= rel {
                next_guess = next_guess_sh;
                next_like = next_like_sh;
                rel = next_rel;
                step_multiplier = half;
            } else {
                step_multiplier *= half;
            }
            step_halves += 1;
        }

        if rel > F::zero() {
            Some(self.step_with(next_guess, next_like, step_halves))
        } else {
            // We can end up here if the step direction is a poor one.
            // This signals the end of iteration, but more checks should be done
            // to see how valid the result is.
            None
        }
    }
}
