//! Iteratively re-weighed least squares algorithm
use crate::{
    data::Dataset,
    error::{RegressionError, RegressionResult},
    fit::options::FitOptions,
    glm::Glm,
    link::Transform,
    model::Model,
    num::Float,
    regularization::*,
};
use ndarray::{Array1, Array2, ArrayRef2};
use ndarray_linalg::SolveH;
use std::marker::PhantomData;

/// Iterate over updates via iteratively re-weighted least-squares until
/// reaching a specified tolerance.
pub(crate) struct Irls<'a, M, F>
where
    M: Glm,
    F: Float,
    ArrayRef2<F>: SolveH<F>,
{
    model: PhantomData<M>,
    data: &'a Dataset<F>,
    /// The current parameter guess. No longer public because the full history is passed and this
    /// may not end up being the optimal one. make_last_step() can be used to build an IrlsStep
    /// from the current values, which is primarily used as a fallback when no iterations occur.
    guess: Array1<F>,
    /// The options for the fit
    pub(crate) options: FitOptions<F>,
    /// The regularizer object, which may be stateful
    pub(crate) reg: Box<dyn IrlsReg<F>>,
    /// The number of iterations taken so far
    pub n_iter: usize,
    /// The data likelihood for the previous iteration, unregularized and unaugmented.
    /// This is cached separately from the guess because it demands expensive matrix
    /// multiplications. The augmented and/or regularized terms are relatively cheap, so they
    /// aren't stored.
    last_like_data: F,
    /// Sometimes the next guess is better than the previous but within
    /// tolerance, so we want to return the current guess but exit immediately
    /// in the next iteration.
    done: bool,
    /// Internally track the fit history, and allow the Fit to expose it. Note that this history is
    /// stored on the internal standardized scale as it can know nothing about any external
    /// transformations.
    pub(crate) history: Vec<IrlsStep<F>>,
}

impl<'a, M, F> Irls<'a, M, F>
where
    M: Glm,
    F: Float,
    ArrayRef2<F>: SolveH<F>,
{
    pub(crate) fn new(model: &'a Model<M, F>, initial: Array1<F>, options: FitOptions<F>) -> Self {
        let data = &model.data;
        let reg = get_reg(&options, data.x.ncols(), data.has_intercept);
        let initial_like_data: F = M::log_like(data, &initial);
        Self {
            model: PhantomData,
            data,
            guess: initial,
            options,
            reg,
            n_iter: 0,
            last_like_data: initial_like_data,
            done: false,
            history: Vec::new(),
        }
    }

    /// A helper function to step to a new guess, while incrementing the number
    /// of iterations and checking that it is not over the maximum.
    fn step_with(&mut self, next_guess: Array1<F>, next_like_data: F) -> <Self as Iterator>::Item {
        self.guess.assign(&next_guess);
        self.last_like_data = next_like_data;
        let model_like = next_like_data + self.reg.likelihood(&next_guess);
        let step = IrlsStep {
            guess: next_guess,
            like: model_like,
        };
        self.history.push(step.clone());
        self.n_iter += 1;
        if self.n_iter > self.options.max_iter {
            return Err(RegressionError::MaxIter {
                n_iter: self.options.max_iter,
                history: self.history.clone(),
            });
        }
        Ok(step)
    }

    /// Returns the (LHS, RHS) of the IRLS update matrix equation. This is a bit
    /// faster than computing the Fisher matrix and the Jacobian separately.
    /// The returned matrix and vector are not regularized.
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

        // Similarly, we don't use M::get_adjusted_variance(linear_predictor) because intermediate
        // results are used as well, and we need to correct the error term too.

        // The prediction of y given the current model.
        // This does cause an unnecessary clone with an identity link, but we
        // need the linear predictor around for the future.
        let predictor: Array1<F> = M::mean(&linear_predictor);

        // The variances predicted by the model. This should have weights with
        // it and must be non-zero.
        // This could become a full covariance with weights.
        let var_diag: Array1<F> = predictor.mapv(M::variance);

        // The errors represent the difference between observed and predicted.
        let errors = &self.data.y - &predictor;

        // Adjust the errors and variance using the appropriate derivatives of
        // the link function. With the canonical link function, this is a no-op.
        let (errors, var_diag) =
            M::Link::adjust_errors_variance(errors, var_diag, &linear_predictor);

        // condition after the adjustment in case the derivatives are zero. Or
        // should the Hessian itself be conditioned?
        // TODO: allow the variance conditioning to be a configurable parameter.
        let var_diag: Array1<F> = var_diag.mapv_into(|v| v + F::epsilon());

        // X weighted by the model variance for each observation
        // This is really the negative Hessian of the likelihood.
        let neg_hessian: Array2<F> = (self.data.x_conj() * &var_diag).dot(&self.data.x);

        // This isn't quite the jacobian because the H*beta_old term is subtracted out.
        let rhs: Array1<F> = {
            // NOTE: This w*X should not include the linear offset, because it
            // comes from the Hessian times the last guess.
            let target: Array1<F> = (var_diag * linear_predictor_no_control) + errors;
            let target: Array1<F> = self.data.x_conj().dot(&target);
            target
        };
        (neg_hessian, rhs)
    }

    /// Get the most recent step as a new object. This is most useful as a fallback when there
    /// are no iterations because the iteration started at the maximum, and we need to recover the
    /// default with proper regularization. It could also be used within step_with() itself, after
    /// setting the guess and likelihood, though that could be a little opaque.
    pub(crate) fn make_last_step(&self) -> IrlsStep<F> {
        // It's crucial that this regularization is applied the same way here as it is in
        // step_with().
        let model_like = self.last_like_data + self.reg.likelihood(&self.guess);
        IrlsStep {
            guess: self.guess.clone(),
            like: model_like,
        }
    }
}

/// Represents a step in the IRLS. Holds the current guess and likelihood.
#[derive(Clone, Debug)]
pub struct IrlsStep<F> {
    /// The current parameter guess.
    pub guess: Array1<F>,
    /// The regularized log-likelihood of the current guess.
    pub like: F,
    // TODO: Consider tracking data likelihood, regularized likelihood, and augmented likelihood
    // separately.
}

impl<'a, M, F> Iterator for Irls<'a, M, F>
where
    M: Glm,
    F: Float,
    ArrayRef2<F>: SolveH<F>,
{
    type Item = RegressionResult<IrlsStep<F>, F>;

    /// Acquire the next IRLS step based on the previous one.
    fn next(&mut self) -> Option<Self::Item> {
        // if the last step was an improvement but within tolerance, this step
        // has been flagged to terminate early.
        if self.done {
            return None;
        }

        let (irls_mat, irls_vec) = self.irls_mat_vec();
        let next_guess: Array1<F> = match self.reg.next_guess(&self.guess, irls_vec, irls_mat) {
            Ok(solution) => solution,
            Err(err) => return Some(Err(err)),
        };

        // This is the raw, unregularized and unaugmented
        let next_like_data = M::log_like(self.data, &next_guess);

        // The augmented likelihood to maximize may not be the same as the regularized model
        // likelihood.
        // NOTE: This must be computed after self.reg.next_guess() is called, because that step can
        // change the penalty parameter in ADMM. last_like_obj does not represent the previous
        // objective; it represents the current version of the objective function using the
        // previous guess. These may be different because the augmentation parameter and dual
        // variables for the regularization can change.
        let last_like_obj = self.last_like_data + self.reg.irls_like(&self.guess);
        let next_like_obj = next_like_data + self.reg.irls_like(&next_guess);

        // If the regularization isn't convergent (which should only happen with ADMM in
        // L1/ElasticNet), precision tolerance checks don't matter and we should continue on with
        // the iteration.
        if !self.reg.terminate_ok(self.options.tol) {
            return Some(self.step_with(next_guess, next_like_data));
        }

        // If the next guess is literally equal to the last, then additional IRLS or step-halving
        // procedures won't get use anywhere further.
        // This should be true even under ADMM, given that it's converged per the previous check.
        // This will fire if the optimum is passed in as the initial guess, in which case the
        // iteration will have length zero and the GLM regression function will query this IRLS
        // object for its initial step object.
        if next_guess == self.guess {
            return None;
        }

        // Terminate when both the likelihood change and the parameter step are within tolerance.
        // This check comes before the strict-improvement shortcut so that problems where every
        // step is a tiny improvement (e.g. logistic with y = p_true) converge rather than
        // exhausting max_iter.
        // If ADMM hasn't converged, we've already iterated to the next step, so we don't need to
        // check self.reg.terminate_ok() again.
        let n_obs = F::from(self.data.y.len()).unwrap();
        if small_delta(next_like_obj, last_like_obj, self.options.tol * n_obs)
            && small_delta_vec(&next_guess, &self.guess, self.options.tol)
        {
            self.done = true;
        }

        // If this guess is a strict improvement, continue. If we're within convergence tolerance
        // at this stage, this will be the last step.
        if next_like_obj > last_like_obj {
            return Some(self.step_with(next_guess, next_like_data));
        }

        // apply step halving if the new likelihood is the same or worse as the previous guess.
        // NOTE: It's difficult to engage the step halving because it's rarely necessary, so this
        // part of the algorithm is undertested. It may be more common using L1 regularization.
        // next_guess != self.guess because we've already checked that case above.
        let f_step = |x: F| {
            let b = &next_guess * x + &self.guess * (F::one() - x);
            // The augmented and unaugmented checks should be close to equivalent at this point
            // because the regularization has reported that the internals have converged via
            // `terminate_ok()`. Since we are potentially finding the final best guess, look for
            // the best model likelihood in the step search.
            // NOTE: It's possible there are some edge cases with an inconsistency, given that the
            // checks above use the augmented likelihood and this checks directly.
            M::log_like(self.data, &b) + self.reg.likelihood(&b) // unaugmented
            // M::log_like(self.data, &b) + self.reg.irls_like(&b) // augmented
        };
        let beta_tol_factor = num_traits::Float::sqrt(self.guess.mapv(|b| F::one() + b * b).sum());
        let step_mult: F = step_scale(&f_step, beta_tol_factor * self.options.tol);
        if step_mult.is_zero() {
            // can't find a strictly better minimum if the step multiplier returns zero
            return None;
        }

        // If the step multiplier is not zero, it found a better guess
        let next_guess = &next_guess * step_mult + &self.guess * (F::one() - step_mult);
        let next_like_data = M::log_like(self.data, &next_guess);

        Some(self.step_with(next_guess, next_like_data))
    }
}

fn small_delta<F>(new: F, old: F, tol: F) -> bool
where
    F: Float,
{
    num_traits::Float::abs(new - old) <= tol
}

fn small_delta_vec<F>(new: &Array1<F>, old: &Array1<F>, tol: F) -> bool
where
    F: Float,
{
    // this method interpolates between relative and absolute differences
    let delta = new - old;
    let n = F::from(delta.len()).unwrap();

    let new2: F = new.mapv(|d| d * d).sum();
    let delta2: F = delta.mapv(|d| d * d).sum();

    // use sum of absolute values to indicate magnitude of beta
    // sum of squares might be better
    delta2 <= (n + new2) * tol * tol
}

/// Zero the first element of the array `l` if `use_intercept == true`
fn zero_first_maybe<F>(mut l: Array1<F>, use_intercept: bool) -> Array1<F>
where
    F: Float,
{
    // if an intercept term is included it should not be subject to
    // regularization.
    if use_intercept {
        l[0] = F::zero();
    }
    l
}

/// Generate a regularizer from the set of options
fn get_reg<F: Float>(
    options: &FitOptions<F>,
    n: usize,
    use_intercept: bool,
) -> Box<dyn IrlsReg<F>> {
    if options.l1 < F::zero() || options.l2 < F::zero() {
        eprintln!("WARNING: regularization parameters should not be negative.");
    }
    let use_l1 = options.l1 > F::zero();
    let use_l2 = options.l2 > F::zero();

    if use_l1 && use_l2 {
        let l1_diag: Array1<F> = Array1::<F>::from_elem(n, options.l1);
        let l1_diag: Array1<F> = zero_first_maybe(l1_diag, use_intercept);
        let l2_diag: Array1<F> = Array1::<F>::from_elem(n, options.l2);
        let l2_diag: Array1<F> = zero_first_maybe(l2_diag, use_intercept);
        Box::new(ElasticNet::from_diag(l1_diag, l2_diag))
    } else if use_l2 {
        let l2_diag: Array1<F> = Array1::<F>::from_elem(n, options.l2);
        let l2_diag: Array1<F> = zero_first_maybe(l2_diag, use_intercept);
        Box::new(Ridge::from_diag(l2_diag))
    } else if use_l1 {
        let l1_diag: Array1<F> = Array1::<F>::from_elem(n, options.l1);
        let l1_diag: Array1<F> = zero_first_maybe(l1_diag, use_intercept);
        Box::new(Lasso::from_diag(l1_diag))
    } else {
        Box::new(Null {})
    }
}

/// Find a better step scale to optimize and objective function.
/// Looks for a new solution better than x = 1 looking first at 0 < x < 1 and returning any value
/// found to be a strict improvement.
/// If none are found, it will check a single negative step.
fn step_scale<F: Float>(f: &dyn Fn(F) -> F, tol: F) -> F {
    let tol = num_traits::Float::abs(tol);
    // TODO: Add list of values to explicitly try (for instance with zeroed parameters)

    let zero: F = F::zero();
    let one: F = F::one();
    // `scale = 0.5` should also work, but using the golden ratio is prettier and might be less
    // likely to fail in pathological cases.
    let scale = F::from(0.618033988749894).unwrap();
    let mut x: F = one;
    let f0: F = f(zero);

    // start at scale < 1, since 1 has already been checked.
    while x > tol {
        x *= scale;
        let fx = f(x);
        if fx > f0 {
            return x;
        }
    }

    // If f(1) > f(0), then an improvement has already been found. However, if the optimization is
    // languishing, it could be useful to try x > 1. It's pretty rare to get to this state,
    // however.

    // If we're here a strict improvement hasn't been found, but it's possible that the likelihoods
    // are equal.
    // check a single step in the negative direction, in case this is an improvement.
    if f(-scale) > f0 {
        return -scale;
    }

    // Nothing checked has been an improvement, so return zero.
    F::zero()
}
