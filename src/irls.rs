//! Iteratively re-weighed least squares algorithm
use crate::glm::Glm;
use crate::link::Transform;
use crate::model::{Dataset, Model};
use crate::regularization::{ElasticNet, Lasso, Null, Ridge};
use crate::{
    error::{RegressionError, RegressionResult},
    fit::options::FitOptions,
    num::Float,
    regularization::IrlsReg,
};
use ndarray::{Array1, Array2};
use ndarray_linalg::SolveH;
use std::marker::PhantomData;

/// Iterate over updates via iteratively re-weighted least-squares until
/// reaching a specified tolerance.
pub(crate) struct Irls<'a, M, F>
where
    M: Glm,
    F: Float,
    Array2<F>: SolveH<F>,
{
    model: PhantomData<M>,
    data: &'a Dataset<F>,
    /// The current parameter guess.
    pub(crate) guess: Array1<F>,
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
    pub(crate) last_like_data: F,
    /// Sometimes the next guess is better than the previous but within
    /// tolerance, so we want to return the current guess but exit immediately
    /// in the next iteration.
    done: bool,
}

impl<'a, M, F> Irls<'a, M, F>
where
    M: Glm,
    F: Float,
    Array2<F>: SolveH<F>,
{
    pub fn new(model: &'a Model<M, F>, initial: Array1<F>, options: FitOptions<F>) -> Self {
        let data = &model.data;
        let reg = get_reg(&options, data.x.ncols(), model.use_intercept);
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
        }
    }

    /// A helper function to step to a new guess, while incrementing the number
    /// of iterations and checking that it is not over the maximum.
    fn step_with(&mut self, next_guess: Array1<F>, next_like_data: F) -> <Self as Iterator>::Item {
        self.guess.assign(&next_guess);
        self.last_like_data = next_like_data;
        let model_like = next_like_data + self.reg.likelihood(&next_guess);
        self.n_iter += 1;
        if self.n_iter > self.options.max_iter {
            // NOTE: This could also return the best guess so far. Including the data in the error
            // type would necessitate either a conversion to f32 or a parameterization.
            return Err(RegressionError::MaxIter(self.options.max_iter));
        }
        Ok(IrlsStep {
            guess: next_guess,
            like: model_like,
        })
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
        (neg_hessian, rhs)
    }
}

/// Represents a step in the IRLS. Holds the current guess, likelihood, and the
/// number of steps taken this iteration.
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

        // NOTE: might be optimizable by not checking the likelihood until step
        // = next_guess - &self.guess stops decreasing. There could be edge
        // cases that lead to poor convergence.
        // Ideally we could only check the step difference but that might not be
        // as stable. Some parameters might be at different scales.

        // If this guess is a strict improvement, return it immediately.
        if next_like_obj > last_like_obj {
            return Some(self.step_with(next_guess, next_like_data));
        }

        // Indicates if the likelihood change is small, within tolerance, even if it is not
        // positive.
        let small_delta_like = small_delta(next_like_obj, last_like_obj, self.options.tol);

        // If the parameters have changed significantly but the likelihood hasn't improved,
        // step halving needs to be engaged. The parameter delta should probably ideally be
        // tested using the spread of the covariate data, but in principle the data can be
        // standardized so this will just compare to the raw tolerance.
        let small_delta_guess = small_delta_vec(&next_guess, &self.guess, self.options.tol);

        // Terminate if the difference is close to zero and the parameters haven't changed
        // significantly.
        if small_delta_like && small_delta_guess {
            // If this guess is an improvement then go ahead and return it, but
            // quit early on the next iteration. The equivalence with zero is
            // necessary in order to return a value when the iteration starts at
            // the best guess. This comparison includes zero so that the
            // iteration terminates if the likelihood hasn't changed at all.
            if next_like_obj >= last_like_obj {
                // assert_eq!(next_like_obj, last_like_obj); // this should still hold
                self.done = true;
                return Some(self.step_with(next_guess, next_like_data));
            }
            return None;
        }

        // Don't go through step halving if the regularization isn't convergent
        if !self.reg.terminate_ok(self.options.tol) {
            return Some(self.step_with(next_guess, next_like_data));
        }

        // apply step halving if the new likelihood is the same or worse as the previous guess.
        // NOTE: It's difficult to engage the step halving because it's rarely necessary, so this
        // part of the algorithm is undertested. It may be more common using L1 regularization.
        let f_step = |x: F| {
            let b = &next_guess * x + &self.guess * (F::one() - x);
            // Using the real likelihood in the step finding avoids potential issues with the
            // augmentation. They should be close to equivalent at this point because the
            // regularization has reported that the internals have converged.
            M::log_like(self.data, &b) + self.reg.likelihood(&b)
        };
        let beta_tol_factor = num_traits::Float::sqrt(self.guess.mapv(|b| F::one() + b * b).sum());
        let step_mult: F = step_scale(&f_step, beta_tol_factor * self.options.tol);
        if step_mult.is_zero() {
            // can't find a better minimum if the step multiplier returns zero
            return None;
        }
        // If step_mult == 1, that means the guess is a good one according to the un-augmented
        // regularized likelihood, so go ahead and use it.

        // If the step multiplier is not zero, it found a better guess
        let next_guess = &next_guess * step_mult + &self.guess * (F::one() - step_mult);
        let next_like_data = M::log_like(self.data, &next_guess);
        let next_like = M::log_like(self.data, &next_guess) + self.reg.likelihood(&next_guess);
        let last_like = self.last_like_data + self.reg.likelihood(&self.guess);
        if next_like < last_like {
            return None;
        }

        let small_delta_like = small_delta(next_like, last_like, self.options.tol);
        let small_delta_guess = small_delta_vec(&next_guess, &self.guess, self.options.tol);
        if small_delta_like && small_delta_guess {
            self.done = true;
        }

        Some(self.step_with(next_guess, next_like_data))
    }
}

fn small_delta<F>(new: F, old: F, tol: F) -> bool
where
    F: Float,
{
    let rel = (new - old) / (F::epsilon() + num_traits::Float::abs(new));
    num_traits::Float::abs(rel) <= tol
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
    // `scale = 0.5` should also work, but using the golden ratio is prettier.
    let scale = F::from(0.618033988749894).unwrap();
    let mut x: F = one;
    let f0: F = f(zero);

    while x > tol {
        let fx = f(x);
        if fx > f0 {
            return x;
        }
        x *= scale;
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

    x
}
