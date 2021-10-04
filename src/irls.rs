//! Iteratively re-weighed least squares algorithm
use std::marker::PhantomData;
use crate::glm::Glm;
use crate::link::Transform;
use crate::model::Dataset;
use crate::{
    error::{RegressionError, RegressionResult},
    fit::options::FitOptions,
    num::Float,
};
use ndarray::{Array1, Array2};
use ndarray_linalg::SolveH;

/// Iterate over updates via iteratively re-weighted least-squares until
/// reaching a specified tolerance.
pub struct Irls<'a, M, F>
where
    M: Glm,
    F: Float,
    Array2<F>: SolveH<F>,
{
    model: PhantomData<M>,
    data: &'a Dataset<F>,
    /// The current parameter guess.
    guess: Array1<F>,
    /// The options for the fit
    options: &'a FitOptions<F>,
    /// The number of iterations taken so far
    pub n_iter: usize,
    /// The likelihood for the previous iteration
    last_like: F,
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
    pub fn new(
        data: &'a Dataset<F>,
        initial: Array1<F>,
        options: &'a FitOptions<F>,
        initial_like: F,
    ) -> Self {
        Self {
            model: PhantomData,
            data,
            guess: initial,
            options,
            n_iter: 0,
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
        if self.n_iter > self.options.max_iter {
            return Err(RegressionError::MaxIter(self.options.max_iter));
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
        let lhs: Array2<F> = (*self.options.reg).irls_mat(neg_hessian, &self.guess);
        let rhs: Array1<F> = (*self.options.reg).irls_vec(rhs, &self.guess);
        (lhs, rhs)
    }
}

/// Represents a step in the IRLS. Holds the current guess, likelihood, and the
/// number of steps taken this iteration.
pub struct IrlsStep<F> {
    /// The current parameter guess.
    pub guess: Array1<F>,
    /// The log-likelihood of the current guess.
    pub like: F,
    /// The number of steps taken this iteration. Often equal to 1, but step
    /// halving increases it.
    pub steps: usize,
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
        let mut next_guess: Array1<F> = match irls_mat.solveh_into(irls_vec) {
            Ok(solution) => solution,
            Err(err) => return Some(Err(err.into())),
        };

        // NOTE: might be optimizable by not checking the likelihood until step
        // = next_guess - &self.guess stops decreasing. There could be edge
        // cases that lead to poor convergence.
        // Ideally we could only check the step difference but that might not be
        // as stable. Some parameters might be at different scales.
        let mut next_like = M::log_like_reg(&self.data, &next_guess, self.options.reg.as_ref());
        // This should be positive for an improved guess
        let mut rel =
            (next_like - self.last_like) / (F::epsilon() + num_traits::Float::abs(next_like));
        // If this guess is a strict improvement, return it immediately.
        if rel > F::zero() {
            return Some(self.step_with(next_guess, next_like, 0));
        }
        // Terminate if the difference is close to zero
        if num_traits::Float::abs(rel) <= self.options.tol {
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
        // TODO: None of the tests result in step-halving, so this part is untested.
        let mut step_halves = 0;
        let half: F = F::from(0.5).unwrap();
        let mut step_multiplier = half;
        while rel < -self.options.tol && step_halves < self.options.max_step_halves {
            // The next guess for the step-halving
            let next_guess_sh = next_guess.map(|&x| x * (step_multiplier))
                + &self.guess.map(|&x| x * (F::one() - step_multiplier));
            let next_like_sh =
                M::log_like_reg(&self.data, &next_guess_sh, self.options.reg.as_ref());
            let next_rel = (next_like_sh - self.last_like)
                / (F::epsilon() + num_traits::Float::abs(next_like_sh));
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
