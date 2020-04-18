//! trait defining a generalized linear model and providing common functionality
//! Models are fit such that E[Y] = g^-1(X*B) where g is the link function.

use crate::{error::RegressionError, fit::Fit, model::Model};
use ndarray::{Array1, Array2, ArrayViewMut1};
use ndarray_linalg::{lapack::Lapack, SolveH};
use num_traits::Float;
use std::marker::PhantomData;

/// Trait describing generalized linear model that enables the IRLS algorithm
/// for fitting.
pub trait Glm {
    /// the link function
    // fn link<F: 'static + Float>(y: Self::Domain) -> F;
    fn link<F: Float>(y: F) -> F;

    // TODO: return both mean and variance as function of eta at once and avoid FPE

    /// inverse link function which maps the linear predictors to the expected value of the prediction.
    fn mean<F: Float>(x: F) -> F;

    /// The variance as a function of the mean. This should be related to the
    /// Laplacian of the log-partition function, or in other words, the
    /// derivative of the inverse link function mu = g^{-1}(eta).
    fn variance<F: Float>(mean: F) -> F;

    /// Returns the log-likelihood if it is well-defined. If not (like in
    /// unweighted OLS) returns an objective function to be maximized.
    fn quasi_log_likelihood<F>(data: &Model<Self, F>, regressors: &Array1<F>) -> F
    where
        F: Float + Lapack,
        Self: Sized;

    /// Do the regression and return a result. Returns object holding fit result.
    fn regression<F>(data: &Model<Self, F>) -> Result<Fit<Self, F>, RegressionError>
    where
        F: 'static + Float + Lapack,
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
    // some models.
}

/// A subtrait for GLMs that have an unambiguous likelihood function.
// Not all regression types have a well-defined likelihood. E.g. logistic
// (binomial) and Poisson do; linear (normal) and negative binomial do not due
// to the extra parameter. If the dispersion term can be calculated, this can be
// fixed, although it will be best to separate the true likelihood from an
// effective one for minimization.
// pub trait Likelihood<M, F>: Glm<F>
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
    // M: Glm<F>,
    M: Glm,
    F: 'static + Float + Lapack,
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
    // M: Glm<F>,
    M: Glm,
    F: 'static + Float + Lapack,
    Array2<F>: SolveH<F>,
{
    fn new(data: &'a Model<M, F>, initial: Array1<F>) -> Self {
        let tolerance: F = F::epsilon(); // * F::from(data.y.len()).unwrap();
        let init_like = M::quasi_log_likelihood(&data, &initial);
        Self {
            data,
            guess: initial,
            max_iter: data.max_iter.unwrap_or(50),
            n_iter: 0,
            // As a ratio with the variance, this epsilon could be too small.
            tolerance,
            // last_like: -F::infinity(),
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
}

impl<'a, M, F> Iterator for Irls<'a, M, F>
where
    // M: Glm<F>,
    M: Glm,
    F: 'static + Float + Lapack,
    Array2<F>: SolveH<F>,
{
    type Item = Result<Array1<F>, RegressionError>;

    fn next(&mut self) -> Option<Self::Item> {
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
        let predictor: Array1<F> = linear_predictor.mapv(M::mean);

        // The variances predicted by the model. This should have weights with
        // it and must be non-zero.
        // This could become a full covariance with weights.
        // TODO: implement a mean_and_var function to return both the mean and
        // variance while avoiding over/underflow.
        let var_diag: Array1<F> = predictor.mapv(|mu| M::variance(mu) + F::epsilon());

        // X weighted by the model variance for each observation
        // This is really the negative Hessian of the likelihood.
        // When adding correlations between observations this statement will
        // need to be modified.
        let neg_hessian: Array2<F> = {
            let mut hess: Array2<F> = (&self.data.x.t() * &var_diag).dot(&self.data.x);
            // If L2 regularization is set, add lambda * I to the Hessian.
            let mut hess_diag: ArrayViewMut1<F> = hess.diag_mut();
            hess_diag += &self.data.l2_reg;
            hess
        };

        // This isn't quite the jacobian because the H*beta_old term is subtracted out.
        let rhs: Array1<F> = {
            // NOTE: this isn't actually the residuals, which would be divided by
            // the standard deviation.
            let residuals = &self.data.y - &predictor;
            // NOTE: This w*X should not include the linear offset.
            let target: Array1<F> = (var_diag * linear_predictor_no_control) + &residuals;
            // let target: Array1<F> = (var_diag * linear_predictor) + &residuals; // WRONG
            let target: Array1<F> = self.data.x.t().dot(&target);
            target
        };

        let mut next_guess: Array1<F> = match neg_hessian.solveh_into(rhs) {
            Ok(solution) => solution,
            Err(err) => return Some(Err(err.into())),
        };

        // NOTE: might be optimizable by not checking the likelihood until step
        // = next_guess - &self.guess stops decreasing.
        // Ideally we could only check the step difference but that might not be
        // as stable. Some parameters might be at different scales.
        let mut like = M::quasi_log_likelihood(&self.data, &next_guess);
        // This should be positive for an improved guess
        let mut rel = (like - self.last_like) / (F::epsilon() + like.abs());
        // Terminate if the difference is close to zero
        if rel.abs() <= self.tolerance {
            return None;
        }

        // apply step halving if rel < 0, which means the likelihood has decreased.
        // Don't terminate if rel gets back to within tolerance as a result of this.
        const MAX_STEP_HALVES: usize = 6;
        let mut step_halves = 0;
        let half: F = F::from(0.5).unwrap();
        let mut step_multiplier = half;
        while rel < -self.tolerance && step_halves < MAX_STEP_HALVES {
            let next_guess_sh = next_guess.map(|&x| x * (step_multiplier))
                + &self.guess.map(|&x| x * (F::one() - step_multiplier));
            like = M::quasi_log_likelihood(&self.data, &next_guess);
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
