//! trait defining a generalized linear model and providing common functionality
//! Models are fit such that E[Y] = g^-1(X*B) where g is the link function.

use crate::{error::RegressionError, fit::Fit, model::Model};
use ndarray::{Array1, Array2};
use ndarray_linalg::{lapack::Lapack, SolveH};
use num_traits::Float;
use std::marker::PhantomData;

// Does F need 'static + Float?
pub trait Glm<F: Float> {
    // the domain of the model
    // i.e. integer for Poisson, float for Linear, bool for logistic
    // TODO: perhaps create a custom Domain type or trait to deal with constraints
    // we typically work with floats as EVs, though.
    // A (private?) function that maps a general domain to the floating point
    // type could work as well.
    type Domain;

    /// Converts the domain to a floating-point value for IRLS
    fn y_float(y: Self::Domain) -> F;

    // TODO: a function to check if a Y-value is valid

    /// the link function
    // fn link<F: 'static + Float>(y: Self::Domain) -> F;
    fn link(y: F) -> F;

    // TODO: return both mean and variance as function of eta at once and avoid FPE

    /// inverse link function which maps the linear predictors to the expected value of the prediction.
    fn mean(x: F) -> F;

    /// The variance as a function of the mean. This should be related to the
    /// Laplacian of the log-partition function, or in other words, the
    /// derivative of the inverse link function mu = g^{-1}(eta).
    fn variance(mean: F) -> F;

    /// Returns the log-likelihood if it is well-defined. If not (like in
    /// unweighted OLS) returns an objective function to be maximized.
    fn quasi_log_likelihood(data: &Model<Self, F>, regressors: &Array1<F>) -> F
    where
        Self: Sized;

    /// Do the regression and return a result. Returns object holding fit result.
    fn regression(data: &Model<Self, F>) -> Result<Fit<Self, F>, RegressionError>
    where
        F: 'static + Float + Lapack,
        Self: Sized,
    {
        let n_data = data.y.len();

        // TODO: determine first element based on fraction of cases in sample
        // This is only a possible improvement when the x points are centered
        // around zero, and may introduce more complications than it's worth.
        // For logistic regression, beta = 0 is typically reasonable.
        let initial = Array1::<F>::zeros(data.x.ncols());
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

/// A subtrait for GLMs that have an unambiguous likelihood function.
// Not all regression types have a well-defined likelihood. E.g. logistic
// (binomial) and Poisson do; linear (normal) and negative binomial do not due
// to the extra parameter.
pub trait Likelihood<M, F>: Glm<F>
where
    M: Glm<F>,
    F: Float,
{
    /// logarithm of the likelihood given the data and fit parameters
    fn log_likelihood(data: &Model<M, F>, regressors: &Array1<F>) -> F;
}

/// Iterate over updates via iteratively re-weighted least-squares until
/// reaching a specified tolerance.
struct Irls<'a, M, F>
where
    M: Glm<F>,
    F: 'static + Float + Lapack,
    Array2<F>: SolveH<F>,
{
    data: &'a Model<M, F>,
    guess: Array1<F>,
    // model: PhantomData<M>,
    max_iter: usize,
    pub n_iter: usize,
    tolerance: F,
    last_like: F,
}

impl<'a, M, F> Irls<'a, M, F>
where
    M: Glm<F>,
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
    M: Glm<F>,
    F: 'static + Float + Lapack,
    Array2<F>: SolveH<F>,
{
    type Item = Result<Array1<F>, RegressionError>;

    fn next(&mut self) -> Option<Self::Item> {
        // The linear predictor without control terms
        let linear_predictor_no_control: Array1<F> = self.data.x.dot(&self.guess);
        // the linear predictor given the model, including offsets if present
        let linear_predictor = match &self.data.linear_offset {
            Some(off) => &linear_predictor_no_control + &off,
            None => linear_predictor_no_control.clone(),
        };
        // The data.linear_predictor() function is not used above because we will use
        // both versions, with and without the linear offset, and we don't want
        // to repeat the matrix multiplication.

        // The prediction of y given the current model.
        let predictor: Array1<F> = linear_predictor.mapv(M::mean);

        // The variances predicted by the model. This should have weights with
        // it (inversely?) and must be non-zero.
        // TODO: implement a mean_and_var function to return both the mean and
        // variance while avoiding over/underflow.
        let var_diag: Array1<F> = predictor.mapv(|mu| M::variance(mu) + F::epsilon());
        // This can be a full covariance with weights
        // the diagonal covariance matrix given the model
        // let variance: Array2<F> = Array2::from_diag(&predictor.mapv(M::variance));
        // positive definite
        // let hessian: Array2<F> = &self.data.x.t().dot(&variance).dot(&self.data.x);

        // X weighted by the model variance for each observation
        let mut hessian: Array2<F> = (&self.data.x.t() * &var_diag).dot(&self.data.x);

        // If L2 regularization is set, add lambda * I to the Hessian.
        if self.data.l2_reg != F::zero() {
            hessian += &Array2::from_diag(&Array1::from_elem(hessian.nrows(), self.data.l2_reg));
        }

        // NOTE: this isn't actually the residuals, which would be divided by
        // the standard deviation.
        let residuals = &self.data.y - &predictor;
        // NOTE: This w*X should not include the linear offset.
        let target: Array1<F> = (var_diag * linear_predictor_no_control) + &residuals;
        // let target: Array1<F> = (var_diag * linear_predictor) + &residuals; // WRONG
        let target: Array1<F> = self.data.x.t().dot(&target);

        let mut next_guess: Array1<F> = match hessian.solveh_into(target) {
            Ok(solution) => solution,
            Err(err) => return Some(Err(err.into())),
        };

        // NOTE: might be optimizable by not checking the likelihood until step
        // = next_guess - &self.guess stops decreasing.
        // Ideally we could only check the step difference but that might not be
        // as stable. Some parameters might be at different scales.
        let mut like = M::quasi_log_likelihood(&self.data, &next_guess);
        // This should be positive for an improvement
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
