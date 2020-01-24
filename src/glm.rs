//! trait defining a generalized linear model and providing common functionality
//! Models are fit such that E[Y] = g^-1(X*B) where g is the link function.

use crate::{data::DataConfig, error::RegressionError, fit::Fit};
use approx::AbsDiffEq;
// use itertools::{all, Itertools};
use ndarray::{Array1, Array2};
use ndarray_linalg::{lapack::Lapack, SolveH};
use num_traits::Float;
use std::marker::PhantomData;

pub trait Glm {
    // the domain of the model
    // i.e. integer for Poisson, float for Linear, bool for logistic
    // TODO: perhaps create a custom Domain type or trait to deal with constraints
    // we typically work with floats as EVs, though.
    // A (private?) function that maps a general domain to the floating point
    // type could work as well.
    // type Domain;
    // fn y_float(y: Self::Domain) -> F: Float {
    // }

    // TODO: a function to check if a Y-value is valid

    /// the link function
    // fn link<F: 'static + Float>(y: Self::Domain) -> F;
    fn link<F: Float>(y: F) -> F;

    // TODO: return both mean and variance at once and avoid FPE

    /// inverse link function which maps the linear predictors to the expected value of the prediction.
    fn mean<F: Float>(x: F) -> F;

    /// the variance as a function of the mean
    fn variance<F: Float>(mean: F) -> F;

    /// returns object holding fit result
    // TODO: make more robust, for instance using step-halving if issues are detected.
    // Non-standard link functions could still cause issues. See for instance
    // https://journal.r-project.org/archive/2011-2/RJournal_2011-2_Marschner.pdf

    /// Returns the log-likelihood if it is well-defined. If not (like in
    /// unweighted OLS) returns an objective function to be maximized.
    fn quasi_log_likelihood<F: 'static + Float>(data: &DataConfig<F>, regressors: &Array1<F>) -> F;

    /// Do the regression and return a result
    fn regression<F>(data: &DataConfig<F>) -> Result<Fit<Self, F>, RegressionError>
    where
        F: 'static + Float + Lapack,
        Array1<F>: AbsDiffEq,
        Self: Sized,
    {
        let n_data = data.y.len();

        let initial = Array1::<F>::zeros(data.x.ncols());
        // let mut last = Array1::<F>::zeros(data.x.ncols());
        // TODO: determine first element based on fraction of cases in sample
        // This is only a possible improvement when the x points are centered
        // around zero, and may introduce more complications than it's worth.
        // For logistic regression, beta = 0 is typically reasonable.
        // let mut next: Array1<F> = next_irls::<Self, F>(&data, &last)?;
        // store the maximum change of each component.
        // let mut max_delta = F::infinity();
        // let mut delta: Array1<F> = &next - &last;
        let mut n_iter: usize = 0;

        // Step halving is applied when the size of the change is equal or larger.

        // TODO: This epsilon should be configurable.
        // let epsilon: F = F::from(8.0).unwrap() * Float::epsilon();

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
pub trait Likelihood: Glm {
    /// logarithm of the likelihood given the data and fit parameters
    fn log_likelihood<F: 'static + Float>(data: &DataConfig<F>, regressors: &Array1<F>) -> F;
}

/// Struct to iterate over updates via iteratively re-weighted least-squares until reaching a specified tolerance
struct Irls<'a, M, F>
where
    M: Glm,
    F: 'static + Float + Lapack,
    Array2<F>: SolveH<F>,
{
    data: &'a DataConfig<F>,
    guess: Array1<F>,
    model: PhantomData<M>,
    max_iter: usize,
    pub n_iter: usize,
    tolerance: F,
    last_like: F,
}

impl<'a, M, F> Irls<'a, M, F>
where
    M: Glm,
    F: 'static + Float + Lapack,
    Array2<F>: SolveH<F>,
{
    fn new(data: &'a DataConfig<F>, initial: Array1<F>) -> Self {
        let tolerance: F = F::epsilon(); // * F::from(data.y.len()).unwrap();
        Self {
            data,
            guess: initial,
            model: PhantomData,
            // TODO: builder pattern to set optional parameters like this
            max_iter: 40,
            n_iter: 0,
            // As a ratio with the variance, this epsilon could be too small.
            tolerance,
            last_like: -F::infinity(),
        }
    }
}

impl<'a, M, F> Iterator for Irls<'a, M, F>
where
    M: Glm,
    F: 'static + Float + Lapack,
    Array2<F>: SolveH<F>,
{
    type Item = Result<Array1<F>, RegressionError>;

    fn next(&mut self) -> Option<Self::Item> {
        // the linear predictor given the model
        let linear_predictor: Array1<F> = self.data.x.dot(&self.guess);
        // The prediction of y given the current model.
        let predictor: Array1<F> = if let Some(offset) = &self.data.linear_offset {
            (&linear_predictor + offset).mapv_into(M::mean)
        } else {
            linear_predictor.mapv(M::mean)
        };
        // The variances predicted by the model. This should have weights with
        // it (inversely?) and must be non-zero.
        // TODO: implement a mean_and_var function to return both the mean and
        // variance while avoiding over/underflow.
        let var_diag: Array1<F> = predictor.mapv(|mu| M::variance(mu) + F::epsilon());
        // This can be a full covariance with weights
        // the diagonal covariance matrix given the model
        // let variance: Array2<F> = Array2::from_diag(&predictor.mapv(M::variance));
        // positive definite
        // let solve_matrix: Array2<F> = data.x.t().dot(&variance).dot(&data.x);

        // X weighted by the model variant
        let solve_matrix: Array2<F> = (&self.data.x.t() * &var_diag).dot(&self.data.x);
        // TODO: quadratic regularization can be implemented by adding lambda*I to solve_matrix.
        // This will need to be implemented in the log-likelihoods as well (or at
        // least the Z-scores must be modified).
        // TODO: How would such regularization affect the termination condition?
        // Check via the change in the likelihood that includes - 0.5*lambda*||beta||^2

        // TODO: use the Self::Domain -> F function to get floating point values for y (do in initialization)
        // NOTE: this isn't actually the residuals, which would be divided by
        // the standard deviation (rather than the standard deviation).
        let residuals = &self.data.y - &predictor;
        let target: Array1<F> = (var_diag * &linear_predictor) + &residuals;
        let target: Array1<F> = self.data.x.t().dot(&target);

        let mut next_guess: Array1<F> = match solve_matrix.solveh_into(target) {
            Ok(solution) => solution,
            Err(err) => return Some(Err(err.into())),
        };

        // NOTE: might be optimizable by not checking the likelihood until step
        // = next_guess - &self.guess stops decreasing.
        // Ideally we could only check the step difference but that might not be
        // as stable. Some parameters might be at different scales.
        // TODO: add regularization term to likelihood
        let mut like = M::quasi_log_likelihood(&self.data, &next_guess);
        let mut rel = (like - self.last_like) / (F::epsilon() + like.abs());
        // Terminate
        if rel.abs() < self.tolerance {
            return None;
        }
        // apply step halving if rel < 0, which means the likelihood has decreased.
        // Don't terminate if rel gets back to within tolerance as a result of this.
        while rel < -self.tolerance {
            let half: F = F::from(0.5).unwrap();
            let next_guess_sh =
                Array1::<F>::from_elem(next_guess.len(), half) * (&next_guess + &self.guess);
            like = M::quasi_log_likelihood(&self.data, &next_guess);
            let next_rel = (like - self.last_like) / (F::epsilon() + like.abs());
            // If this halving step isn't an improvement, stop halving and let
            // the next IRLS iteration have a go.
            if next_rel <= rel {
                break;
            }
            next_guess = next_guess_sh;
            rel = next_rel;
        }

        self.guess.assign(&next_guess);
        self.last_like = like;
        self.n_iter += 1;
        if self.n_iter > self.max_iter {
            return Some(Err(RegressionError::MaxIter(self.max_iter)));
        }
        Some(Ok(next_guess))
    }
}
