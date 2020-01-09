//! trait defining a generalized linear model and providing common functionality
//! Models are fit such that E[Y] = g^-1(X*B) where g is the link function.

use crate::{error::RegressionError, fit::Fit, utility::one_pad};
use approx::AbsDiffEq;
use ndarray::{Array1, Array2};
use ndarray_linalg::{lapack::Lapack, SolveH};
use num_traits::Float;
use std::marker::PhantomData;

pub trait Glm {
    // the domain of the model
    // i.e. integer for Poisson, float for Linear, bool for logistic
    // TODO: perhaps create a custom Domain type or trait to deal with constraints
    // we typically work with floats as EVs, though.
    type Domain;

    /// a function to check if a Y-value is value

    /// the link function
    // fn link<F: 'static + Float>(y: Self::Domain) -> F;
    fn link<F: Float>(y: F) -> F;

    /// inverse link function which maps the linear predictors to the expected value of the prediction.
    fn mean<F: Float>(x: F) -> F;

    /// the variance as a function of the mean
    fn variance<F: Float>(mean: F) -> F;

    /// logarithm of the likelihood given the data and fit parameters
    fn log_likelihood<F: 'static + Float>(
        data_y: &Array1<Self::Domain>,
        data_x: &Array2<F>,
        regressors: &Array1<F>,
    ) -> F;

    /// returns object holding fit result
    // TODO: make more robust, for instance using step-halving if issues are detected.
    // Non-standard link functions could still cause issues. See for instance
    // https://journal.r-project.org/archive/2011-2/RJournal_2011-2_Marschner.pdf

    fn regression<F>(
        data_y: &Array1<bool>,
        data_x: &Array2<F>,
    ) -> Result<Fit<Self, F>, RegressionError>
    where
        F: 'static + Float + Lapack,
        Array1<F>: AbsDiffEq,
        Self: Sized,
    {
        let n_data = data_y.len();
        if n_data != data_x.nrows() {
            return Err(RegressionError::BadInput(
                "y and x data must have same number of points".to_string(),
            ));
        }
        if n_data < data_x.ncols() + 1 {
            // The regression can find a solution if n_data == ncols + 1, but there will be no estimate for the uncertainty.
            return Err(RegressionError::Underconstrained);
        }
        // prepend the x data with a constant 1 column
        let data_x = one_pad(data_x);

        let mut last = Array1::<F>::zeros(data_x.ncols());
        // TODO: determine first element based on fraction of cases in sample
        // This is only a possible improvement when the x points are centered
        // around zero, and may introduce more complications than it's worth.
        // For logistic regression, beta = 0 is typically reasonable.
        let mut next: Array1<F> = next_irls::<Self, F>(&data_y, &data_x, &last)?;
        // store the maximum change of each component.
        // let mut max_delta = F::infinity();
        let mut delta: Array1<F> = &next - &last;
        let mut n_iter = 0;
        // TODO: more sophisticated termination conditions.
        // This one could easily loop forever.
        // MAYBE: if delta is similar to negative last delta?
        // let tolerance: F = F::from(2.).unwrap() * F::epsilon();

        while next.abs_diff_ne(&last, Array1::<F>::default_epsilon()) {
            // dbg!(&last);
            // dbg!(&next);
            last = next.clone();
            next = next_irls::<Self, F>(&data_y, &data_x, &next)?;
            n_iter += 1;
        }
        // ndf is guaranteed to be > 0 because of the underconstrained check
        let ndf = n_data - next.len();
        Ok(Fit {
            model: PhantomData::<Self>,
            result: next,
            ndf,
            n_iter,
        })
    }
}

/// Private function to retrieve next step in IRLS (specialized for logistic
/// regression).
/// The data vector is assumed to already be padded with ones if necessary.
/// TODO: add offset terms to account for fixed effects / controlled covariates.
/// these affect the calculation of mu, the current expectation.
fn next_irls<M, F>(
    data_y: &Array1<bool>,
    data_x: &Array2<F>,
    previous: &Array1<F>,
) -> Result<Array1<F>, RegressionError>
where
    M: Glm,
    F: 'static + Float + Lapack,
    Array2<F>: SolveH<F>,
{
    assert_eq!(
        data_y.len(),
        data_x.nrows(),
        "must have same number of data points in X and Y"
    );
    assert_eq!(
        data_x.ncols(),
        previous.len(),
        "must have same number of parameters in X and solution"
    );
    // the linear predictor given the model
    let linear_predictor: Array1<F> = data_x.dot(previous);
    // The probability of each observation being true given the current guess.
    // TODO: The offset should be added to the linear predictor in here, when implemented.
    let offset = Array1::<F>::zeros(linear_predictor.len());
    let predictor: Array1<F> = (&linear_predictor + &offset).mapv_into(M::mean);
    // the diagonal covariance matrix given the model
    let variance: Array2<F> = Array2::from_diag(&predictor.mapv(M::variance));
    // positive definite
    let solve_matrix: Array2<F> = data_x.t().dot(&variance).dot(data_x);
    let target: Array1<F> = variance.dot(&linear_predictor)
        + &data_y.map(|&b| if b { F::one() } else { F::zero() })
        - &predictor;
    let target: Array1<F> = data_x.t().dot(&target);
    // Ok(solve_matrix.solveh_into(target)?)
    Ok(solve_matrix.solveh_into(target)?)
}
