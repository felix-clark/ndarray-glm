//! functions for solving logistic regression

use crate::error::RegressionError;
use crate::fit::Fit;
use crate::glm::Glm;
use crate::utility::one_pad;
use approx::AbsDiffEq;
use ndarray::{Array1, Array2, Zip};
use ndarray_linalg::{lapack::Lapack, SolveH};
use num_traits::float::Float;

// right now implement LL for logistic, but this should be moved to a trait.
fn log_likelihood<F: 'static + Float>(
    data_y: &Array1<bool>,
    data_x: &Array2<F>,
    regressors: &Array1<F>,
) -> F {
    // this assertion should be a result, or these references should be stored in Fit so they can be checked ahead of time.
    assert_eq!(
        data_y.len(),
        data_x.nrows(),
        "must have same number of data points in X and Y"
    );
    assert_eq!(
        data_x.ncols(),
        regressors.len(),
        "must have same number of explanatory variables as regressors"
    );
    // convert y data to floats, although this may not be needed in the future if we generalize to floats
    let data_y: Array1<F> = data_y.map(|&y| if y { F::one() } else { F::zero() });
    let linear_predictor: Array1<F> = data_x.dot(regressors);
    // initialize the log likelihood terms
    let mut log_like_terms: Array1<F> = Array1::zeros(data_y.len());
    Zip::from(&mut log_like_terms)
        .and(&data_y)
        .and(&linear_predictor)
        .apply(|l, &y, &wx| {
            // Both of these expressions are mathematically identical.
            // The distinction is made to avoid under/overflow.
            let (yt, xt) = if wx < F::zero() {
                (y, wx)
            } else {
                (F::one() - y, -wx)
            };
            *l = yt * xt - xt.exp().ln_1p()
        });
    log_like_terms.sum()
}

impl<F> Fit<F>
where
    F: 'static + Float,
{
    /// return the signed Z-score for each regression parameter, which should
    /// follow the Chi distribution under the null hypothesis.
    pub fn z_scores(&self, data_y: &Array1<bool>, data_x: &Array2<F>) -> Array1<F> {
        let data_x = one_pad(data_x);
        let model_like = log_likelihood(data_y, &data_x, &self.result);
        // -2 likelihood deviation is asymptotically chi^2 with ndf degrees of freedom.
        let mut chi_sqs: Array1<F> = Array1::zeros(self.result.len());
        // TODO (style): move away from explicit indexing
        for i_like in 0..self.result.len() {
            let mut adjusted = self.result.clone();
            adjusted[i_like] = F::zero();
            let null_like = log_likelihood(data_y, &data_x, &adjusted);
            let chi_sq = F::from(2.).unwrap() * (model_like - null_like);
            assert!(
                chi_sq >= F::zero(),
                "negative chi-sq. may not be an error if small."
            );
            chi_sqs[i_like] = chi_sq;
        }
        let signs = self.result.mapv(F::signum);
        let chis = chi_sqs.mapv_into(F::sqrt);
        // return the Z-scores
        signs * chis
    }
}

/// retrieve next step in IRLS (specialized for logistic regression).
/// The data vector is assumed to already be padded with ones if necessary.
/// TODO: add offset terms to account for fixed effects / controlled covariates. these affect the calculation of mu, the current expectation.
/// TODO: generalize to other models using trait system
fn next_guess<M, F>(
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

/// returns object holding fit result
pub fn regression<F>(data_y: &Array1<bool>, data_x: &Array2<F>) -> Result<Fit<F>, RegressionError>
where
    F: 'static + Float + Lapack,
    Array1<F>: AbsDiffEq,
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
    // This is only an improvement when the x points are centered around zero.
    // Perhaps that aspect can be worked around.
    // last[0] = f32::ln(data_y.iter().filter(|&&b| b).count() as f32 / data_y.iter().filter(|&&b| !b).count() as f32);
    let mut next: Array1<F> = next_guess::<Logistic, F>(&data_y, &data_x, &last)?;
    // let mut delta: Array1<F> = &next - &last;
    let mut n_iter = 0;
    // TODO: more sophisticated termination conditions.
    // This one could easily loop forever.
    // MAYBE: if delta is similar to negative last delta?
    // let tolerance: F = F::from(2.).unwrap() * F::epsilon();
    while next.abs_diff_ne(&last, Array1::<F>::default_epsilon()) {
        // dbg!(&last);
        // dbg!(&next);
        last = next.clone();
        next = next_guess::<Logistic, F>(&data_y, &data_x, &next)?;
        n_iter += 1;
    }
    // ndf is guaranteed to be > 0 because of the underconstrained check
    let ndf = n_data - next.len();
    Ok(Fit {
        result: next,
        ndf,
        n_iter,
    })
}

/// trait-based implementation to work towards generalization
struct Logistic;

impl Glm for Logistic {
    type Domain = bool;

    // the logit function
    fn link<F: Float>(y: F) -> F {
        F::ln(y / (F::one() - y))
    }

    // inverse link function, the expit function
    fn mean<F: Float>(lin_pred: F) -> F {
        F::one() / (F::one() + (-lin_pred).exp())
    }

    // var = mu*(1-mu)
    fn variance<F: Float>(mean: F) -> F {
        mean * (F::one() - mean)
    }
}
