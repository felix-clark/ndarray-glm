//! functions for solving logistic regression

use crate::error::RegressionError;
use crate::utility::one_pad;
use approx::AbsDiffEq;
use ndarray::{Array1, Array2, Zip};
use ndarray_linalg::SolveH;

/// the result of a successful GLM fit (logistic for now)
pub struct Fit {
    // the parameter values that maximize the likelihood
    pub result: Array1<f32>,
    // number of data points minus number of free parameters
    pub ndf: usize,
    // the number of iterations taken
    pub n_iter: usize,
}

// right now implement LL for logistic, but this should be moved to a trait.
// uses self.result in the likelihood. could modify to keep references to the data too.
fn log_likelihood(data_y: &Array1<bool>, data_x: &Array2<f32>, regressors: &Array1<f32>) -> f32 {
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
    let data_y: Array1<f32> = data_y.map(|&y| if y { 1.0 } else { 0.0 });
    let linear_predictor: Array1<f32> = data_x.dot(regressors);
    // initialize the log likelihood terms
    let mut log_like_terms: Array1<f32> = Array1::zeros(data_y.len());
    Zip::from(&mut log_like_terms)
        .and(&data_y)
        .and(&linear_predictor)
        .apply(|l, &y, &wx| {
            // Both of these expressions are mathematically identical.
            // The distinction is made to avoid under/overflow.
            *l = if wx < 0. {
                y * wx - wx.exp().ln_1p()
            } else {
                -(1. - y) * wx - (-wx).exp().ln_1p()
            };
        });
    log_like_terms.sum()
}

impl Fit {
    // likelihood ratio test for each element of the result
    // TODO: the Z-score can probably be calculated easily using the exact 2nd derivative.
    pub fn significance(&self) {
        todo!("likelihood ratio test for each element")
    }
}

/// retrieve next step in IRLS (specialized for logistic regression).
/// The data vector is assumed to already be padded with ones if necessary.
/// TODO: add offset terms to account for fixed effects / controlled covariates. these affect the calculation of mu, the current expectation.
/// TODO: generalize to other models using trait system
fn next_guess(
    data_y: &Array1<bool>,
    data_x: &Array2<f32>,
    previous: &Array1<f32>,
) -> Result<Array1<f32>, RegressionError> {
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
    let linear_predictor: Array1<f32> = data_x.dot(previous);
    // The probability of each observation being true given the current guess.
    // TODO: The offset should be added to the linear predictor in here, when implemented.
    let offset = 0.;
    let predictor: Array1<f32> =
        (&linear_predictor + offset).mapv_into(|wx| 1. / (1. + f32::exp(-wx)));
    // the diagonal covariance matrix given the model
    let variance: Array2<f32> = Array2::from_diag(&predictor.mapv(|mu| mu * (1. - mu)));
    // positive definite
    let solve_matrix: Array2<f32> = data_x.t().dot(&variance).dot(data_x);
    let target: Array1<f32> =
        variance.dot(&linear_predictor) + &data_y.map(|&b| if b { 1.0 } else { 0.0 }) - &predictor;
    let target: Array1<f32> = data_x.t().dot(&target);
    Ok(solve_matrix.solveh_into(target)?)
}

/// returns object holding fit result
pub fn regression(data_y: &Array1<bool>, data_x: &Array2<f32>) -> Result<Fit, RegressionError> {
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

    let mut last = Array1::<f32>::zeros(data_x.ncols());
    // TODO: determine first element based on fraction of cases in sample
    // This is only an improvement when the x points are centered around zero.
    // Perhaps this can be fixed up.
    // last[0] = f32::ln(data_y.iter().filter(|&&b| b).count() as f32 / data_y.iter().filter(|&&b| !b).count() as f32);
    let mut next = next_guess(&data_y, &data_x, &last)?;
    let mut n_iter = 0;
    // TODO: more sophisticated termination conditions.
    // This one could easily loop forever.
    // MAYBE: if delta is similar to negative last delta?
    while !next.abs_diff_eq(&last, 2.0 * std::f32::EPSILON) {
        // dbg!(&last);
        // dbg!(&next);
        last = next.clone();
        next = next_guess(&data_y, &data_x, &next)?;
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
