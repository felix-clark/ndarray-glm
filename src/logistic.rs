//! functions for solving logistic regression

use crate::utility::one_pad;
use ndarray::{Array1, Array2};
use ndarray_linalg::SolveH;
use approx::AbsDiffEq;

/// retrieve next step in IRLS (specialized for logistic regression).
/// The data vector is assumed to already be padded with ones if necessary.
/// TODO: add offset terms to account for fixed effects / controlled covariates. these affect the calculation of mu, the current expectation.
/// TODO: generalize to other models using trait system
fn next_guess(data_y: &Array1<bool>, data_x: &Array2<f32>, previous: &Array1<f32>) -> Array1<f32> {
    assert_eq!(data_y.len(), data_x.nrows(), "must have same number of data points in X and Y");
    assert_eq!(data_x.ncols(), previous.len(), "must have same number of parameters in X and solution");
    // the linear predictor given the model
    let linear_predictor: Array1<f32> = data_x.dot(previous);
    // The probability of each observation being true given the current guess.
    // TODO: The offset should be added to the linear predictor in here, when implemented.
    let offset = 0.;
    let predictor: Array1<f32> = (&linear_predictor + offset).mapv(|wx| 1. / (1. + f32::exp(-wx)));
    // the diagonal covariance matrix given the model
    let variance: Array2<f32> = Array2::from_diag(&predictor.mapv(|mu| mu * (1. - mu)));
    let solve_matrix: Array2<f32> = data_x.t().dot(&variance).dot(data_x);
    let target: Array1<f32> = variance.dot(&linear_predictor) + &data_y.map(|&b| if b {1.0} else {0.0}) - &predictor;
    let target: Array1<f32> = data_x.t().dot(&target);
    solve_matrix.solveh_into(target).expect("underconstrained update")
}

/// TODO: return result
pub fn regression(data_y: &Array1<bool>, data_x: &Array2<f32>) -> Array1<f32> {
    let n_data = data_y.len();
    assert_eq!(
        n_data,
        data_x.nrows(),
        "y and x data must have same number of points"
    );
    // TODO: check that the result is overdetermined and return an error if not
    // prepend the x data with a constant 1 column
    let data_x = one_pad(data_x);

    // TODO: determine first element based on fraction of cases in sample
    let mut last = Array1::<f32>::zeros(data_x.ncols());
    let mut next = next_guess(&data_y, &data_x, &last);
    // TODO: more sophisticated termination conditions
    while ! next.abs_diff_eq(&last, 2.0 * std::f32::EPSILON) {
        dbg!(&last);
        dbg!(&next);
        last = next.clone();
        next = next_guess(&data_y, &data_x, &next);
    }
    // TODO: look at next_guess.abs_diff_eq(last_guess) to check termination condition?
    next
}
