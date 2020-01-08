//! functions for solving logistic regression

use crate::utility::one_pad;
use ndarray::{Array1, Array2};
use ndarray_linalg::SolveH;

/// retrieve next step in IRLS (specialized for logistic regression).
/// The data vector is assumed to already be padded with ones if necessary.
/// TODO: add offset terms to account for fixed effects / controlled covariates. these affect the calculation of mu, the current expectation.
/// TODO: generalize to other models using trait system
fn next_guess(data_y: &Array1<bool>, data_x: &Array2<f32>, previous: &Array1<f32>) -> Array1<f32> {
    let n_data = data_y.len();
    assert_eq!(n_data, data_x.nrows());
    assert_eq!(n_data, previous.len());
    // the linear predictor given the model
    let linear_predictor: Array1<f32> = data_x.dot(previous);
    // placeholder return
    previous.to_owned()
}

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

    let start = Array1::<f32>::zeros(data_x.ncols());
    let first_guess = next_guess(&data_y, &data_x, &start);

    // TODO: look at next_guess.abs_diff_eq(last_guess) to check termination condition?

    // // the vector X^T * y
    // let xty: Array1<f32> = data_y.dot(&data_x);
    // // the positive-definite matrix X^T * X
    // let xtx: Array2<f32> = data_x.t().dot(&data_x);
    // xtx.solveh_into(xty)
    //     .expect("underdetermined linear regression")

    // placeholder return
    first_guess
}
