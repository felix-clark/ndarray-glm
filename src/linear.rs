//! Functions for solving linear regression

use crate::utility::one_pad;
use ndarray::{Array1, Array2};
use ndarray_linalg::SolveH;

/// data_y is an array of the y values, data_x is an array with rows indicating the data point and columns indicating the regressor
/// Returns ordinary least squares solution of length 1 greater than the width of X
/// TODO: return a result which includes validity and uncertainty information, rather than just the solution
pub fn regression(data_y: &Array1<f32>, data_x: &Array2<f32>) -> Array1<f32> {
    let n_data = data_y.len();
    assert_eq!(
        n_data,
        data_x.nrows(),
        "y and x data must have same number of points"
    );
    // TODO: check that the result is overdetermined and return an error if not
    // prepend the x data with a constant 1 column
    let data_x = one_pad(data_x);
    // the vector X^T * y
    let xty: Array1<f32> = data_y.dot(&data_x);
    // the positive-definite matrix X^T * X
    let xtx: Array2<f32> = data_x.t().dot(&data_x);
    xtx.solveh_into(xty)
        .expect("underdetermined linear regression")
}
