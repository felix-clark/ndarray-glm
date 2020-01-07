//! Functions for solving linear regression

use ndarray::{Array1, Array2, Axis, stack};

/// prepend the input with a column of ones.
/// useful to describe a constant term in a regression in a general way with the data.
/// NOTE: This creates a copy, which may not be memory efficient. It can be used in such a way such that the old value is dropped.
fn one_pad(data: &Array2<f32>) -> Array2<f32> {
    // create the ones column
    let ones: Array2<f32> = Array2::ones((data.nrows(), 1));
    // This should be guaranteed to succeed, so just unwrap it.
    stack(Axis(1), &[ones.view(), data.view()]).unwrap()
}

/// data_y is an array of the y values, data_x is an array with rows indicating the data point and columns indicating the regressor
/// Returns ordinary least squares solution of length 1 greater than the width of X
/// TODO: return a result which includes validity and uncertainty information, rather than just the solution
pub fn regression(data_y: &Array1<f32>, data_x: &Array2<f32>) -> Array1<f32> {
    let n_data = data_y.len();
    assert_eq!(n_data, data_x.nrows(), "y and x data must have same number of points");
    // prepend the x data with a constant 1 column
    let data_x = one_pad(data_x);
    // let xt: Array2<f32> = data_x.t().to_owned();
    // let xty: Array1<f32> = xt.dot(&data_y);
    // x^T * y
    let xty = data_y.dot(&data_x);
    // dummy value
    Array1::from(vec![0.3, 0.4, 0.5])
}
