//! Functions for solving linear regression

use crate::{data::DataConfig, error::RegressionError, glm::Glm};
use ndarray::{
    Array1,
    Array2,
    // Zip
};
use ndarray_linalg::SolveH;
use num_traits::Float;

/// data_y is an array of the y values, data_x is an array with rows indicating the data point and columns indicating the regressor
/// Returns ordinary least squares solution of length 1 greater than the width of X
/// TODO: return a result which includes validity and uncertainty information, rather than just the solution
pub fn regression(data: &DataConfig<f32>) -> Result<Array1<f32>, RegressionError> {
    let n_data = data.y.len();
    if n_data != data.x.nrows() {
        return Err(RegressionError::BadInput(
            "y and x must have same number of data points".to_string(),
        ));
    }
    // TODO: linear offset
    // the vector X^T * y
    let xty: Array1<f32> = data.y.dot(&data.x);
    // the positive-definite matrix X^T * X
    let xtx: Array2<f32> = data.x.t().dot(&data.x);
    Ok(xtx.solveh_into(xty)?)
}

pub struct Linear;

impl Glm for Linear {
    // the link function, identity
    fn link<F: Float>(y: F) -> F {
        y
    }

    // inverse link function, identity
    fn mean<F: Float>(lin_pred: F) -> F {
        lin_pred
    }

    // variance is not a function of the mean
    fn variance<F: Float>(_mean: F) -> F {
        F::one()
    }

    // This version doesn't have the variances - either setting them to 1 or
    // 1/2pi to simplify the expression. It returns a simple sum of squares.
    fn quasi_log_likelihood<F: Float>(data: &DataConfig<F>, regressors: &Array1<F>) -> F {
        let squares: Array1<F> = (&data.y - &data.x.dot(regressors)).map(|&d| d * d);
        squares.sum()
    }
}
