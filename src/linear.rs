//! Functions for solving linear regression

use crate::utility::one_pad; // TODO: this should be able to be removed long term
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
pub fn regression(
    data_y: &Array1<f32>,
    data_x: &Array2<f32>,
) -> Result<Array1<f32>, RegressionError> {
    let n_data = data_y.len();
    if n_data != data_x.nrows() {
        return Err(RegressionError::BadInput(
            "y and x must have same number of data points".to_string(),
        ));
    }
    // prepend the x data with a constant 1 column
    let data_x = one_pad(data_x);
    // the vector X^T * y
    let xty: Array1<f32> = data_y.dot(&data_x);
    // the positive-definite matrix X^T * X
    let xtx: Array2<f32> = data_x.t().dot(&data_x);
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

    // specialize LL for linear
    // FIXME: it might depend on the variance, which is undetermined?
    fn log_likelihood<F: 'static + Float>(data: &DataConfig<F>, regressors: &Array1<F>) -> F {
        // TODO: this assertion should be a result, or these references should
        // be stored in Fit so they can be checked ahead of time.
        assert_eq!(
            data.x.ncols(),
            regressors.len(),
            "must have same number of explanatory variables as regressors"
        );
        unimplemented!(
            "Linear regression likelihood not determined - may depend on additional parameter?"
        );
    }
}
