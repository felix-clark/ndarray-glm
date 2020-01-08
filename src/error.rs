//! define the error enum for the result of regressions

use thiserror::Error;
use ndarray_linalg::error::LinalgError;

#[derive(Error, Debug)]
pub enum RegressionError {
    #[error("Inconsistent input: {0}")]
    BadInput(String),
    #[error("Linear algebra")]
    LinalgError {
        #[from]
        source: LinalgError,
    },
    #[error("Underconstrained data")]
    Underconstrained,
}
