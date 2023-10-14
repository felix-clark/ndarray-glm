//! define the error enum for the result of regressions

use ndarray_linalg::error::LinalgError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum RegressionError {
    #[error("Inconsistent input: {0}")]
    BadInput(String),
    #[error("Invalid response data: {0}")]
    InvalidY(String),
    #[error("Model build error: {0}")]
    BuildError(String),
    #[error("Linear algebra")]
    LinalgError {
        #[from]
        source: LinalgError,
    },
    #[error("Underconstrained data")]
    Underconstrained,
    #[error("Colinear data (X^T * X is not invertible)")]
    ColinearData,
    #[error("Maximum iterations ({0}) reached")]
    MaxIter(usize),
}

pub type RegressionResult<T> = Result<T, RegressionError>;
