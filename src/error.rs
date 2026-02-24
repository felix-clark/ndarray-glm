//! define the error enum for the result of regressions

use ndarray_linalg::error::LinalgError;
use thiserror::Error;

use crate::{irls::IrlsStep, num::Float};

#[derive(Error, Debug)]
pub enum RegressionError<F: Float> {
    #[error("Inconsistent input: {0}")]
    BadInput(String),
    #[error("Invalid response data: {0}")]
    InvalidY(String),
    #[error("Model build error: {0}")]
    BuildError(String),
    #[error("Linear algebra error. Consider adding L2 regularization.")]
    LinalgError {
        #[from]
        source: LinalgError,
    },
    #[error("Maximum iterations ({n_iter}) reached")]
    MaxIter {
        n_iter: usize,
        history: Vec<IrlsStep<F>>,
    },
}

pub type RegressionResult<T, F> = Result<T, RegressionError<F>>;
