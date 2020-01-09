//! struct holding the fit result of a regression
use ndarray::Array1;
use num_traits::Float;

/// the result of a successful GLM fit (logistic for now)
/// TODO: finish generalizing, take ownership of Y and X data?
#[derive(Debug)]
pub struct Fit<F>
where
    F: Float,
{
    // the parameter values that maximize the likelihood
    pub result: Array1<F>,
    // number of data points minus number of free parameters
    pub ndf: usize,
    // the number of iterations taken
    pub n_iter: usize,
}
