//! wrapper holding data and other configuration settings.
//! The X-data is 1-padded as this is used for all GLMs.

// TODO: check for near-exact co-linearity in X data and warn/fail.

use crate::{
    error::{RegressionError, RegressionResult},
    utility::one_pad,
};
use ndarray::{Array1, Array2};
use ndarray_linalg::{types::Scalar, DeterminantH};
use num_traits::Float;

/// Holds the data and configuration settings for a regression
pub struct DataConfig<F>
where
    F: 'static + Float,
{
    // the observation data by event
    pub y: Array1<F>,
    // the regressor data with events in rows and instances in columns
    pub x: Array2<F>,
    // offset in the linear predictor for each data point
    pub linear_offset: Option<Array1<F>>,
    // the maximum number of iterations to try
    pub max_iter: Option<usize>,
}

// TODO: add function to get linear predictor with offsets?

pub struct DataConfigBuilder<F>
where
    F: 'static + Float,
{
    // fields passed to DataConfig
    data_y: Array1<F>,
    data_x: Array2<F>,
    // offset in the linear predictor for each data point
    linear_offset: Option<Array1<F>>,
    max_iter: Option<usize>,
    // fields unique to the builder
    // tolerance for determinant check
    det_tol: F,
}

/// A builder to generate a DataConfig object
impl<F> DataConfigBuilder<F>
where
    F: 'static + Float,
{
    pub fn new(data_y: Array1<F>, data_x: Array2<F>) -> Self {
        // the number of predictors
        let n_pred = data_x.ncols() + 1;
        Self {
            data_y,
            data_x,
            linear_offset: None,
            max_iter: None,
            det_tol: Self::default_epsilon(n_pred),
        }
    }

    /// Default tolerance for colinearity checking.
    /// Uses the square root of the number of data points times machine epsilon.
    /// This may not be particularly well-justified and may be too lenient.
    fn default_epsilon(n_data: usize) -> F {
        // NOTE: should this scaling factor be capped?
        let sqrt_n: F = F::from(n_data).unwrap().sqrt();
        sqrt_n * F::epsilon()
    }

    /// Represents an offset added to the linear predictor for each data point.
    /// This can be used to control for fixed effects or in multi-level models.
    pub fn linear_offset(mut self, linear_offset: Array1<F>) -> Self {
        self.linear_offset = Some(linear_offset);
        panic!("Linear offsets are not implemented everywhere (log_likelihood())");
        // self
    }

    /// Use a maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = Some(max_iter);
        self
    }

    /// Set the tolerance for the co-linearity check.
    pub fn colinearity_tolerance(mut self, tol: F) -> Self {
        self.det_tol = tol;
        self
    }

    pub fn build(self) -> RegressionResult<DataConfig<F>>
    where
        F: 'static + Float,
        Array2<F>: DeterminantH,
        <<Array2<F> as DeterminantH>::Elem as Scalar>::Real: std::convert::Into<F>,
    {
        let n_data = self.data_y.len();
        if n_data != self.data_x.nrows() {
            return Err(RegressionError::BadInput(
                "y and x data must have same number of points".to_string(),
            ));
        }
        if n_data < self.data_x.ncols() + 1 {
            // The regression can find a solution if n_data == ncols + 1, but there will be no estimate for the uncertainty.
            return Err(RegressionError::Underconstrained);
        }
        // If they are provided, check that the offsets have the correct number of entries
        if let Some(lin_off) = &self.linear_offset {
            if n_data != lin_off.len() {
                return Err(RegressionError::BadInput(
                    "Offsets must have same dimension as observations".to_string(),
                ));
            }
        }

        let xtx: Array2<F> = self.data_x.t().dot(&self.data_x);
        let det: <<Array2<F> as DeterminantH>::Elem as Scalar>::Real = xtx.deth()?;
        let det: F = det.into();
        if det.abs() < self.det_tol {
            return Err(RegressionError::ColinearData);
        }

        Ok(DataConfig {
            y: self.data_y,
            x: one_pad(&self.data_x),
            linear_offset: self.linear_offset,
            max_iter: self.max_iter,
        })
    }
}
