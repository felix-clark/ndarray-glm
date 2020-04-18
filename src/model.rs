//! Collect data for and configure a model

use crate::{
    error::{RegressionError, RegressionResult},
    fit::Fit,
    glm::Glm,
    utility::one_pad,
};
use ndarray::{Array1, Array2};
use ndarray_linalg::lapack::Lapack;
use ndarray_linalg::{types::Scalar, DeterminantH};
use num_traits::Float;
use std::marker::PhantomData;

/// Holds the data and configuration settings for a regression
pub struct Model<M, F>
where
    // M: Glm<F>,
    M: Glm,
    F: 'static + Float,
{
    model: PhantomData<M>,
    // the observation data by event
    pub y: Array1<F>,
    // the regressor data with events in rows and instances in columns
    pub x: Array2<F>,
    // The offset in the linear predictor for each data point. This can be used
    // to fix the effect of control variables.
    pub linear_offset: Option<Array1<F>>,
    // the maximum number of iterations to try
    pub max_iter: Option<usize>,
    // L2 regularization applied to all but the intercept term.
    pub l2_reg: Array1<F>,
}

impl<M, F> Model<M, F>
where
    // M: Glm<F>,
    M: Glm,
    F: Float + Lapack,
{
    pub fn fit(&self) -> Result<Fit<M, F>, RegressionError> {
        M::regression(&self)
    }

    pub fn linear_predictor(&self, regressors: &Array1<F>) -> Array1<F> {
        let linear_predictor: Array1<F> = self.x.dot(regressors);
        // Add linear offsets to the predictors if they are set
        if let Some(lin_offset) = &self.linear_offset {
            linear_predictor + lin_offset
        } else {
            linear_predictor
        }
    }

    /// The contribution to the likelihood from the L2 term.
    pub fn l2_like_term(&self, regressors: &Array1<F>) -> F {
        -F::from(0.5).unwrap() * (&self.l2_reg * &regressors.map(|&b| b * b)).sum()
    }
}

pub struct ModelBuilder<'a, M, F>
where
    // M: Glm<F>,
    M: Glm,
    F: 'static + Float,
{
    model: PhantomData<M>,
    // fields passed to Model
    data_y: &'a Array1<M::Domain>,
    data_x: &'a Array2<F>,
    // offset in the linear predictor for each data point
    linear_offset: Option<Array1<F>>,
    max_iter: Option<usize>,
    // fields unique to the builder
    add_constant_term: bool,
    // tolerance for determinant check
    det_tol: F,
    // L2 regularization
    l2_reg: F,
}

/// A builder to generate a Model object
impl<'a, M, F> ModelBuilder<'a, M, F>
where
    // M: Glm<F>,
    M: Glm,
    F: 'static + Float,
    // <M as Glm<F>>::Domain: Copy,
    <M as Glm>::Domain: Copy,
{
    pub fn new(data_y: &'a Array1<M::Domain>, data_x: &'a Array2<F>) -> Self {
        // the number of predictors
        let n_pred = data_x.ncols() + 1;
        Self {
            model: PhantomData,
            data_y,
            data_x,
            linear_offset: None,
            max_iter: None,
            add_constant_term: true,
            det_tol: Self::default_epsilon(n_pred),
            l2_reg: F::zero(),
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
        self
    }

    /// Use a maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = Some(max_iter);
        self
    }

    /// Use to set a L2 regularization parameter
    pub fn l2_reg(mut self, l2: F) -> Self {
        self.l2_reg = l2;
        self
    }

    /// Do not add a constant term to the design matrix
    pub fn no_constant(mut self) -> Self {
        self.add_constant_term = false;
        self
    }

    /// Set the tolerance for the co-linearity check.
    // TODO: perhaps this should be optional
    pub fn colinearity_tolerance(mut self, tol: F) -> Self {
        self.det_tol = tol;
        self
    }

    pub fn build(self) -> RegressionResult<Model<M, F>>
    where
        // M: Glm<F>,
        M: Glm,
        F: Float,
        Array2<F>: DeterminantH,
        <<Array2<F> as DeterminantH>::Elem as Scalar>::Real: std::convert::Into<F>,
    {
        let n_data = self.data_y.len();
        if n_data != self.data_x.nrows() {
            return Err(RegressionError::BadInput(
                "y and x data must have same number of points".to_string(),
            ));
        }
        // If they are provided, check that the offsets have the correct number of entries
        if let Some(lin_off) = &self.linear_offset {
            if n_data != lin_off.len() {
                return Err(RegressionError::BadInput(
                    "Offsets must have same dimension as observations".to_string(),
                ));
            }
        }

        // Check for co-linearity by ensuring that the determinant of X^T * X is non-zero.
        let xtx: Array2<F> = self.data_x.t().dot(self.data_x);
        let det: <<Array2<F> as DeterminantH>::Elem as Scalar>::Real = xtx.deth()?;
        let det: F = det.into();
        if det.abs() < self.det_tol {
            // Perhaps this error should be left to a linear algebra failure,
            // but in that case an error message should be informative. Maybe
            // only do the check in that case.
            return Err(RegressionError::ColinearData);
        }

        // add constant term to X data
        let data_x = if self.add_constant_term {
            one_pad(&self.data_x)
        } else {
            self.data_x.clone()
        };
        // Check if the data is under-constrained
        if n_data < data_x.ncols() {
            // The regression can find a solution if n_data == ncols, but
            // there will be no estimate for the uncertainty.
            return Err(RegressionError::Underconstrained);
        }

        // convert to floating-point
        let data_y: Array1<F> = self.data_y.map(|&y| M::y_float(y));

        // make the vector of L2 coefficients
        let l2_diag: Array1<F> = {
            let mut l2_diag: Array1<F> = Array1::<F>::from_elem(data_x.ncols(), self.l2_reg);
            // if an intercept term is included it should not be subject to
            // regularization.
            if self.add_constant_term {
                l2_diag[0] = F::zero();
            }
            l2_diag
        };

        Ok(Model {
            model: PhantomData,
            y: data_y,
            x: data_x,
            linear_offset: self.linear_offset,
            max_iter: self.max_iter,
            l2_reg: l2_diag,
        })
    }
}
