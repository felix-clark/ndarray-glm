//! Collect data for and configure a model

use crate::{
    error::{RegressionError, RegressionResult},
    fit::Fit,
    glm::{Glm, Response},
    math::is_rank_deficient,
    num::Float,
    regularization::{Null, Regularize, Ridge},
    utility::one_pad,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::marker::PhantomData;

/// Holds the data and configuration settings for a regression.
pub struct Model<M, F>
where
    M: Glm,
    F: Float,
{
    pub model: PhantomData<M>,
    /// the observation of response data by event
    pub y: Array1<F>,
    /// the design matrix with events in rows and instances in columns
    pub x: Array2<F>,
    /// The offset in the linear predictor for each data point. This can be used
    /// to fix the effect of control variables.
    // TODO: Consider making this an option of a reference.
    pub linear_offset: Option<Array1<F>>,
    /// the maximum number of iterations to try before failure.
    pub max_iter: Option<usize>,
    /// Regularization method
    pub reg: Box<dyn Regularize<F>>,
    /// Whether the intercept term is used (commonly true)
    pub use_intercept: bool,
}

impl<M, F> Model<M, F>
where
    M: Glm,
    F: Float,
{
    /// Perform the regression and return a fit object holding the results.
    pub fn fit(&self) -> RegressionResult<Fit<M, F>> {
        M::regression(&self)
    }

    /// Returns the linear predictors, i.e. the design matrix multiplied by the
    /// regression parameters. Each entry in the resulting array is the linear
    /// predictor for a given observation. If linear offsets for each
    /// observation are provided, these are added to the linear predictors
    pub fn linear_predictor(&self, regressors: &Array1<F>) -> Array1<F> {
        let linear_predictor: Array1<F> = self.x.dot(regressors);
        // Add linear offsets to the predictors if they are set
        if let Some(lin_offset) = &self.linear_offset {
            linear_predictor + lin_offset
        } else {
            linear_predictor
        }
    }
}

/// Provides an interface to create the full model option struct with convenient
/// type inference.
pub struct ModelBuilder<M: Glm> {
    _model: PhantomData<M>,
}

impl<M: Glm> ModelBuilder<M> {
    /// Borrow the Y and X data where each row in the arrays is a new
    /// observation, and create the full model builder with the data to allow
    /// for adjusting additional options.
    pub fn data<'a, Y, F>(
        data_y: ArrayView1<'a, Y>,
        data_x: ArrayView2<'a, F>,
    ) -> ModelBuilderData<'a, M, Y, F>
    where
        Y: Response<M>,
        F: Float,
    {
        ModelBuilderData {
            model: PhantomData,
            data_y,
            data_x,
            linear_offset: None,
            max_iter: None,
            use_intercept_term: true,
            colin_tol: F::epsilon(),
            l2_reg: F::zero(),
        }
    }
}

/// Holds the data and all the specifications for the model and provides
/// functions to adjust the settings.
pub struct ModelBuilderData<'a, M, Y, F>
where
    M: Glm,
    Y: Response<M>,
    F: 'static + Float,
{
    model: PhantomData<M>,
    /// Observed response variable data where each entry is a new observation.
    data_y: ArrayView1<'a, Y>,
    /// Design matrix of observed covariate data where each row is a new
    /// observation and each column represents a different dependent variable.
    data_x: ArrayView2<'a, F>,
    /// The offset in the linear predictor for each data point. This can be used
    /// to incorporate control terms.
    // TODO: consider making this a reference/ArrayView. Y and X are effectively
    // cloned so perhaps this isn't a big deal.
    linear_offset: Option<Array1<F>>,
    /// The maximum number of iterations before the regression reports failure.
    max_iter: Option<usize>,
    /// Whether to use an intercept term. Defaults to `true`.
    use_intercept_term: bool,
    /// tolerance for determinant check on rank of data matrix X.
    colin_tol: F,
    /// L2 regularization parameter for ridge regression. Defaults to zero.
    l2_reg: F,
}

/// A builder to generate a Model object
impl<'a, M, Y, F> ModelBuilderData<'a, M, Y, F>
where
    M: Glm,
    Y: Response<M> + Copy,
    F: Float,
{
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
        self.use_intercept_term = false;
        self
    }

    /// Set the tolerance for the co-linearity check.
    /// The check can be effectively disabled by setting the tolerance to a negative value.
    pub fn colinearity_tolerance(mut self, tol: F) -> Self {
        self.colin_tol = tol;
        self
    }

    pub fn build(self) -> RegressionResult<Model<M, F>>
    where
        M: Glm,
        F: Float,
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

        // add constant term to X data
        let data_x = if self.use_intercept_term {
            one_pad(self.data_x)
        } else {
            self.data_x.to_owned()
        };
        // Check if the data is under-constrained
        if n_data < data_x.ncols() {
            // The regression can find a solution if n_data == ncols, but
            // there will be no estimate for the uncertainty.
            return Err(RegressionError::Underconstrained);
        }
        // Check for co-linearity up to a tolerance
        let xtx: Array2<F> = data_x.t().dot(&data_x);
        if is_rank_deficient(xtx, self.colin_tol)? {
            return Err(RegressionError::ColinearData);
        }

        // convert to floating-point
        let data_y: Array1<F> = self
            .data_y
            .iter()
            .map(|&y| y.to_float())
            .collect::<Result<_, _>>()?;

        let reg: Box<dyn Regularize<F>> = if self.l2_reg != F::zero() {
            // make the vector of L2 coefficients
            let l2_diag: Array1<F> = {
                let mut l2_diag: Array1<F> = Array1::<F>::from_elem(data_x.ncols(), self.l2_reg);
                // if an intercept term is included it should not be subject to
                // regularization.
                if self.use_intercept_term {
                    l2_diag[0] = F::zero();
                }
                l2_diag
            };
            Box::new(Ridge::from_diag(l2_diag))
        } else {
            Box::new(Null {})
        };

        Ok(Model {
            model: PhantomData,
            y: data_y,
            x: data_x,
            linear_offset: self.linear_offset,
            max_iter: self.max_iter,
            reg,
            use_intercept: self.use_intercept_term,
        })
    }
}
