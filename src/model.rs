//! Collect data for and configure a model

use crate::{
    data::Dataset,
    error::{RegressionError, RegressionResult},
    fit::{self, Fit},
    glm::Glm,
    math::is_rank_deficient,
    num::Float,
    response::Response,
};
use fit::options::{FitConfig, FitOptions};
use ndarray::{Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Data, Ix1, Ix2};
use std::marker::PhantomData;

/// Holds the data and configuration settings for a regression.
pub struct Model<M, F>
where
    M: Glm,
    F: Float,
{
    pub(crate) model: PhantomData<M>,
    /// The dataset
    pub data: Dataset<F>,
}

impl<M, F> Model<M, F>
where
    M: Glm,
    F: Float,
{
    /// Perform the regression and return a fit object holding the results.
    pub fn fit(&self) -> RegressionResult<Fit<'_, M, F>, F> {
        self.fit_options().fit()
    }

    /// Fit options builder interface
    pub fn fit_options(&self) -> FitConfig<'_, M, F> {
        FitConfig {
            model: self,
            options: FitOptions::default(),
        }
    }

    /// An experimental interface that would allow fit options to be set externally.
    pub fn with_options(&self, options: FitOptions<F>) -> FitConfig<'_, M, F> {
        FitConfig {
            model: self,
            options,
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
    pub fn data<'a, Y, F, YD, XD>(
        data_y: &'a ArrayBase<YD, Ix1>,
        data_x: &'a ArrayBase<XD, Ix2>,
    ) -> ModelBuilderData<'a, M, Y, F>
    where
        Y: Response<M>,
        F: Float,
        YD: Data<Elem = Y>,
        XD: Data<Elem = F>,
    {
        ModelBuilderData {
            model: PhantomData,
            data_y: data_y.view(),
            data_x: data_x.view(),
            linear_offset: None,
            var_weights: None,
            freq_weights: None,
            use_intercept_term: true,
            standardize: true,
            colin_tol: F::epsilon(),
            error: None,
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
    /// observation and each column represents a different independent variable.
    data_x: ArrayView2<'a, F>,
    /// The offset in the linear predictor for each data point. This can be used
    /// to incorporate control terms.
    // TODO: consider making this a reference/ArrayView. Y and X are effectively
    // cloned so perhaps this isn't a big deal.
    linear_offset: Option<Array1<F>>,
    /// The variance/analytic weights for each observation.
    var_weights: Option<Array1<F>>,
    /// The frequency/count of each observation.
    freq_weights: Option<Array1<F>>,
    /// Whether to standardize the input data. Defaults to `true`.
    standardize: bool,
    /// Whether to use an intercept term. Defaults to `true`.
    use_intercept_term: bool,
    /// tolerance for determinant check on rank of data matrix X.
    colin_tol: F,
    /// An error that has come up in the build compilation.
    error: Option<RegressionError<F>>,
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
        if self.linear_offset.is_some() {
            self.error = Some(RegressionError::BuildError(
                "Offsets specified multiple times".to_string(),
            ));
        }
        self.linear_offset = Some(linear_offset);
        self
    }

    /// Frequency weights (a.k.a. counts) for each observation. Traditionally these are positive
    /// integers representing the number of times each observation appears identically.
    pub fn freq_weights(mut self, freqs: Array1<usize>) -> Self {
        if self.freq_weights.is_some() {
            self.error = Some(RegressionError::BuildError(
                "Frequency weights specified multiple times".to_string(),
            ));
        }
        let ffreqs: Array1<F> = freqs.mapv(|c| F::from(c).unwrap());
        // TODO: consider adding a check for non-negative weights
        self.freq_weights = Some(ffreqs);
        self
    }

    /// Variance weights (a.k.a. analytic weights) of each observation. These could represent the
    /// inverse square of the uncertainties of each measurement.
    pub fn var_weights(mut self, weights: Array1<F>) -> Self {
        if self.var_weights.is_some() {
            self.error = Some(RegressionError::BuildError(
                "Variance weights specified multiple times".to_string(),
            ));
        }
        // TODO: consider adding a check for non-negative weights
        self.var_weights = Some(weights);
        self
    }

    /// Do not add a constant intercept term of `1`s to the design matrix. This is rarely
    /// recommended, so you probably don't want to use this option unless you have a very clear
    /// sense of why. Note that you can supply uniform or per-observation constant terms using
    /// [`ModelBuilderData::linear_offset`].
    pub fn no_constant(mut self) -> Self {
        self.use_intercept_term = false;
        self
    }

    /// Don't perform standarization (i.e. scale to 0-mean and 1-variance) of the design matrix.
    /// Note that the standardization is handled internally, so the reported result coefficients
    /// should be compatible with the input data directly, meaning the user shouldn't have to
    /// interact with them.
    pub fn no_standardize(mut self) -> Self {
        self.standardize = false;
        self
    }

    /// Set the tolerance for the co-linearity check.
    /// The check can be effectively disabled by setting the tolerance to a negative value.
    pub fn colinear_tol(mut self, tol: F) -> Self {
        self.colin_tol = tol;
        self
    }

    pub fn build(self) -> RegressionResult<Model<M, F>, F>
    where
        M: Glm,
        F: Float,
    {
        if let Some(err) = self.error {
            return Err(err);
        }

        let n_data = self.data_y.len();
        if n_data != self.data_x.nrows() {
            return Err(RegressionError::BadInput(
                "y and x data must have same number of points".to_string(),
            ));
        }
        // If they are provided, check that the offsets have the correct number of entries
        if let Some(lin_off) = &self.linear_offset
            && n_data != lin_off.len()
        {
            return Err(RegressionError::BadInput(
                "Offsets must have same dimension as observations".to_string(),
            ));
        }

        // Check if the data is under-constrained
        if n_data < self.data_x.ncols() {
            // The regression can find a solution if n_data == ncols, but there will be
            // no estimate for the uncertainty. Regularization can solve this, so keep
            // it to a warning.
            // return Err(RegressionError::Underconstrained);
            eprintln!("Warning: data is underconstrained");
        }

        // Check for co-linearity up to a tolerance
        // NOTE: Should this use the weights? If so, it should be checked after the
        // unpadded Dataset is built so we can use x_conj(). The weights might not impact the
        // collinearity check, though, since they are applied to each column equally.
        let xtx: Array2<F> = self.data_x.t().dot(&self.data_x);
        if is_rank_deficient(xtx, self.colin_tol)? {
            return Err(RegressionError::ColinearData {
                tol: self.colin_tol,
            });
        }

        // convert y-values to floating-point
        let data_y: Array1<F> = self
            .data_y
            .iter()
            .map(|&y| y.into_float())
            .collect::<Result<_, _>>()?;

        // Build the Dataset object
        let mut data = Dataset {
            y: data_y,
            x: self.data_x.to_owned(),
            linear_offset: self.linear_offset,
            weights: self.var_weights,
            freqs: self.freq_weights,
            has_intercept: false,
            standardizer: None,
        };

        data.finalize_design_matrix(self.standardize, self.use_intercept_term);

        Ok(Model {
            model: PhantomData,
            data,
        })
    }
}
