//! Fit-specific configuration and fit builder
use super::Fit;
use crate::{Array1, error::RegressionResult, glm::Glm, model::Model, num::Float};

/// A builder struct for fit configuration
pub struct FitConfig<'a, M, F>
where
    M: Glm,
    F: Float,
{
    pub(crate) model: &'a Model<M, F>,
    pub options: FitOptions<F>,
}

impl<'a, M, F> FitConfig<'a, M, F>
where
    M: Glm,
    F: Float,
{
    pub fn fit(self) -> RegressionResult<Fit<'a, M, F>, F> {
        M::regression(self.model, self.options)
    }

    /// Use a maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.options.max_iter = max_iter;
        self
    }

    /// Set the tolerance of iteration
    pub fn tol(mut self, tol: F) -> Self {
        self.options.tol = tol;
        self
    }

    /// Set the L2 (ridge) regularization penalty weight.
    ///
    /// NOTE: The fit is sensitive to the scale of the data under L2 regularization. By default,
    /// the data and parameters are internally standardized so that the contributions from features
    /// with low variances relative to their offsets are not overly suppressed. The reported
    /// coefficients are transformed back to the scale of the data, so that they can be applied
    /// directly to the input data. This default is the recommended approach, and should be
    /// invisible to the user.
    ///
    /// To disable this internal regularization, use
    /// [`crate::model::ModelBuilderData::no_standardize`].
    pub fn l2_reg(mut self, l2: F) -> Self {
        self.options.l2 = l2;
        self
    }

    /// Set the L1 (lasso) regularization penalty weight.
    ///
    /// L1 regularization incurs the same scale sensitivity as L2 regularization.
    pub fn l1_reg(mut self, l1: F) -> Self {
        self.options.l1 = l1;
        self
    }
}

/// Specifies the fitting options
#[derive(Clone)]
pub struct FitOptions<F>
where
    F: Float,
{
    /// The maximum number of IRLS iterations
    pub max_iter: usize,
    /// The relative tolerance of the likelihood
    pub tol: F,
    /// The regularization of the fit
    pub l2: F,
    pub l1: F,
    /// An initial guess. A sensible default is selected if this is not provided.
    pub init_guess: Option<Array1<F>>,
}

impl<F> Default for FitOptions<F>
where
    F: Float,
{
    fn default() -> Self {
        Self {
            max_iter: 128,
            // This tolerance is rather small, but it is used in the context of a
            // fraction of the total likelihood.
            tol: F::epsilon(),
            l2: F::zero(),
            l1: F::zero(),
            init_guess: None,
        }
    }
}
