//! Fit-specific configuration and fit builder
use super::Fit;
use crate::{
    error::RegressionResult,
    glm::Glm,
    model::Model,
    num::Float,
    regularization::{IrlsReg, LassoSmooth, Null, Ridge},
    Array1,
};

/// A builder struct for fit configuration
pub struct FitConfig<'a, M, F>
where
    M: Glm,
    F: Float,
{
    pub(crate) model: &'a Model<M, F>,
    pub(crate) options: FitOptions<F>,
}

impl<'a, M, F> FitConfig<'a, M, F>
where
    M: Glm,
    F: Float,
{
    pub fn fit(self) -> RegressionResult<Fit<'a, M, F>> {
        M::regression(self.model, self.options)
    }

    /// Use a maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.options.max_iter = max_iter;
        self
    }

    /// Use to set a L2 regularization parameter
    pub fn l2_reg(mut self, l2: F) -> Self {
        // This check could be made at compile-time with more complex typing, but it
        // will be kept simple for now. There isn't yet support for elastic net, but
        // calling both types of regularization could induce it.
        if !self.options.reg.as_ref().is_null() {
            eprintln!("WARNING: regularization set twice")
        }
        self.options.reg = {
            // make the vector of L2 coefficients
            let l2_diag: Array1<F> = {
                let mut l2_diag: Array1<F> = Array1::<F>::from_elem(self.model.x.ncols(), l2);
                // if an intercept term is included it should not be subject to
                // regularization.
                if self.model.use_intercept {
                    l2_diag[0] = F::zero();
                }
                l2_diag
            };
            Box::new(Ridge::from_diag(l2_diag))
        };
        self
    }

    /// Use to set a L1 regularization parameter with a smoother tolerance
    pub fn l1_smooth_reg(mut self, l1: F, eps: F) -> Self {
        if !self.options.reg.as_ref().is_null() {
            eprintln!("WARNING: regularization set twice")
        }
        self.options.reg = {
            let l1_diag: Array1<F> = {
                let mut l1_diag: Array1<F> = Array1::<F>::from_elem(self.model.x.ncols(), l1);
                // if an intercept term is included it should not be subject to
                // regularization.
                if self.model.use_intercept {
                    l1_diag[0] = F::zero();
                }
                l1_diag
            };
            Box::new(LassoSmooth::from_diag(l1_diag, eps))
        };
        self
    }
}

/// Specifies the fitting options
pub struct FitOptions<F>
where
    F: Float,
{
    pub max_iter: usize,
    pub tol: F,
    pub reg: Box<dyn IrlsReg<F>>,
    pub init_guess: Option<Array1<F>>,
    pub max_step_halves: usize,
}

impl<F> Default for FitOptions<F>
where
    F: Float,
{
    fn default() -> Self {
        Self {
            max_iter: 50,
            // This tolerance is rather small, but it is used as a fraction of the likelihood.
            tol: F::epsilon(),
            reg: Box::new(Null {}),
            init_guess: None,
            max_step_halves: 8,
        }
    }
}
