//! Regularization methods and their effect on the likelihood and the matrix and
//! vector components of the IRLS step.
use crate::{error::RegressionResult, num::Float, Array1, Array2};
use ndarray::ArrayViewMut1;
use ndarray_linalg::SolveH;

/// Penalize the likelihood with a smooth function of the regression parameters.
pub(crate) trait IrlsReg<F>
where
    F: Float,
{
    /// Defines the impact of the regularization approach on the likelihood. It
    /// must be zero when the regressors are zero, otherwise some assumptions in
    /// the fitting statistics section may be invalidated.
    fn likelihood(&self, regressors: &Array1<F>) -> F;
    /// Defines the regularization effect on the gradient of the likelihood with respect
    /// to beta.
    fn gradient(&self, l: Array1<F>, regressors: &Array1<F>) -> Array1<F>;
    /// Defines the adjustment to the vector side of the IRLS update equation.
    /// It is the negative gradient of the penalty plus the hessian times the
    /// regressors. A default implementation is provided, but this is zero for
    /// ridge regression so it is allowed to be overridden.
    fn irls_vec(&self, vec: Array1<F>, regressors: &Array1<F>) -> Array1<F>;
    /// Defines the change to the matrix side of the IRLS update equation. It
    /// subtracts the Hessian of the penalty from the matrix. The difference is
    /// typically only on the diagonal.
    fn irls_mat(&self, mat: Array2<F>, regressors: &Array1<F>) -> Array2<F>;
    /// Defines additional changes to the next guess vector after the IRLS matrix solving step.
    /// This is particularly relevant for Lasso/L1 regularization as ADMM is used in an additional
    /// step to handle the discontinuities. This must be called after solving the IRLS step using
    /// the regularized Hessian and adjusted Jacobian.
    fn irls_guess(&mut self, regressors: Array1<F>) -> Array1<F>;
    /// Return the next guess under regularization given the current guess and the RHS and LHS of
    /// the unregularized IRLS matrix solution equation for the next guess.
    fn next_guess(
        &mut self,
        guess: &Array1<F>,
        irls_vec: Array1<F>,
        irls_mat: Array2<F>,
    ) -> RegressionResult<Array1<F>> {
        // Apply the regularization effects to the Hessian (LHS)
        let lhs = self.irls_mat(irls_mat, guess);
        // Apply regularization effects to the modified Jacobian (RHS)
        let rhs = self.irls_vec(irls_vec, guess);
        let next_guess = lhs.solveh_into(rhs)?;
        let next_guess = self.irls_guess(next_guess);
        Ok(next_guess)
    }
}

/// Represents a lack of regularization.
pub struct Null {}

impl<F: Float> IrlsReg<F> for Null {
    #[inline]
    fn likelihood(&self, _: &Array1<F>) -> F {
        F::zero()
    }
    #[inline]
    fn gradient(&self, jac: Array1<F>, _: &Array1<F>) -> Array1<F> {
        jac
    }
    #[inline]
    fn irls_vec(&self, vec: Array1<F>, _: &Array1<F>) -> Array1<F> {
        vec
    }
    #[inline]
    fn irls_mat(&self, mat: Array2<F>, _: &Array1<F>) -> Array2<F> {
        mat
    }
    #[inline]
    fn irls_guess(&mut self, guess: Array1<F>) -> Array1<F> {
        guess
    }
}

/// Penalizes the regression by lambda/2 * |beta|^2.
pub struct Ridge<F: Float> {
    l2_vec: Array1<F>,
}

impl<F: Float> Ridge<F> {
    /// Create the regularization from the diagonal. This outsources the
    /// question of whether to include the first term (usually the intercept) in
    /// the regularization.
    pub fn from_diag(l2: Array1<F>) -> Self {
        Self { l2_vec: l2 }
    }
}

impl<F: Float> IrlsReg<F> for Ridge<F> {
    /// The likelihood is penalized by 0.5 * lambda_2 * |beta|^2
    fn likelihood(&self, beta: &Array1<F>) -> F {
        -F::from(0.5).unwrap() * (&self.l2_vec * &beta.mapv(|b| b * b)).sum()
    }
    /// The gradient is penalized by lambda_2 * beta.
    fn gradient(&self, jac: Array1<F>, beta: &Array1<F>) -> Array1<F> {
        jac - (&self.l2_vec * beta)
    }
    /// Ridge regression has no effect on the vector side of the IRLS step equation.
    #[inline]
    fn irls_vec(&self, vec: Array1<F>, _: &Array1<F>) -> Array1<F> {
        vec
    }
    /// Add lambda to the diagonals of the information matrix.
    fn irls_mat(&self, mut mat: Array2<F>, _: &Array1<F>) -> Array2<F> {
        let mut mat_diag: ArrayViewMut1<F> = mat.diag_mut();
        mat_diag += &self.l2_vec;
        mat
    }
    /// Nothing additional is done to the guess after solving
    #[inline]
    fn irls_guess(&mut self, guess: Array1<F>) -> Array1<F> {
        guess
    }
}

/// Penalizes the likelihood by the L1-norm of the parameters.
pub struct Lasso<F: Float> {
    /// The L1 parameters for each element
    l1_vec: Array1<F>,
    /// Previous guesses (both may not be needed)
    beta: Array1<F>,
    gamma: Array1<F>,
    /// The cumulative sum of residuals
    u: F,
    /// ADMM parameter
    rho: F,
}

impl<F: Float> Lasso<F> {
    /// Create the regularization from the diagonal, outsourcing the question of whether to include
    /// the first term (commonly the intercept, which is left out) in the diagonal.
    pub fn from_diag(l1: Array1<F>) -> Self {
        let beta = Array1::zeros(l1.len());
        let gamma = Array1::zeros(l1.len());
        Self {
            l1_vec: l1,
            beta,
            gamma,
            u: F::zero(),
            rho: F::one(),
        }
    }
}

impl<F: Float> IrlsReg<F> for Lasso<F> {
    fn likelihood(&self, beta: &Array1<F>) -> F {
        -(&self.l1_vec * beta.mapv(num_traits::Float::abs)).sum()
    }

    fn gradient(&self, jac: Array1<F>, regressors: &Array1<F>) -> Array1<F> {
        jac - &self.l1_vec * &regressors.mapv(sign)
    }

    fn irls_vec(&self, _vec: Array1<F>, _regressors: &Array1<F>) -> Array1<F> {
        todo!()
    }

    fn irls_mat(&self, _mat: Array2<F>, _regressors: &Array1<F>) -> Array2<F> {
        todo!()
    }

    fn irls_guess(&mut self, _regressors: Array1<F>) -> Array1<F> {
        todo!()
    }
}

// TODO: Elastic Net (L1 + L2)
pub struct ElasticNet {}

/// Returns 1 if x > 0, -1 if x < 0, and 0 if x == 0.
fn sign<F: Float>(x: F) -> F {
    // signum returns +-1 for +-0, surprisingly.
    // https://github.com/rust-lang/rust/issues/57543
    if x == F::zero() {
        F::zero()
    } else {
        x.signum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn ridge_matrix() {
        let l = 1e-4;
        let ridge = Ridge::from_diag(array![0., l]);
        let mat = array![[0.5, 0.1], [0.1, 0.2]];
        let mut target_mat = mat.clone();
        target_mat[[1, 1]] += l;
        let dummy_beta = array![0., 0.];
        assert_eq!(ridge.irls_mat(mat, &dummy_beta), target_mat);
    }
}
