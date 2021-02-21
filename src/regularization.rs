//! Regularization methods and their effect on the likelihood and the matrix and
//! vector components of the IRLS step.
use crate::{num::Float, Array1, Array2};
use ndarray::ArrayViewMut1;

/// Penalize the likelihood with a smooth function of the regression parameters.
pub trait IrlsReg<F>
where
    F: Float,
{
    /// Defines the impact of the regularization approach on the likelihood. It
    /// must be zero when the regressors are zero, otherwise some assumptions in
    /// the fitting statistics section may be invalidated.
    fn likelihood(&self, l: F, regressors: &Array1<F>) -> F;
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

    /// True for the empty regularization
    fn is_null(&self) -> bool;
}

/// Represents a lack of regularization.
pub struct Null {}
impl<F: Float> IrlsReg<F> for Null {
    #[inline]
    fn likelihood(&self, l: F, _: &Array1<F>) -> F {
        l
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

    fn is_null(&self) -> bool {
        true
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
    fn likelihood(&self, l: F, beta: &Array1<F>) -> F {
        l - F::from(0.5).unwrap() * (&self.l2_vec * &beta.mapv(|b| b * b)).sum()
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

    fn is_null(&self) -> bool {
        false
    }
}

/// Penalizes the regression by lambda * s * log(cosh(beta/s)) which approximates
/// lambda * |beta| for |beta| >> s.
/// This approach is largely untested.
pub struct LassoSmooth<F: Float> {
    l1_vec: Array1<F>,
    /// The smoothing parameter. Regression parameters with a magnitude comparable to or
    /// less than this value can be interpreted as being "zeroed".
    // This could probably be a vector as well.
    s: F,
}

impl<F: Float> LassoSmooth<F> {
    /// Create the regularization from the diagonal. This outsources the
    /// question of whether to include the first term (usually the intercept) in
    /// the regularization.
    pub fn from_diag(l1: Array1<F>, s: F) -> Self {
        Self { l1_vec: l1, s }
    }

    // helper function for the sech^2 terms
    fn sech_sq(&self, beta: F) -> F {
        let sech = num_traits::Float::cosh(beta / self.s).recip();
        sech * sech
    }
}

impl<F: Float> IrlsReg<F> for LassoSmooth<F> {
    /// The likelihood is penalized by lambda_1 * s * log(cosh(beta/s))
    fn likelihood(&self, l: F, beta: &Array1<F>) -> F {
        l - (&self.l1_vec
            * &beta.mapv(|b| self.s * num_traits::Float::ln(num_traits::Float::cosh(b / self.s))))
            .sum()
    }
    /// The gradient is penalized by lambda_1 * tanh(beta/s)
    fn gradient(&self, jac: Array1<F>, beta: &Array1<F>) -> Array1<F> {
        jac - (&self.l1_vec * &beta.mapv(|b| num_traits::Float::tanh(b / self.s)))
    }
    /// The gradient and hessian terms of the penalty don't cancel here.
    fn irls_vec(&self, vec: Array1<F>, beta: &Array1<F>) -> Array1<F> {
        vec - &self.l1_vec
            * &beta.mapv(|b| num_traits::Float::tanh(b / self.s) - (b / self.s) * self.sech_sq(b))
    }
    /// Add sech^2 term to diagonals of Hessian.
    fn irls_mat(&self, mut mat: Array2<F>, beta: &Array1<F>) -> Array2<F> {
        let mut mat_diag: ArrayViewMut1<F> = mat.diag_mut();
        mat_diag += &(&self.l1_vec * &beta.mapv(|b| self.s.recip() * self.sech_sq(b)));
        mat
    }

    fn is_null(&self) -> bool {
        false
    }
}

// TODO: Piecewise Lasso with a penalty shaped like \_/
// TODO: Elastic Net (L1 + L2)

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
