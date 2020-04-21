//! Regularization methods and their effect on the likelihood and the matrix and
//! vector components of the IRLS step.
use ndarray::{Array1, Array2, ArrayViewMut1};
use ndarray_linalg::lapack::Lapack;
use num_traits::Float;

/// Penalize the likelihood with a smooth function of the regression parameters.
pub trait Regularize<F>
where
    F: Float,
{
    /// Defines the impact of the regularization approach on the likelihood.
    fn likelihood(&self, l: F, regressors: &Array1<F>) -> F;
    /// Defines the adjustment to the vector side of the IRLS update equation.
    /// It is the negative gradient of the penalty plus the hessian times the
    /// regressors.
    fn irls_vec(&self, vec: Array1<F>, regressors: &Array1<F>) -> Array1<F>;
    /// Defines the change to the matrix side of the IRLS update equation. It
    /// subtracts the Hessian of the penalty from the matrix. The difference is
    /// typically only on the diagonal.
    fn irls_mat(&self, mat: Array2<F>, regressors: &Array1<F>) -> Array2<F>;
}

/// Represents a lack of regularization.
pub struct Null {}
impl<F: Float> Regularize<F> for Null {
    #[inline]
    fn likelihood(&self, l: F, _: &Array1<F>) -> F {
        l
    }
    #[inline]
    fn irls_vec(&self, vec: Array1<F>, _: &Array1<F>) -> Array1<F> {
        vec
    }
    #[inline]
    fn irls_mat(&self, mat: Array2<F>, _: &Array1<F>) -> Array2<F> {
        mat
    }
}

/// Penalizes the regression by lambda/2 * |beta|^2.
pub struct Ridge<F: Float + Lapack> {
    l2_vec: Array1<F>,
}
impl<F: Float + Lapack> Ridge<F> {
    /// Create the regularization from the diagonal. This outsources the
    /// question of whether to include the first term (usually the intercept) in
    /// the regularization.
    pub fn from_diag(l2: Array1<F>) -> Self {
        Self { l2_vec: l2 }
    }
}

impl<F: Float + Lapack> Regularize<F> for Ridge<F> {
    /// The likelihood is penalized by 0.5 * lambda_2 * |beta|^2
    fn likelihood(&self, l: F, beta: &Array1<F>) -> F {
        l - F::from(0.5).unwrap() * (&self.l2_vec * &beta.mapv(|b| b * b)).sum()
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
}

// TODO: Smoothed LASSO, Elastic Net (L1 + L2)

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
