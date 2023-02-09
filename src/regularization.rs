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

    /// Processing to do before each step.
    fn prepare(&mut self, _guess: &Array1<F>) {}

    /// For ADMM, the likelihood in the IRLS step is augmented with a rho term and does not include
    /// the L1 component. Without ADMM this should return the actual un-augmented likelihood.
    fn irls_like(&self, regressors: &Array1<F>) -> F {
        self.likelihood(regressors)
    }

    /// Defines the adjustment to the vector side of the IRLS update equation.
    /// It is the negative gradient of the penalty plus the hessian times the
    /// regressors. A default implementation is provided, but this is zero for
    /// ridge regression so it is allowed to be overridden.
    fn irls_vec(&self, vec: Array1<F>, regressors: &Array1<F>) -> Array1<F>;

    /// Defines the change to the matrix side of the IRLS update equation. It
    /// subtracts the Hessian of the penalty from the matrix. The difference is
    /// typically only on the diagonal.
    fn irls_mat(&self, mat: Array2<F>, regressors: &Array1<F>) -> Array2<F>;

    /// Return the next guess under regularization given the current guess and the RHS and LHS of
    /// the unregularized IRLS matrix solution equation for the next guess.
    fn next_guess(
        &mut self,
        guess: &Array1<F>,
        irls_vec: Array1<F>,
        irls_mat: Array2<F>,
    ) -> RegressionResult<Array1<F>> {
        self.prepare(guess);
        // Apply the regularization effects to the Hessian (LHS)
        let lhs = self.irls_mat(irls_mat, guess);
        // Apply regularization effects to the modified Jacobian (RHS)
        let rhs = self.irls_vec(irls_vec, guess);
        let next_guess = lhs.solveh_into(rhs)?;
        Ok(next_guess)
    }

    fn terminate_ok(&self, _tol: F) -> bool {
        true
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
    /// Ridge regression has no effect on the vector side of the IRLS step equation, because the
    /// 1st and 2nd order derivative terms exactly cancel.
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

/// Penalizes the likelihood by the L1-norm of the parameters.
pub struct Lasso<F: Float> {
    /// The L1 parameters for each element
    l1_vec: Array1<F>,
    /// The dual solution
    dual: Array1<F>,
    /// The cumulative sum of residuals for each element
    cum_res: Array1<F>,
    /// ADMM penalty parameter
    rho: F,
    /// L2-Norm of primal residuals |r|^2
    r_sq: F,
    /// L2-Norm of dual residuals |s|^2
    s_sq: F,
}

impl<F: Float> Lasso<F> {
    /// Create the regularization from the diagonal, outsourcing the question of whether to include
    /// the first term (commonly the intercept, which is left out) in the diagonal.
    pub fn from_diag(l1: Array1<F>) -> Self {
        let n: usize = l1.len();
        let gamma = Array1::zeros(n);
        let u = Array1::zeros(n);
        Self {
            l1_vec: l1,
            dual: gamma,
            cum_res: u,
            rho: F::one(),
            r_sq: F::infinity(), // or should it be NaN?
            s_sq: F::infinity(),
        }
    }

    fn update_rho(&mut self) {
        // Can these be declared const?
        let mu: F = F::from(8.).unwrap();
        let tau: F = F::from(2.).unwrap();
        if self.r_sq > mu * mu * self.s_sq {
            self.rho *= tau;
            self.cum_res /= tau;
        }
        if self.r_sq * mu * mu < self.s_sq {
            self.rho /= tau;
            self.cum_res *= tau;
        }
    }
}

impl<F: Float> IrlsReg<F> for Lasso<F> {
    fn likelihood(&self, beta: &Array1<F>) -> F {
        -(&self.l1_vec * beta.mapv(num_traits::Float::abs)).sum()
    }

    // This is used in the fit's score function, for instance. Thus it includes the regularization
    // terms and not the augmented term.
    fn gradient(&self, jac: Array1<F>, regressors: &Array1<F>) -> Array1<F> {
        jac - &self.l1_vec * &regressors.mapv(F::sign)
    }

    /// Update the dual solution and the cumulative residuals.
    fn prepare(&mut self, beta: &Array1<F>) {
        // Apply adaptive penalty term updating
        self.update_rho();

        let old_dual = self.dual.clone();

        self.dual = soft_thresh(beta + &self.cum_res, &self.l1_vec / self.rho);
        // the primal residuals
        let r: Array1<F> = beta - &self.dual;
        // the dual residuals
        let s: Array1<F> = (&self.dual - old_dual) * self.rho;
        self.cum_res += &r;

        self.r_sq = r.mapv(|r| r * r).sum();
        self.s_sq = s.mapv(|s| s * s).sum();
    }

    fn irls_like(&self, regressors: &Array1<F>) -> F {
        -F::from(0.5).unwrap()
            * self.rho
            * (regressors - &self.dual + &self.cum_res)
                .mapv(|x| x * x)
                .sum()
    }

    /// The beta term from the gradient is cancelled by the corresponding term from the Hessian.
    /// The dual and residual terms remain.
    fn irls_vec(&self, vec: Array1<F>, _regressors: &Array1<F>) -> Array1<F> {
        let d: Array1<F> = &self.dual - &self.cum_res;
        vec + d * self.rho
    }

    /// Add the constant rho to all elements of the diagonal of the Hessian.
    fn irls_mat(&self, mut mat: Array2<F>, _: &Array1<F>) -> Array2<F> {
        let mut mat_diag: ArrayViewMut1<F> = mat.diag_mut();
        mat_diag += self.rho;
        mat
    }

    fn terminate_ok(&self, tol: F) -> bool {
        // Expressed like this, it should perhaps instead be an epsilon^2.
        let n: usize = self.dual.len();
        let n_sq = F::from((n as f64).sqrt()).unwrap();
        let r_pass = self.r_sq < n_sq * tol;
        let s_pass = self.s_sq < n_sq * tol;
        r_pass && s_pass
    }
}

/// Penalizes the likelihood with both an L1-norm and L2-norm.
pub struct ElasticNet<F: Float> {
    /// The L1 parameters for each element
    l1_vec: Array1<F>,
    /// The L2 parameters for each element
    l2_vec: Array1<F>,
    /// The dual solution
    dual: Array1<F>,
    /// The cumulative sum of residuals for each element
    cum_res: Array1<F>,
    /// ADMM penalty parameter
    rho: F,
    /// L2-Norm of primal residuals |r|^2
    r_sq: F,
    /// L2-Norm of dual residuals |s|^2
    s_sq: F,
}

impl<F: Float> ElasticNet<F> {
    /// Create the regularization from the diagonal, outsourcing the question of whether to include
    /// the first term (commonly the intercept, which is left out) in the diagonal.
    pub fn from_diag(l1: Array1<F>, l2: Array1<F>) -> Self {
        let n: usize = l1.len();
        let gamma = Array1::zeros(n);
        let u = Array1::zeros(n);
        Self {
            l1_vec: l1,
            l2_vec: l2,
            dual: gamma,
            cum_res: u,
            rho: F::one(),
            r_sq: F::infinity(), // or should it be NaN?
            s_sq: F::infinity(),
        }
    }

    fn update_rho(&mut self) {
        // Can these be declared const?
        let mu: F = F::from(8.).unwrap();
        let tau: F = F::from(2.).unwrap();
        if self.r_sq > mu * mu * self.s_sq {
            self.rho *= tau;
            self.cum_res /= tau;
        }
        if self.r_sq * mu * mu < self.s_sq {
            self.rho /= tau;
            self.cum_res *= tau;
        }
    }
}

impl<F: Float> IrlsReg<F> for ElasticNet<F> {
    fn likelihood(&self, beta: &Array1<F>) -> F {
        -(&self.l1_vec * beta.mapv(num_traits::Float::abs)).sum()
            -F::from(0.5).unwrap() * (&self.l2_vec * &beta.mapv(|b| b * b)).sum()
    }

    // This is used in the fit's score function, for instance. Thus it includes the regularization
    // terms and not the augmented term.
    fn gradient(&self, jac: Array1<F>, regressors: &Array1<F>) -> Array1<F> {
        jac - &self.l1_vec * &regressors.mapv(F::sign) - &self.l2_vec * regressors
    }

    /// Update the dual solution and the cumulative residuals.
    fn prepare(&mut self, beta: &Array1<F>) {
        // Apply adaptive penalty term updating
        self.update_rho();

        let old_dual = self.dual.clone();
       
        self.dual = soft_thresh(beta + &self.cum_res, &self.l1_vec / self.rho);
        // the primal residuals
        let r: Array1<F> = beta - &self.dual;
        // the dual residuals
        let s: Array1<F> = (&self.dual - old_dual) * self.rho;
        self.cum_res += &r;

        self.r_sq = r.mapv(|r| r * r).sum();
        self.s_sq = s.mapv(|s| s * s).sum();
    }

    fn irls_like(&self, regressors: &Array1<F>) -> F {
        -F::from(0.5).unwrap()
            * self.rho
            * (regressors - &self.dual + &self.cum_res)
                .mapv(|x| x * x)
                .sum()
            -F::from(0.5).unwrap() * (&self.l2_vec * &regressors.mapv(|b| b * b)).sum()
    }

    /// The beta term from the gradient is cancelled by the corresponding term from the Hessian.
    /// The dual and residual terms remain.
    fn irls_vec(&self, vec: Array1<F>, _regressors: &Array1<F>) -> Array1<F> {
        let d: Array1<F> = &self.dual - &self.cum_res;
        vec + d * self.rho
    }

    /// Add the constant rho to all elements of the diagonal of the Hessian.
    fn irls_mat(&self, mut mat: Array2<F>, _: &Array1<F>) -> Array2<F> {
        let mut mat_diag: ArrayViewMut1<F> = mat.diag_mut();
        mat_diag += &self.l2_vec;
        mat_diag += self.rho;
        mat
    }

    fn terminate_ok(&self, tol: F) -> bool {
        // Expressed like this, it should perhaps instead be an epsilon^2.
        let n: usize = self.dual.len();
        let n_sq = F::from((n as f64).sqrt()).unwrap();
        let r_pass = self.r_sq < n_sq * tol;
        let s_pass = self.s_sq < n_sq * tol;
        r_pass && s_pass
    }
}

/// The soft thresholding operator
fn soft_thresh<F: Float>(x: Array1<F>, lambda: Array1<F>) -> Array1<F> {
    let sign_x = x.mapv(F::sign);
    let abs_x = x.mapv(<F as num_traits::Float>::abs);
    let red_x = abs_x - lambda;
    let clipped = red_x.mapv(|x| if x < F::zero() { F::zero() } else { x });
    sign_x * clipped
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
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

    #[test]
    fn soft_thresh_correct() {
        let x = array![0.25, -0.1, -0.4, 0.3, 0.5, -0.5];
        let lambda = array![-0., 0.0, 0.1, 0.1, 1.0, 1.0];
        let target = array![0.25, -0.1, -0.3, 0.2, 0., 0.];
        let output = soft_thresh(x, lambda);
        assert_abs_diff_eq!(target, output);
    }
}
