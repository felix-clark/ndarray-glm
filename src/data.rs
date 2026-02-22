//! Structs and utilities to represent input fit data.
use crate::{glm::Glm, num::Float};
use ndarray::{Array1, Array2, ArrayView2, Axis, concatenate, s};
use num_traits::{FromPrimitive, One};
use std::ops::{AddAssign, DivAssign, MulAssign};

#[derive(Clone, Debug)]
pub struct Dataset<F>
where
    F: Float,
{
    /// the observation of response data by event
    pub y: Array1<F>,
    /// the design matrix with observations in rows and covariates in columns
    pub x: Array2<F>,
    /// The offset in the linear predictor for each data point. This can be used
    /// to fix the effect of control variables.
    pub linear_offset: Option<Array1<F>>,
    /// The variance weight of each observation (a.k.a. analytic weights)
    pub weights: Option<Array1<F>>,
    /// The frequency of each observation (traditionally positive integers)
    pub freqs: Option<Array1<F>>,
    /// Tracks whether the design matrix has a constant column of 1s prepended for the intercept
    /// terms
    pub(crate) has_intercept: bool,
    /// If not None, indicates that the design matrix has been standardized and holds the
    /// Standardizer object
    pub(crate) standardizer: Option<Standardizer<F>>,
}

impl<F> Dataset<F>
where
    F: Float,
{
    /// Returns the linear predictors for unstandardized (external) parameters.
    pub(crate) fn linear_predictor_ext(&self, beta: Array1<F>) -> Array1<F> {
        let regressors = match &self.standardizer {
            Some(std) => {
                if self.has_intercept {
                    std.transform_coefficients(beta)
                } else {
                    std.transform_coefficients_no_int(beta)
                }
            }
            None => beta,
        };
        self.linear_predictor_std(&regressors)
    }

    /// Returns the linear predictors, i.e. the design matrix multiplied by the
    /// regression parameters. Each entry in the resulting array is the linear
    /// predictor for a given observation. If linear offsets for each
    /// observation are provided, these are added to the linear predictors.
    /// This is no longer a public function because it is designed to take regressors in the
    /// standardized scale, and exposing that would be error-prone.
    /// It should be agnostic to the question of the intercept.
    pub(crate) fn linear_predictor_std(&self, regressors: &Array1<F>) -> Array1<F> {
        let linear_predictor: Array1<F> = self.x.dot(regressors);
        // Add linear offsets to the predictors if they are set
        if let Some(lin_offset) = &self.linear_offset {
            linear_predictor + lin_offset
        } else {
            linear_predictor
        }
    }

    /// Get the predicted y-values for *standardized* covariates under the GLM model.
    pub(crate) fn predict_with<M>(&self, beta_std: &Array1<F>) -> Array1<F>
    where
        M: Glm,
    {
        let xb: Array1<F> = self.x.dot(beta_std);
        let lin_pred: Array1<F> = if let Some(off) = &self.linear_offset {
            xb + off
        } else {
            xb
        };
        M::mean(&lin_pred)
    }

    /// Total number of observations as given by the sum of the frequencies of observations
    pub fn n_obs(&self) -> F {
        match &self.freqs {
            None => F::from(self.y.len()).unwrap(),
            Some(f) => f.sum(),
        }
    }

    /// Standardize the design matrix and store the standardizer object.
    pub(crate) fn finalize_design_matrix(&mut self, transform: bool, use_intercept: bool) {
        assert!(
            self.standardizer.is_none(),
            "This dataset is already marked as being standardized."
        );
        assert!(!self.has_intercept, "Intercept shouldn't already be set");
        // This must be set before calling Standardizer::from_dataset
        self.has_intercept = use_intercept;
        if transform {
            // NOTE: It's critical that the intercept is specified before passing to this from_dataset
            // method, because it checks pretty much everything: weights, intercept, etc.
            // This isn't a nice factorization, but at least this mess is all internal.
            let standardizer = Standardizer::from_dataset(self);
            self.x = standardizer.transform(self.x.clone());
            self.standardizer = Some(standardizer);
        }
        // Pad the ones after standardization for simplicity
        if use_intercept {
            self.x = one_pad(self.x.view());
        }
    }

    /// Returns the effective sample size corrected for the design effect. This exposes the sum of
    /// the squares of the variance weights.
    pub fn n_eff(&self) -> F {
        match &self.weights {
            None => self.n_obs(),
            Some(w) => {
                let v1 = self.freq_sum(w);
                let w2 = w * w;
                let v2 = self.freq_sum(&w2);
                v1 * v1 / v2
            }
        }
    }

    /// Multiply the input by the frequency weights
    pub(crate) fn apply_freq_weights(&self, rhs: Array1<F>) -> Array1<F> {
        match &self.freqs {
            None => rhs,
            Some(f) => f * rhs,
        }
    }

    /// multiply the input vector element-wise by the weights, if they exist
    pub(crate) fn apply_total_weights(&self, rhs: Array1<F>) -> Array1<F> {
        self.apply_freq_weights(self.apply_var_weights(rhs))
    }

    pub(crate) fn apply_var_weights(&self, rhs: Array1<F>) -> Array1<F> {
        match &self.weights {
            None => rhs,
            Some(w) => w * rhs,
        }
    }

    /// Sum over the input array using the frequencies (and not the variance weights) as weights.
    /// This is a useful operation because the frequency weights fundamentally impact the sum
    /// operator and nothing else.
    pub(crate) fn freq_sum(&self, rhs: &Array1<F>) -> F {
        self.apply_freq_weights(rhs.clone()).sum()
    }

    pub(crate) fn get_variance_weights(&self) -> Array1<F> {
        match &self.weights {
            Some(w) => w.clone(),
            None => Array1::<F>::ones(self.y.len()),
        }
    }

    /// Returns the sum of the weights, or the number of observations if the weights are all equal
    /// to 1.
    pub(crate) fn sum_weights(&self) -> F {
        match &self.weights {
            None => self.n_obs(),
            Some(w) => self.freq_sum(w),
        }
    }

    /// Return the weighted sum of the RHS, where both frequency and variance weights are used.
    pub(crate) fn weighted_sum(&self, rhs: &Array1<F>) -> F {
        self.freq_sum(&self.apply_var_weights(rhs.clone()))
    }

    /// Returns the weighted transpose of the feature data
    pub(crate) fn x_conj(&self) -> Array2<F> {
        let xt = self.x.t().to_owned();
        let xt = match &self.freqs {
            None => xt,
            Some(f) => xt * f,
        };
        match &self.weights {
            None => xt,
            Some(w) => xt * w,
        }
    }

    /// Returns the external data matrix, scaled back to the original level.
    /// NOTE: This does some unnecessary clones if there is no standardization. Perhaps we want to
    /// maintain a reference to the original matrix and use copy-on-write to optionally a
    /// standardized version.
    pub(crate) fn x_ext(&self) -> Array2<F> {
        let std = match &self.standardizer {
            Some(std) => std,
            None => return self.x.clone(),
        };
        if self.has_intercept {
            let x_tr = std.inverse_transform(self.x.slice(s![.., 1..]).to_owned());
            concatenate![Axis(1), self.x.slice(s![.., ..1]), x_tr]
        } else {
            std.inverse_transform(self.x.clone())
        }
    }

    /// Return the weighted transpose of the feature data in the original un-standardized scale
    pub(crate) fn x_conj_ext(&self) -> Array2<F> {
        let x_ext = self.x_ext().t().to_owned();
        let xt = match &self.freqs {
            None => x_ext,
            Some(f) => x_ext * f,
        };
        match &self.weights {
            None => xt,
            Some(w) => xt * w,
        }
    }

    /// Transform the parameters from standardized space back into the external one.
    pub(crate) fn inverse_transform_beta(&self, beta: Array1<F>) -> Array1<F> {
        match &self.standardizer {
            Some(std) => {
                if self.has_intercept {
                    std.inverse_transform_coefficients(beta)
                } else {
                    std.inverse_transform_coefficients_no_int(beta)
                }
            }
            None => beta,
        }
    }

    /// Transform external parameters into internal standardized ones.
    pub(crate) fn transform_beta(&self, beta: Array1<F>) -> Array1<F> {
        match &self.standardizer {
            Some(std) => {
                if self.has_intercept {
                    std.transform_coefficients(beta)
                } else {
                    std.transform_coefficients_no_int(beta)
                }
            }
            None => beta,
        }
    }

    /// Transform an internal Fisher matrix d^2/d\beta'^2 to an external representation
    /// d^2/d\beta^2.
    pub(crate) fn inverse_transform_fisher(&self, fisher: Array2<F>) -> Array2<F> {
        match &self.standardizer {
            Some(std) => {
                if self.has_intercept {
                    std.inverse_transform_fisher(fisher)
                } else {
                    std.inverse_transform_fisher_no_int(fisher)
                }
            }
            None => fisher,
        }
    }
}

/// Stores the per-column means and sample standard deviations learned from a design matrix, and
/// can apply the same transformation to new data.
///
/// This is useful when fitting a regularized model: standardize the training
/// data with [`Dataset::standardize`], then apply the same transformation
/// to held-out inputs via [`Standardizer::transform`] inside of
/// [`Fit::predict`](crate::fit::Fit::predict). Alternatively, use
/// [`Standardizer::inverse_transform_coefficients`] to convert the fitted
/// coefficients back to the original scale, however this will result only in the linear predictor
/// and one would have to apply the inverse link function in order to extract predicted
/// $`y`$-values.
///
/// # Degrees of freedom
///
/// Standard deviations are computed with ddof=1 (sample std), consistent with
/// unbiased estimation from a training set.
///
/// # Zero-variance columns
///
/// The actual standard deviation, including zero for constant columns, is
/// stored. [`transform`](Standardizer::transform) zeros out constant columns
/// rather than producing NaN. [`inverse_transform_coefficients`](Standardizer::inverse_transform_coefficients)
/// maps the corresponding coefficient to zero.
#[derive(Clone, Debug)]
pub(crate) struct Standardizer<F> {
    /// Per-column means.
    pub shifts: Array1<F>,
    /// Per-column sample standard deviations (ddof=1). May contain zeros for
    /// constant columns.
    pub scales: Array1<F>,
}

impl<F> Standardizer<F>
where
    F: Float + FromPrimitive + std::ops::DivAssign,
{
    /// Compute per-column means and sample standard deviations from `data`, using the `x` values
    /// as well as the weights, if present.
    ///
    /// The means are given by the (possibly weight) average of the values. The standard variances
    /// are the weighted average of the squared deviations from the weighted mean, times a bias
    /// term $`1/(1-1/n_eff)`$.
    ///
    /// For empty data (0 rows) the mean defaults to 0 and the std to 1.
    /// For a single row the mean is that row's value and the std defaults to 1
    /// (sample std is undefined with one observation).
    fn from_dataset(data: &Dataset<F>) -> Self {
        let x = &data.x;
        let (n, p) = (x.nrows(), x.ncols());
        let weights: Array1<F> = data.apply_total_weights(Array1::<F>::ones(n));
        let sum_weights: F = weights.sum();
        // Change the shape to broadcast to Array2 weighting
        let weights: Array2<F> = weights.insert_axis(Axis(1));
        let n_eff = data.n_eff();
        let means = if n == 0 {
            Array1::<F>::zeros(p)
        } else {
            let x_w: Array2<F> = &weights * x;
            x_w.sum_axis(Axis(0)) / sum_weights
        };

        let vars: Array1<F> = if n <= 1 {
            Array1::<F>::ones(p)
        } else {
            assert!(n_eff > F::one());
            let dx = x - means.clone();
            let dx2: Array2<F> = dx.mapv_into(|d| d * d);
            let dx2_w = weights * dx2;
            let vars = dx2_w.sum_axis(Axis(0)) / sum_weights;
            let bias = n_eff / (n_eff - F::one());
            vars * bias
        };
        // For columns with zero variance, don't scale.
        let scales = vars
            .mapv(|v| if v > F::zero() { v } else { F::one() })
            .sqrt();
        // We need the actual means to compute the scales, but if we're not using the intercept,
        // the shifts should be returned to zero.
        let shifts = if data.has_intercept {
            means
        } else {
            Array1::<F>::zeros(p)
        };

        Self { shifts, scales }
    }

    /// Apply the fitted standardization to `x`.
    ///
    /// Each column has its training mean subtracted and is divided by the training standard
    /// deviation. Columns whose training std was zero are only adjusted by the mean. The parameter
    /// should be zero in this case anyway (or perhaps undefined with no regularization), so this
    /// prevents predictions being impacted from numerical imprecisions.
    fn transform(&self, x: Array2<F>) -> Array2<F> {
        (x - &self.shifts) / &self.scales
    }

    /// Apply the inverse transformation to get back the original data.
    fn inverse_transform(&self, x: Array2<F>) -> Array2<F> {
        (x * &self.scales) + &self.shifts
    }

    /// Convert coefficient estimates from the standardized scale back to the
    /// original predictor scale. This provides coefficients that can be used to
    ///
    /// `beta` must have length equal to the rank $`K`$ which is 1 longer than the means and stds.
    /// If a non-intercept coefficient vector, prepend it with a zero before passing it as the
    /// transformed coefficients will have an intercept term regardless.
    ///
    /// The back-transformation is:
    ///
    /// ```math
    /// \tilde\beta_j = \beta_j / \sigma_j \qquad j = 1,\ldots,K-1
    /// ```
    ///
    /// ```math
    /// \tilde\beta_0 = \beta_0 - \sum_j \beta_j\,\mu_j / \sigma_j
    /// ```
    ///
    /// Predictors with zero standard deviation contribute zero to both (the
    /// fit cannot identify those coefficients).
    ///
    /// # Errors
    ///
    /// Panics if the arrays are the wrong size, like all array operations.
    fn inverse_transform_coefficients(&self, mut beta: Array1<F>) -> Array1<F> {
        // The scales are the std devs, but fallback to no scaling if there is a std of zero.
        // We'd expect columns with zero variance to result in a beta of zero anyway, but we'll try
        // to handle it precisely anyway in case some assumption is broken.
        // Scale the coefficients.
        beta.slice_mut(s![1..]).div_assign(&self.scales);
        // Adjust the intercept term
        let intercept_adjust: F = (self.shifts.clone() * beta.slice(s![1..])).sum();
        beta[0] -= intercept_adjust;

        beta
    }

    /// Do the inverse transform on the coefficients in the no-intercept context.
    /// Here scaling is applied, but not shifting.
    fn inverse_transform_coefficients_no_int(&self, beta: Array1<F>) -> Array1<F> {
        beta / &self.scales
    }

    /// Transform the coefficients from external back to internal (standardized). This should not
    /// be a public function as the user should be able to engage only with the external scale.
    fn transform_coefficients(&self, mut beta: Array1<F>) -> Array1<F> {
        let intercept_adjust = (self.shifts.clone() * beta.slice(s![1..])).sum();
        beta[0] += intercept_adjust;
        beta.slice_mut(s![1..]).mul_assign(&self.scales);
        beta
    }

    /// Transform the no-intercept coefficients from external back to internal (standardized).
    /// In the no-intercept
    fn transform_coefficients_no_int(&self, beta: Array1<F>) -> Array1<F> {
        beta * &self.scales
    }

    /// Transform the Fisher information matrix from the internal standardized representation to
    /// the external one. The fisher information is the 2nd derivative of the log-likelihood with
    /// respect to the parameters, so the transformation needs to multiply by the Jacobian and it's
    /// transpose on each end. This Jacobian is given by $`\frac{\partial \beta'_i}{\partial
    /// \beta_j}`$.
    fn inverse_transform_fisher(&self, mut fisher: Array2<F>) -> Array2<F> {
        // Express the shifts and scales as column vectors
        // let mu_vec = self.means.clone().insert_axis(Axis(1));
        let scales = &self.scales;
        let f00 = fisher[[0, 0]];
        // The order of these in-place multiplications is important. We're trying to reduce clones
        // by doing it in this order.
        let block_mult: Array2<F> = scales.clone().insert_axis(Axis(1)) * scales.t();
        // l_kk -> sigma * l_kk * sigma
        fisher.slice_mut(s![1.., 1..]).mul_assign(&block_mult);
        // l_k -> sigma * l_k
        fisher.slice_mut(s![1.., 0]).mul_assign(scales);
        fisher.slice_mut(s![0, 1..]).mul_assign(scales);
        // f00 shows up scaled by mu in the rest of the terms.
        let f00_mu = self.shifts.clone() * f00;
        // Add the sigma_kk * l_k * mu_k^T and the other side to the block. Don't assume that
        // fisher is symmetric, just in case.
        // We're using the fact that the vector components of fisher have already been scaled by
        // sigma.
        // Each of the 3 terms is an outer product of a column vector and a row vector.
        // Note: .t() on a 1D array is a no-op in ndarray, so outer products require insert_axis.
        let row0 = fisher.slice(s![0, 1..]).to_owned().insert_axis(Axis(0));
        let col0 = fisher.slice(s![1.., 0]).to_owned().insert_axis(Axis(1));
        let mu_col = self.shifts.clone().insert_axis(Axis(1));
        let mu_row = self.shifts.clone().insert_axis(Axis(0));
        let block_terms: Array2<F> =
            &mu_col * &row0 + &col0 * &mu_row + &mu_col * &f00_mu.clone().insert_axis(Axis(0));
        fisher.slice_mut(s![1.., 1..]).add_assign(&block_terms);
        // Now we add the f0*mu terms to the vector portions
        fisher.slice_mut(s![1.., 0]).add_assign(&f00_mu);
        fisher.slice_mut(s![0, 1..]).add_assign(&f00_mu);

        fisher
    }

    /// Tranform the Fisher information matrix from the internal beta' representation to the
    /// external beta representation, when no shifting is used due to the lack of an intercept.
    /// The matrix part of the Jacobian is diagonal, so we don't need to do full matrix
    /// multiplications.
    fn inverse_transform_fisher_no_int(&self, fisher: Array2<F>) -> Array2<F> {
        // Get a row vector of the scales
        let std = self.scales.to_owned().insert_axis(Axis(0));
        // The diagonal std matrix multiplies the fisher block from both sides, which should be
        // equivalent to this element-wise multiplication
        fisher * std.t() * std
    }
}

/// Prepend the input with a column of ones.
/// Used to incorporate a constant intercept term in a regression.
pub(crate) fn one_pad<T>(data: ArrayView2<T>) -> Array2<T>
where
    T: Copy + One,
{
    // create the ones column
    let ones: Array2<T> = Array2::ones((data.nrows(), 1));
    // This should be guaranteed to succeed since we are manually specifying the dimension
    concatenate![Axis(1), ones, data]
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    fn make_dataset(x: Array2<f64>) -> Dataset<f64> {
        let n = x.nrows();
        let mut ds = Dataset {
            y: Array1::zeros(n),
            x,
            linear_offset: None,
            weights: None,
            freqs: None,
            has_intercept: false,
            standardizer: None,
        };
        ds.finalize_design_matrix(true, true);
        ds
    }

    fn get_standardizer(x: Array2<f64>) -> Standardizer<f64> {
        let ds = make_dataset(x);
        ds.standardizer.unwrap()
    }

    // Basic fit: check means and sample stds (ddof=1).
    #[test]
    fn fit_means_and_stds() {
        let x = array![[1.0_f64, 4.0], [2.0, 6.0], [3.0, 8.0]];
        let s = get_standardizer(x);
        assert_abs_diff_eq!(s.shifts, array![2.0, 6.0], epsilon = 1e-12);
        // sample std: col0 = 1.0, col1 = 2.0
        assert_abs_diff_eq!(s.scales, array![1.0, 2.0], epsilon = 1e-12);
    }

    // Verify ddof=1, not ddof=0, by a case where they differ.
    #[test]
    fn fit_uses_sample_std() {
        // Two observations: ddof=1 gives sqrt(2), ddof=0 gives 1.
        let x = array![[1.0_f64], [3.0]];
        let s = get_standardizer(x);
        assert_abs_diff_eq!(s.scales[0], 2.0_f64.sqrt(), epsilon = 1e-12);
    }

    // transform produces zero-mean, unit-variance columns.
    #[test]
    fn transform_standardizes() {
        let x = array![[1.0_f64, 4.0], [2.0, 6.0], [3.0, 8.0]];
        let s = get_standardizer(x.clone());
        let x_std = s.transform(x);
        assert_abs_diff_eq!(
            x_std,
            array![[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]],
            epsilon = 1e-12
        );
    }

    // from_dataset then transform is idempotent (same result called twice).
    #[test]
    fn fit_transform_consistent() {
        let x = array![[1.0_f64, 4.0], [2.0, 6.0], [3.0, 8.0]];
        let s = get_standardizer(x.clone());
        let x_std = s.transform(x.clone());
        assert_abs_diff_eq!(s.transform(x), x_std, epsilon = 1e-12);
    }

    // n=0: means default to 0, stds default to 1.
    #[test]
    fn fit_empty() {
        let x = Array2::<f64>::zeros((0, 3));
        let s = get_standardizer(x);
        assert_abs_diff_eq!(s.shifts, array![0.0, 0.0, 0.0], epsilon = 1e-12);
        assert_abs_diff_eq!(s.scales, array![1.0, 1.0, 1.0], epsilon = 1e-12);
    }

    // n=1: mean is the single row's value, std defaults to 1.
    #[test]
    fn fit_single_row() {
        let x = array![[3.0_f64, 7.0]];
        let s = get_standardizer(x.clone());
        assert_abs_diff_eq!(s.shifts, array![3.0, 7.0], epsilon = 1e-12);
        assert_abs_diff_eq!(s.scales, array![1.0, 1.0], epsilon = 1e-12);
        // In-sample transform should give zeros.
        let x_std = s.transform(x);
        assert_abs_diff_eq!(x_std, array![[0.0, 0.0]], epsilon = 1e-12);
    }

    // Constant column: stored std is 0, transform doesn't scale them.
    #[test]
    fn transform_constant_column() {
        let x = array![[1.0_f64, 5.0], [2.0, 5.0], [3.0, 5.0]];
        let s = get_standardizer(x.clone());
        assert_abs_diff_eq!(s.scales[1], 1.0, epsilon = 1e-12);
        let x_std = s.transform(x);
        // non-constant column standardizes normally; constant column is zeroed by the shift
        assert_abs_diff_eq!(
            x_std.column(1).to_owned(),
            array![0.0, 0.0, 0.0],
            epsilon = 1e-12
        );
    }

    // inverse_transform_coefficients with intercept: slopes scale by 1/std,
    // intercept absorbs the centering correction.
    #[test]
    fn inverse_transform_with_intercept() {
        let x = array![[1.0_f64, 4.0], [2.0, 6.0], [3.0, 8.0]];
        let s = get_standardizer(x); // means=[2,6], stds=[1,2]
        let beta_std = array![0.5_f64, 2.0, 3.0]; // intercept, slope0, slope1
        let beta_raw = s.inverse_transform_coefficients(beta_std);
        // slope0: 2.0 / 1.0 = 2.0
        // slope1: 3.0 / 2.0 = 1.5
        // intercept: 0.5 - (2.0*2.0/1.0 + 3.0*6.0/2.0) = 0.5 - (4.0 + 9.0) = -12.5
        assert_abs_diff_eq!(beta_raw, array![-12.5_f64, 2.0, 1.5], epsilon = 1e-12);
    }

    // inverse_transform_coefficients without intercept: prepend zero for the
    // intercept slot as documented, then verify only the slope elements.
    #[test]
    fn inverse_transform_no_intercept() {
        let x = array![[1.0_f64, 4.0], [2.0, 6.0], [3.0, 8.0]];
        let s = get_standardizer(x); // means=[2,6], stds=[1,2]
        let beta_std = array![0.0_f64, 2.0, 3.0]; // zero intercept prepended
        let beta_raw = s.inverse_transform_coefficients(beta_std);
        // slope0: 2.0 / 1.0 = 2.0
        // slope1: 3.0 / 2.0 = 1.5
        assert_abs_diff_eq!(
            beta_raw.slice(s![1..]).to_owned(),
            array![2.0_f64, 1.5],
            epsilon = 1e-12
        );
    }

    // inverse_transform_coefficients with a zero-std column: that coefficient
    // maps to zero, but it does contribute to the zero column. Note that this is unrealistic in
    // standard applications, since a column of zero variance should lead to a beta of about zero.
    #[test]
    fn inverse_transform_zero_std_column() {
        let x = array![[1.0_f64, 5.0], [2.0, 5.0], [3.0, 5.0]];
        let s = get_standardizer(x); // means=[2,5], stds=[1,0]
        let beta_std = array![0.5_f64, 2.0, 3.0];
        let beta_raw = s.inverse_transform_coefficients(beta_std);
        // slope0: 2.0 / 1.0 = 2.0
        // slope1: zero std → 0.0
        // intercept: 0.5 - (2.0*2.0/1.0) - (3.0*5.0/1.0) = 0.5 - 4.0 - 15.0 = -18.5
        // The last element has zero std, so we don't scale it at all.
        assert_abs_diff_eq!(beta_raw, array![-18.5_f64, 2.0, 3.0], epsilon = 1e-12);
    }

    // inverse_transform_coefficients returns an error for wrong-length input.
    #[test]
    #[should_panic]
    fn inverse_transform_bad_length() {
        let x = array![[1.0_f64, 4.0], [2.0, 6.0], [3.0, 8.0]];
        let s = get_standardizer(x); // p=2, expects length 3
        let bad = array![1.0_f64, 2.0, 3.0, 4.0]; // length 4
        let _ = s.inverse_transform_coefficients(bad);
    }

    // Round-trip: standardized → external → standardized recovers the original.
    #[test]
    fn transform_inverse_transform_roundtrip() {
        let x = array![[1.0_f64, 4.0], [2.0, 6.0], [3.0, 8.0]];
        let s = get_standardizer(x); // means=[2,6], stds=[1,2]
        let beta_std = array![0.5_f64, 2.0, 3.0];
        let beta_ext = s.inverse_transform_coefficients(beta_std.clone());
        let recovered = s.transform_coefficients(beta_ext);
        assert_abs_diff_eq!(recovered, beta_std, epsilon = 1e-12);
    }

    // Round-trip: external → standardized → external recovers the original.
    #[test]
    fn inverse_transform_transform_roundtrip() {
        let x = array![[1.0_f64, 4.0], [2.0, 6.0], [3.0, 8.0]];
        let s = get_standardizer(x); // means=[2,6], stds=[1,2]
        let beta_ext = array![-12.5_f64, 2.0, 1.5];
        let beta_std = s.transform_coefficients(beta_ext.clone());
        let recovered = s.inverse_transform_coefficients(beta_std);
        assert_abs_diff_eq!(recovered, beta_ext, epsilon = 1e-12);
    }

    // Round-trips hold even when a column has zero variance (std=0).
    #[test]
    fn roundtrip_zero_std_column() {
        let x = array![[1.0_f64, 5.0], [2.0, 5.0], [3.0, 5.0]];
        let s = get_standardizer(x); // means=[2,5], stds=[1,0]

        let beta_std = array![0.5_f64, 2.0, 3.0];
        let beta_ext = s.inverse_transform_coefficients(beta_std.clone());
        let recovered_std = s.transform_coefficients(beta_ext);
        assert_abs_diff_eq!(recovered_std, beta_std, epsilon = 1e-12);

        let beta_ext2 = array![-3.5_f64, 2.0, 3.0];
        let beta_std2 = s.transform_coefficients(beta_ext2.clone());
        let recovered_ext = s.inverse_transform_coefficients(beta_std2);
        assert_abs_diff_eq!(recovered_ext, beta_ext2, epsilon = 1e-12);
    }

    // Check that both linear predictor approaches give the same value
    #[test]
    fn lin_pred_consistency() {
        let x = array![[1.0_f64, 3.0], [2.0, 5.0], [-1.0, 2.0]];
        let d = make_dataset(x);
        // d.finalize_design_matrix(true, true);
        let s = d.standardizer.as_ref().unwrap();

        let beta_std = array![1.0_f64, -1.5, 2.0];
        let beta = s.inverse_transform_coefficients(beta_std.clone());

        let lin_pred = d.linear_predictor_ext(beta);
        let lin_pred_std = d.linear_predictor_std(&beta_std);
        assert_abs_diff_eq!(lin_pred, lin_pred_std, epsilon = 1e-12);
    }
}
