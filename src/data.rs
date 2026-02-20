//! Structs and utilities to represent input fit data.
use crate::{
    error::{RegressionError, RegressionResult},
    num::Float,
};
use ndarray::{Array1, Array2, ArrayView2, Axis, concatenate};
use num_traits::{FromPrimitive, One};
use std::ops::DivAssign;

#[derive(Clone)]
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
    pub has_intercept: bool,
    /// If not None, indicates that the design matrix has been standardized and holds the
    /// Standardizer object
    pub(crate) standardizer: Option<Standardizer<F>>,
}

impl<F> Dataset<F>
where
    F: Float,
{
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

    /// Total number of observations as given by the sum of the frequencies of observations
    pub fn n_obs(&self) -> F {
        match &self.freqs {
            None => F::from(self.y.len()).unwrap(),
            Some(f) => f.sum(),
        }
    }

    /// Standardize the design matrix and return the standardizer object.
    pub(crate) fn standardize(&mut self) {
        if self.has_intercept {
            eprintln!("WARNING: post-padded standardization not implemented");
        }
        if let None = self.standardizer {
            eprintln!("WARNING: This dataset is already marked as being standardized. Skipping.");
            return;
        }
        let standardizer = Standardizer::from_dataset(&self);
        self.x = standardizer.transform(self.x.clone());
        self.standardizer = Some(standardizer);
    }

    pub(crate) fn pad_ones(&mut self) {
        if self.has_intercept {
            eprintln!("WARNING: This dataset already has intercept padding. Skipping.");
            return;
        }
        self.x = one_pad(self.x.view());
    }

    /// Returns the sum of the weights, or the number of observations if the weights are all equal
    /// to 1.
    pub(crate) fn sum_weights(&self) -> F {
        match &self.weights {
            None => self.n_obs(),
            Some(w) => self.freq_sum(w),
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

    pub(crate) fn get_variance_weights(&self) -> Array1<F> {
        match &self.weights {
            Some(w) => w.clone(),
            None => Array1::<F>::ones(self.y.len()),
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
#[derive(Clone)]
pub(crate) struct Standardizer<F> {
    /// Per-column means.
    pub means: Array1<F>,
    /// Per-column sample standard deviations (ddof=1). May contain zeros for
    /// constant columns.
    pub stds: Array1<F>,
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
        // These
        let weights: Array1<F> = data.apply_total_weights(Array1::<F>::ones(n));
        let sum_weights: F = weights.sum();
        let n_eff = data.n_eff();
        assert_eq!(n == 0, n_eff == F::zero());
        assert_eq!(n == 1, n_eff <= F::one());
        let means = if n_eff == F::zero() {
            Array1::<F>::zeros(p)
        } else {
            let x_w: Array2<F> = &weights * x;
            x_w.sum_axis(Axis(0)) / sum_weights
        };

        let vars: Array1<F> = if n_eff <= F::one() {
            Array1::<F>::ones(p)
        } else {
            let dx = x - means.clone();
            let dx2: Array2<F> = dx.mapv_into(|d| d * d);
            let dx2_w = weights * dx2;
            let vars = dx2_w.sum_axis(Axis(0)) / sum_weights;
            let bias = n_eff / (n_eff - F::one());
            vars * bias
        };
        let stds = vars.sqrt();

        Self { means, stds }
    }

    /// Apply the fitted standardization to `x`.
    ///
    /// Each column has its training mean subtracted and is divided by the training standard
    /// deviation. Columns whose training std was zero are only adjusted by the mean. The parameter
    /// should be zero in this case anyway (or perhaps undefined with no regularization), so this
    /// prevents predictions being impacted from numerical imprecisions.
    pub(crate) fn transform(&self, x: Array2<F>) -> Array2<F> {
        let mut x = &x - &self.means;
        x.zip_mut_with(&self.stds, |xi, &s| {
            if s > F::zero() {
                *xi /= s;
            }
            // If the scale is zero, don't scale but just leave the mean subtracted.
        });
        x
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
    /// Returns [`RegressionError::BadInput`] if `beta.len()` is not $`K`$.
    /// NOTE: This error may not be needed if all use is internal.
    pub(crate) fn inverse_transform_coefficients(
        &self,
        beta: &Array1<F>,
    ) -> RegressionResult<Array1<F>> {
        use ndarray::s;
        let p = self.means.len();
        if beta.len() != p + 1 {
            return Err(RegressionError::BadInput(format!(
                "coefficient vector length {} does not match expected {}",
                beta.len(),
                p + 1,
            )));
        };

        let mut result = beta.clone();
        // The scales are the std devs, but fallback to no scaling if there is a std of zero.
        // We'd expect columns with zero variance to result in a beta of zero anyway, but we'll try
        // to handle it precisely anyway in case some assumption is broken.
        let scales: Array1<F> = self.stds.mapv(|s| if s > F::zero() { s } else { F::one() });
        // Scale the coefficients.
        result.slice_mut(s![1..]).div_assign(&scales);
        // Adjust the intercept term
        let intercept_adjust: F = (self.means.clone() * result.slice(s![1..])).sum();
        result[0] -= intercept_adjust;

        Ok(result)
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
    use ndarray::{array, s};

    fn make_dataset(x: Array2<f64>) -> Dataset<f64> {
        let n = x.nrows();
        Dataset {
            y: Array1::zeros(n),
            x,
            linear_offset: None,
            weights: None,
            freqs: None,
            has_intercept: false,
            standardizer: None,
        }
    }

    // Basic fit: check means and sample stds (ddof=1).
    #[test]
    fn fit_means_and_stds() {
        let x = array![[1.0_f64, 4.0], [2.0, 6.0], [3.0, 8.0]];
        let s = Standardizer::from_dataset(&make_dataset(x));
        assert_abs_diff_eq!(s.means, array![2.0, 6.0], epsilon = 1e-12);
        // sample std: col0 = 1.0, col1 = 2.0
        assert_abs_diff_eq!(s.stds, array![1.0, 2.0], epsilon = 1e-12);
    }

    // Verify ddof=1, not ddof=0, by a case where they differ.
    #[test]
    fn fit_uses_sample_std() {
        // Two observations: ddof=1 gives sqrt(2), ddof=0 gives 1.
        let x = array![[1.0_f64], [3.0]];
        let s = Standardizer::from_dataset(&make_dataset(x));
        assert_abs_diff_eq!(s.stds[0], 2.0_f64.sqrt(), epsilon = 1e-12);
    }

    // transform produces zero-mean, unit-variance columns.
    #[test]
    fn transform_standardizes() {
        let x = array![[1.0_f64, 4.0], [2.0, 6.0], [3.0, 8.0]];
        let s = Standardizer::from_dataset(&make_dataset(x.clone()));
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
        let s = Standardizer::from_dataset(&make_dataset(x.clone()));
        let x_std = s.transform(x.clone());
        assert_abs_diff_eq!(s.transform(x), x_std, epsilon = 1e-12);
    }

    // n=0: means default to 0, stds default to 1.
    #[test]
    fn fit_empty() {
        let x = Array2::<f64>::zeros((0, 3));
        let s = Standardizer::from_dataset(&make_dataset(x));
        assert_abs_diff_eq!(s.means, array![0.0, 0.0, 0.0], epsilon = 1e-12);
        assert_abs_diff_eq!(s.stds, array![1.0, 1.0, 1.0], epsilon = 1e-12);
    }

    // n=1: mean is the single row's value, std defaults to 1.
    #[test]
    fn fit_single_row() {
        let x = array![[3.0_f64, 7.0]];
        let s = Standardizer::from_dataset(&make_dataset(x.clone()));
        assert_abs_diff_eq!(s.means, array![3.0, 7.0], epsilon = 1e-12);
        assert_abs_diff_eq!(s.stds, array![1.0, 1.0], epsilon = 1e-12);
        // In-sample transform should give zeros.
        let x_std = s.transform(x);
        assert_abs_diff_eq!(x_std, array![[0.0, 0.0]], epsilon = 1e-12);
    }

    // Constant column: stored std is 0, transform zeros the column out.
    #[test]
    fn transform_constant_column() {
        let x = array![[1.0_f64, 5.0], [2.0, 5.0], [3.0, 5.0]];
        let s = Standardizer::from_dataset(&make_dataset(x.clone()));
        assert_abs_diff_eq!(s.stds[1], 0.0, epsilon = 1e-12);
        let x_std = s.transform(x);
        // non-constant column standardizes normally; constant column is zeroed
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
        let s = Standardizer::from_dataset(&make_dataset(x)); // means=[2,6], stds=[1,2]
        let beta_std = array![0.5_f64, 2.0, 3.0]; // intercept, slope0, slope1
        let beta_raw = s.inverse_transform_coefficients(&beta_std).unwrap();
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
        let s = Standardizer::from_dataset(&make_dataset(x)); // means=[2,6], stds=[1,2]
        let beta_std = array![0.0_f64, 2.0, 3.0]; // zero intercept prepended
        let beta_raw = s.inverse_transform_coefficients(&beta_std).unwrap();
        // slope0: 2.0 / 1.0 = 2.0
        // slope1: 3.0 / 2.0 = 1.5
        assert_abs_diff_eq!(
            beta_raw.slice(s![1..]).to_owned(),
            array![2.0_f64, 1.5],
            epsilon = 1e-12
        );
    }

    // inverse_transform_coefficients with a zero-std column: that coefficient
    // maps to zero, and it does not contribute to the intercept correction.
    #[test]
    fn inverse_transform_zero_std_column() {
        let x = array![[1.0_f64, 5.0], [2.0, 5.0], [3.0, 5.0]];
        let s = Standardizer::from_dataset(&make_dataset(x)); // means=[2,5], stds=[1,0]
        let beta_std = array![0.5_f64, 2.0, 3.0];
        let beta_raw = s.inverse_transform_coefficients(&beta_std).unwrap();
        // slope0: 2.0 / 1.0 = 2.0
        // slope1: zero std → 0.0
        // intercept: 0.5 - (2.0*2.0/1.0) = 0.5 - 4.0 = -3.5
        // The last element has zero std, so we don't scale it at all.
        assert_abs_diff_eq!(beta_raw, array![-3.5_f64, 2.0, 3.0], epsilon = 1e-12);
    }

    // inverse_transform_coefficients returns an error for wrong-length input.
    #[test]
    fn inverse_transform_bad_length() {
        let x = array![[1.0_f64, 4.0], [2.0, 6.0], [3.0, 8.0]];
        let s = Standardizer::from_dataset(&make_dataset(x)); // p=2, expects length 3
        let bad = array![1.0_f64, 2.0, 3.0, 4.0]; // length 4
        assert!(matches!(
            s.inverse_transform_coefficients(&bad),
            Err(RegressionError::BadInput(_))
        ));
    }
}
