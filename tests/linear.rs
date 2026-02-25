use anyhow::Result;

use approx::assert_abs_diff_eq;
use ndarray::Array;
use ndarray_glm::{Linear, ModelBuilder};
mod common;
use common::{array_from_csv, load_linear_data};

// --- Linear model weight test suite ---

/// Load R reference data from a scenario directory and validate all quantities that are
/// comparable between R's glm() and our library. For scenarios without frequency weights,
/// all per-observation quantities can be compared. For freq-weighted scenarios, only
/// aggregates and response/working residuals are compared (leverage semantics differ).
/// NOTE: This method and testing scheme was generated with claude code.
fn check_linear_scenario(
    fit: &ndarray_glm::Fit<Linear, f64>,
    dir: &str,
    eps: f64,
    has_var_weights: bool,
    has_freq_weights: bool,
) -> Result<()> {
    let r = |name: &str| array_from_csv::<f64>(&format!("tests/R/linear_results/{dir}/{name}.csv"));

    let n_par = fit.result.len();
    let n_obs = 25usize; // original observations

    // --- Coefficients ---
    let r_coef = r("coefficients")?;
    assert_abs_diff_eq!(&fit.result, &r_coef, epsilon = eps);

    // --- Deviance ---
    let r_dev = r("deviance")?[0];
    assert_abs_diff_eq!(fit.deviance(), r_dev, epsilon = eps);

    // --- Null deviance (via lr_test) ---
    let r_null_dev = r("null_deviance")?[0];
    let our_null_dev = fit.lr_test() + r_dev;
    assert_abs_diff_eq!(our_null_dev, r_null_dev, epsilon = 10. * eps);

    // --- R² ---
    let r_r_sq = r("r_sq")?[0];
    assert_abs_diff_eq!(fit.r_sq(), r_r_sq, epsilon = 10. * eps);

    // --- Dispersion ---
    // Our library uses an effective sample size correction for variance weights:
    //   dispersion = deviance / ((1 - p/n_eff) * sum_weights)
    // R uses: dispersion = deviance / (n - p)
    // These match when there are no variance weights.
    if !has_var_weights {
        let r_disp = r("dispersion")?[0];
        assert_abs_diff_eq!(fit.dispersion(), r_disp, epsilon = eps);

        // Covariance and Wald Z depend on dispersion, so only check when dispersion matches
        let r_cov_flat = r("covariance")?;
        let r_cov = Array::from_shape_vec((n_par, n_par), r_cov_flat.into_raw_vec_and_offset().0)?;
        assert_abs_diff_eq!(fit.covariance()?, &r_cov, epsilon = eps);

        let r_wald_z = r("wald_z")?;
        assert_abs_diff_eq!(fit.wald_z()?, r_wald_z, epsilon = 10. * eps);
    }

    // --- AIC / BIC ---
    // For gaussian, R's AIC includes the full log-likelihood (with normalizing constant
    // -n/2 * log(2*pi*sigma^2)) and counts sigma^2 as a parameter. Our library uses
    // deviance + 2*p, which differs by a data-dependent additive constant.
    // Both are valid (AIC is only meaningful for differences), but values don't match.
    // We skip direct AIC/BIC comparison for the linear family.

    // --- Residuals: response ---
    let r_resid_resp = r("resid_resp")?;
    assert_abs_diff_eq!(fit.resid_resp(), r_resid_resp, epsilon = eps);

    // --- Residuals: working ---
    // For gaussian with identity link, working = response
    let r_resid_work = r("resid_work")?;
    assert_abs_diff_eq!(fit.resid_work(), r_resid_work, epsilon = eps);

    // --- Partial residuals ---
    // One column per predictor (intercept excluded), row-major in the CSV.
    let partial = fit.resid_part();
    let n_feat = partial.ncols();
    // Our library uses fully-weighted (variance × frequency) column means for centering, while R
    // uses only frequency weights. Results agree when there are no variance weights, or when there
    // is no intercept (no centering applied). Detect has_intercept by checking n_par vs n_feat.
    let has_intercept = n_feat + 1 == n_par;
    if !has_var_weights || !has_intercept {
        let r_partial_flat = r("resid_partial")?;
        let r_partial =
            Array::from_shape_vec((n_obs, n_feat), r_partial_flat.into_raw_vec_and_offset().0)?;
        assert_abs_diff_eq!(partial, r_partial, epsilon = eps);
    }

    if !has_freq_weights {
        // Per-observation quantities that depend on leverage (which has different semantics
        // under frequency weights due to row duplication in R vs. freq weighting in our library)

        // --- Pearson residuals ---
        let r_resid_pear = r("resid_pear")?;
        assert_abs_diff_eq!(*fit.resid_pear(), r_resid_pear, epsilon = eps);

        // --- Deviance residuals ---
        let r_resid_dev = r("resid_dev")?;
        assert_abs_diff_eq!(fit.resid_dev(), r_resid_dev, epsilon = eps);

        // --- Leverage ---
        let r_leverage = r("leverage")?;
        assert_abs_diff_eq!(fit.leverage()?, r_leverage, epsilon = eps);

        if !has_var_weights {
            // Standardized residuals depend on dispersion, which differs with var weights

            // --- Standardized Pearson residuals ---
            let r_resid_pear_std = r("resid_pear_std")?;
            assert_abs_diff_eq!(fit.resid_pear_std()?, r_resid_pear_std, epsilon = eps);

            // --- Standardized deviance residuals ---
            let r_resid_dev_std = r("resid_dev_std")?;
            assert_abs_diff_eq!(fit.resid_dev_std()?, r_resid_dev_std, epsilon = eps);

            // --- Studentized residuals ---
            // R's rstudent.glm() uses a different one-step approximation than ours.
            // Our method is exact for linear models (verified in fit.rs::residuals_linear
            // which compares against actual LOO refits). R's approximation can differ
            // significantly (>10% relative error on some observations), so we skip this
            // comparison here.
            let _r_resid_student = r("resid_student")?;
            let _our_resid_student = fit.resid_student()?;

            // --- Cook's distance ---
            let r_cooks = r("cooks")?;
            assert_abs_diff_eq!(fit.cooks()?, r_cooks, epsilon = eps);
        }

        // --- LOO influence coefficients ---
        let r_loo_flat = r("loo_coef")?;
        let r_loo = Array::from_shape_vec((n_obs, n_par), r_loo_flat.into_raw_vec_and_offset().0)?;
        // R's influence() returns the contribution to subtract from coefficients.
        // Our infl_coef() returns the same thing.
        let infl = fit.infl_coef()?;
        assert_abs_diff_eq!(infl, r_loo, epsilon = 10. * eps);

        // Also check exact LOO against the one-step approximation
        let loo_exact = fit.loo_exact()?;
        let infl_exact = &fit.result - &loo_exact;
        // For linear models, the one-step approximation should be exact
        assert_abs_diff_eq!(infl, infl_exact, epsilon = eps);
    }

    // --- P-values (stats feature) ---
    #[cfg(feature = "stats")]
    {
        // pvalue_lr_test: chi-squared omnibus test — valid under all weight configurations.
        let r_lr_p = r("pvalue_lr_test")?[0];
        assert_abs_diff_eq!(fit.pvalue_lr_test(), r_lr_p, epsilon = eps);

        // pvalue_wald and pvalue_exact both use the dispersion estimate in the denominator,
        // so they only match R when our dispersion agrees with R's (i.e. no variance weights).
        if !has_var_weights {
            let r_wald_p = r("pvalue_wald")?;
            assert_abs_diff_eq!(fit.pvalue_wald()?, r_wald_p, epsilon = eps);

            // pvalue_exact: drop-one F-test (intercept via anova, predictors via drop1 in R).
            let r_exact_p = r("pvalue_exact")?;
            assert_abs_diff_eq!(fit.pvalue_exact()?, r_exact_p, epsilon = 10. * eps);
        }
    }

    Ok(())
}

#[test]
fn linear_no_weights() -> Result<()> {
    let (y, x, _var_wt, _freq_wt) = load_linear_data()?;
    let model = ModelBuilder::<Linear>::data(&y, &x).build()?;
    let fit = model.fit()?;
    check_linear_scenario(&fit, "none", 1e-10, false, false)
}

#[test]
fn linear_var_weights() -> Result<()> {
    let (y, x, var_wt, _freq_wt) = load_linear_data()?;
    let model = ModelBuilder::<Linear>::data(&y, &x)
        .var_weights(var_wt)
        .build()?;
    let fit = model.fit()?;
    check_linear_scenario(&fit, "var", 1e-10, true, false)
}

#[test]
fn linear_freq_weights() -> Result<()> {
    let (y, x, _var_wt, freq_wt) = load_linear_data()?;
    let model = ModelBuilder::<Linear>::data(&y, &x)
        .freq_weights(freq_wt)
        .build()?;
    let fit = model.fit()?;
    check_linear_scenario(&fit, "freq", 1e-10, false, true)
}

#[test]
fn linear_both_weights() -> Result<()> {
    let (y, x, var_wt, freq_wt) = load_linear_data()?;
    let model = ModelBuilder::<Linear>::data(&y, &x)
        .var_weights(var_wt)
        .freq_weights(freq_wt)
        .build()?;
    let fit = model.fit()?;
    check_linear_scenario(&fit, "both", 1e-10, true, true)
}

// --- Linear model with offset and no intercept ---
// These test the null model code path where linear_offset is Some and use_intercept is false.

/// Compute the offset vector matching the R script: off = 0.3 * x1
fn offset_from_x(x: &ndarray::Array2<f64>) -> ndarray::Array1<f64> {
    x.column(0).mapv(|v| 0.3 * v)
}

#[test]
fn linear_offset_noint_no_weights() -> Result<()> {
    let (y, x, _var_wt, _freq_wt) = load_linear_data()?;
    let off = offset_from_x(&x);
    // Fit y ~ x2 + x3 - 1 with offset = 0.3*x1
    let x_sub = x.slice(ndarray::s![.., 1..]).to_owned();
    let model = ModelBuilder::<Linear>::data(&y, &x_sub)
        .linear_offset(off)
        .no_constant()
        .build()?;
    let fit = model.fit()?;
    check_linear_scenario(&fit, "off_none", 1e-10, false, false)
}

#[test]
fn linear_offset_noint_var_weights() -> Result<()> {
    let (y, x, var_wt, _freq_wt) = load_linear_data()?;
    let off = offset_from_x(&x);
    let x_sub = x.slice(ndarray::s![.., 1..]).to_owned();
    let model = ModelBuilder::<Linear>::data(&y, &x_sub)
        .linear_offset(off)
        .no_constant()
        .var_weights(var_wt)
        .build()?;
    let fit = model.fit()?;
    check_linear_scenario(&fit, "off_var", 1e-10, true, false)
}

#[test]
fn linear_offset_noint_freq_weights() -> Result<()> {
    let (y, x, _var_wt, freq_wt) = load_linear_data()?;
    let off = offset_from_x(&x);
    let x_sub = x.slice(ndarray::s![.., 1..]).to_owned();
    let model = ModelBuilder::<Linear>::data(&y, &x_sub)
        .linear_offset(off)
        .no_constant()
        .freq_weights(freq_wt)
        .build()?;
    let fit = model.fit()?;
    check_linear_scenario(&fit, "off_freq", 1e-10, false, true)
}

#[test]
fn linear_offset_noint_both_weights() -> Result<()> {
    let (y, x, var_wt, freq_wt) = load_linear_data()?;
    let off = offset_from_x(&x);
    let x_sub = x.slice(ndarray::s![.., 1..]).to_owned();
    let model = ModelBuilder::<Linear>::data(&y, &x_sub)
        .linear_offset(off)
        .no_constant()
        .var_weights(var_wt)
        .freq_weights(freq_wt)
        .build()?;
    let fit = model.fit()?;
    check_linear_scenario(&fit, "off_both", 1e-10, true, true)
}

#[test]
fn counting_weights() -> Result<()> {
    // This actually gets at the distinction between variance weights and frequency weights.
    // The parameter results should be the same, but the covariance and other statistics will
    // distinguish.

    use ndarray::array;
    let x_duped = array![
        [-4.3, 0.2],
        [2.3, 0.4],
        [2.3, 0.4],
        [-1.2, -0.1],
        [2.3, 0.4],
        [-4.3, 0.2],
        [0.5, -0.5],
    ];
    let y_duped = array![0.5, 1.2, 1.2, 0.3, 1.2, 0.5, 0.8];
    let x_red = array![[-4.3, 0.2], [2.3, 0.4], [-1.2, -0.1], [0.5, -0.5]];
    let y_red = array![0.5, 1.2, 0.3, 0.8];
    let freqs_red = array![2, 3, 1, 1];

    let model_duped = ModelBuilder::<Linear>::data(&y_duped, &x_duped).build()?;
    let fit_duped = model_duped.fit()?;

    let model_red = ModelBuilder::<Linear>::data(&y_red, &x_red)
        .freq_weights(freqs_red)
        .build()?;
    let fit_red = model_red.fit()?;

    // let eps = f32::EPSILON as f64;
    let eps = 64.0 * f64::EPSILON;

    assert_abs_diff_eq!(fit_duped.result, fit_red.result, epsilon = eps);
    assert_abs_diff_eq!(
        fit_duped.covariance()?,
        fit_red.covariance()?,
        epsilon = eps
    );

    Ok(())
}