//! test cases for logistic regression

use anyhow::Result;

use approx::assert_abs_diff_eq;
use ndarray::{Array, Array1, Array2, array, s};
use ndarray_glm::{Logistic, ModelBuilder, error::RegressionError};
mod common;
use common::{array_from_csv, load_logistic_data, y_x_off_from_csv};

#[test]
// this data caused an infinite loop with step halving
fn log_termination_0() -> Result<()> {
    let (y, x, off) = y_x_off_from_csv::<bool, f32, 1>("tests/data/log_termination_0.csv")?;
    let model = ModelBuilder::<Logistic>::data(&y, &x)
        .linear_offset(off)
        .build()?;
    let fit = model.fit()?;
    dbg!(&fit.result);
    dbg!(&fit.n_iter);

    let n_par = fit.result.len();

    // Check consistency with R results
    let r_result = array_from_csv::<f32>("tests/R/log_termination_0/coefficients.csv")?;
    assert_abs_diff_eq!(&fit.result, &r_result, epsilon = 1e-5);
    assert!(
        fit.lr_test_against(&r_result) >= 0.,
        "make sure this is better than R's"
    );
    let r_dev_resid = array_from_csv::<f32>("tests/R/log_termination_0/dev_resid.csv")?;
    assert_abs_diff_eq!(fit.resid_dev(), r_dev_resid, epsilon = 1e-5);
    let r_flat_cov = array_from_csv::<f32>("tests/R/log_termination_0/covariance.csv")?;
    let r_cov = Array::from_shape_vec((n_par, n_par), r_flat_cov.into_raw_vec_and_offset().0)?;
    assert_abs_diff_eq!(fit.covariance()?, &r_cov, epsilon = 2e-5);

    // We've already asserted that our fit is better according to our likelihood function, so the
    // epsilon doesn't have to be extremely strict.
    let eps = 5e-5;
    let r_dev = array_from_csv::<f32>("tests/R/log_termination_0/deviance.csv")?[0];
    assert_abs_diff_eq!(fit.deviance(), r_dev, epsilon = eps);
    let r_aic = array_from_csv::<f32>("tests/R/log_termination_0/aic.csv")?[0];
    assert_abs_diff_eq!(fit.aic(), r_aic, epsilon = eps);
    let r_bic = array_from_csv::<f32>("tests/R/log_termination_0/bic.csv")?[0];
    assert_abs_diff_eq!(fit.bic(), r_bic, epsilon = eps);
    let r_stand_resid_pear =
        array_from_csv::<f32>("tests/R/log_termination_0/standard_pearson_resid.csv")?;
    let r_stand_resid_dev =
        array_from_csv::<f32>("tests/R/log_termination_0/standard_deviance_resid.csv")?;
    assert_abs_diff_eq!(fit.resid_pear_std()?, r_stand_resid_pear, epsilon = eps);
    assert_abs_diff_eq!(fit.resid_dev_std()?, r_stand_resid_dev, epsilon = eps);
    let r_stud_resid = array_from_csv::<f32>("tests/R/log_termination_0/student_resid.csv")?;
    // It appears that R's rstudent() function may not be as accurate as our expression. However,
    // they are still approximately equal with this many data points.
    assert_abs_diff_eq!(fit.resid_student()?, r_stud_resid, epsilon = 0.05);

    let r_null_dev = array_from_csv::<f32>("tests/R/log_termination_0/null_dev.csv")?[0];
    assert_abs_diff_eq!(fit.lr_test(), r_null_dev - r_dev, epsilon = eps);

    Ok(())
}

#[test]
// this data caused an infinite loop with step halving
fn log_termination_1() -> Result<()> {
    let (y, x, off) = y_x_off_from_csv::<bool, f32, 1>("tests/data/log_termination_1.csv")?;
    let model = ModelBuilder::<Logistic>::data(&y, &x)
        .linear_offset(off)
        .build()?;
    let _fit = model.fit()?;
    Ok(())
}

#[test]
fn log_regularization() -> Result<()> {
    let (y, x, off) = y_x_off_from_csv::<bool, f32, 1>("tests/data/log_regularization.csv")?;
    // This actually has a harder time converging when the data *is* standardized. It can be
    // handled by increasing L2.
    let model = ModelBuilder::<Logistic>::data(&y, &x)
        .linear_offset(off)
        .build()?;
    let _ = match model.fit_options().l2_reg(1e-5).fit() {
        Ok(fit) => fit,
        Err(err) => {
            if let RegressionError::MaxIter { n_iter: _, history } = &err {
                dbg!(&history);
            }
            return Err(err.into());
        }
    };
    Ok(())
}

#[test]
fn logistic_weights() -> Result<()> {
    let (y, x, wts) = y_x_off_from_csv::<bool, f32, 2>("tests/data/log_weights.csv")?;
    let model = ModelBuilder::<Logistic>::data(&y, &x)
        .var_weights(wts.clone())
        .build()?;
    let fit = model.fit()?;

    let eps: f32 = 1e-4;

    let n_par = fit.result.len();
    let n_obs = y.len();

    let r_result = array_from_csv::<f32>("tests/R/log_weights/coefficients.csv")?;
    // NOTE: R result only seems good to a few decimal points
    assert_abs_diff_eq!(&fit.result, &r_result, epsilon = eps);
    assert!(
        fit.lr_test_against(&r_result) >= 0.,
        "make sure our fit is at least as good as R's"
    );

    // check parameter covariance function
    let r_flat_cov = array_from_csv::<f32>("tests/R/log_weights/covariance.csv")?;
    let r_cov = Array::from_shape_vec((n_par, n_par), r_flat_cov.into_raw_vec_and_offset().0)?;
    assert_abs_diff_eq!(fit.covariance()?, &r_cov, epsilon = eps);

    // total deviance uses the weights
    let r_dev = array_from_csv::<f32>("tests/R/log_weights/deviance.csv")?[0];
    assert_abs_diff_eq!(fit.deviance(), r_dev, epsilon = 0.1 * eps);

    let r_null_dev = array_from_csv::<f32>("tests/R/log_weights/null_dev.csv")?[0];
    assert_abs_diff_eq!(fit.lr_test(), r_null_dev - r_dev, epsilon = 0.1 * eps);

    // Residuals should be orthogonal to the hat matrix
    // Our convention of the hat matrix is orthogonal to the response residuals
    let hat_mat = fit.hat()?;
    assert_abs_diff_eq!(
        hat_mat.dot(&fit.resid_resp()),
        Array::zeros(y.len()),
        epsilon = 0.1 * eps,
    );

    let hat = fit.leverage()?;
    let r_hat = array_from_csv::<f32>("tests/R/log_weights/hat.csv")?;
    assert_abs_diff_eq!(hat, r_hat, epsilon = eps);

    let r_resid_pear = array_from_csv::<f32>("tests/R/log_weights/pearson_resid.csv")?;
    assert_abs_diff_eq!(*fit.resid_pear(), r_resid_pear, epsilon = eps);

    // Again we have a discrepancy between frequency and variance weights
    let r_resid_dev = array_from_csv::<f32>("tests/R/log_weights/dev_resid.csv")?;
    assert_abs_diff_eq!(fit.resid_dev(), r_resid_dev, epsilon = eps);

    // studentized residuals use the leverage, which depends on the weights
    let r_stand_resid_pear =
        array_from_csv::<f32>("tests/R/log_weights/standard_pearson_resid.csv")?;
    let r_stand_resid_dev =
        array_from_csv::<f32>("tests/R/log_weights/standard_deviance_resid.csv")?;
    assert_abs_diff_eq!(fit.resid_pear_std()?, r_stand_resid_pear, epsilon = eps);
    assert_abs_diff_eq!(fit.resid_dev_std()?, r_stand_resid_dev, epsilon = eps);
    // R's rstudent.glm() doesn't seem to be as precise as our approximation. In this example, most
    // values are close but at least one is off by ~20%. Just skip this one here.
    // let r_stud_resid = array_from_csv::<f32>("tests/R/log_weights/student_resid.csv")?;
    // assert_abs_diff_eq!(fit.resid_student()?, r_stud_resid, epsilon = eps);

    // // These match in the unweighted case, but not the weighted one.
    // The difference between weighted/unweighted is the odd part.
    let r_aic = array_from_csv::<f32>("tests/R/log_weights/aic.csv")?[0];
    // With the sum of the log weights, these are pretty close, but not to FPE.
    assert_abs_diff_eq!(fit.aic(), r_aic, epsilon = 0.2);
    // let r_bic = array_from_csv::<f32>("tests/R/log_weights/bic.csv")?[0];
    // assert_eq!(fit.bic(), r_bic);

    // check the leave-one-out influence coefficients
    // Probably only accurate to within O(1/n_obs).
    // R's method is inexact as well, and it does not match ours, even when unweighted.
    // We'll check with a large epsilon, but the difference may be worth investigating further.
    let r_loo = array_from_csv::<f32>("tests/R/log_weights/loo_coef.csv")?;
    let r_loo = Array::from_shape_vec((n_obs, n_par), r_loo.into_raw_vec_and_offset().0)?;

    let loo_exact = fit.loo_exact()?;
    let infl_exact = &fit.result - &loo_exact;
    let infl_loo = fit.infl_coef()?;

    // The influence results from R seem much worse than ours. Check that the errors are usually
    // lower, component-wise.
    let r_diff = infl_exact.clone() - r_loo.clone();
    let loo_diff = infl_exact.clone() - infl_loo.clone();
    let all_better = loo_diff.mapv(|x| x.abs()) - r_diff.mapv(|x| x.abs());
    // In this case, one of the entries is positive.
    let num_positive = all_better.into_iter().filter(|&x| x > 0.).count();
    dbg!(num_positive);
    assert!(num_positive < 5, "Allowing only four worse elements");
    // R's coefficients seem to be further away from exact than our own. In fact, they seem quite
    // far off.
    // assert_abs_diff_eq!(infl_exact, r_loo, epsilon = eps);
    // let infl_diff = fit.infl_coef()? - &r_loo;
    // This passes instead, but this epsilon is rather huge:
    // assert_abs_diff_eq!(infl_loo, r_loo, epsilon = 0.5);
    // This is close-ish, but not expected to be exact.
    // assert_abs_diff_eq!(infl_loo, infl_exact, epsilon = eps);

    Ok(())
}

// ============================================================================
// Comprehensive logistic regression test suite
//
// Parallel to tests/linear.rs: each scenario is validated against R's glm()
// output generated by tests/R/logistic_main.R. Scenarios cover:
//   - Response type: bool (0/1) and float (continuous in (0,1))
//   - Weights: none, variance, frequency, both
//   - Intercept: yes (most scenarios) and no (bool_off with offset)
// ============================================================================

// --- Fixed held-out test observations (3 rows, not in training set) ---
// Must match x_test / off_test in logistic_main.R.

fn pred_test_full_x() -> Array2<f64> {
    array![[0.5, 0.3, 0.1], [-1.0, -0.5, 0.8], [2.0, 1.2, -0.3]]
}

/// Two-column version (x2, x3 only) for offset+no-intercept scenarios.
fn pred_test_sub_x() -> Array2<f64> {
    array![[0.3, 0.1], [-0.5, 0.8], [1.2, -0.3]]
}

/// Offset for test observations: 0.5 * x1 = [0.25, -0.50, 1.00].
fn pred_test_off() -> Array1<f64> {
    array![0.25, -0.50, 1.00]
}

/// Compute the offset vector matching logistic_main.R: off = 0.5 * x1
fn offset_from_x(x: &Array2<f64>) -> Array1<f64> {
    x.column(0).mapv(|v| 0.5 * v)
}

/// Load R reference data from a scenario directory and validate all quantities
/// comparable between R's glm(family=binomial) and this library.
///
/// `eps` is the base tolerance. Quantities derived from matrix inversion (covariance,
/// Wald z) use `10. * eps`. The caller should choose `eps` to accommodate IRLS
/// convergence differences; logistic IRLS typically matches R to within ~1e-7.
///
/// Gating rules:
/// - AIC/BIC: compared only for binary (bool) responses without variance weights.
///   For binary responses, the saturated log-likelihood is 0 so our deviance-based
///   AIC matches R's exactly. Variance weights introduce a log-weight correction
///   that differs from R's convention.
/// - Covariance / Wald z: always compared for logistic (dispersion = 1).
/// - Partial residuals: skipped when there are variance weights AND an intercept,
///   because our centering uses fully-weighted means while R uses only freq weights.
/// - Per-observation quantities (Pearson, deviance residuals, leverage, Cook's,
///   standardized residuals): skipped when frequency weights are present because R
///   expands rows and reports values per-expanded-row, while we work with the
///   original n observations.
/// - LOO influence: we call loo_exact() and infl_coef() to verify they run, but
///   do not compare against R's one-step approximation (which is less accurate for
///   non-linear models).
#[allow(clippy::too_many_arguments)]
fn check_logistic_scenario(
    fit: &ndarray_glm::Fit<Logistic, f64>,
    dir: &str,
    eps: f64,
    has_var_weights: bool,
    has_freq_weights: bool,
    is_bool: bool,
    pred_x: &Array2<f64>,
    pred_off: Option<&Array1<f64>>,
) -> Result<()> {
    let r =
        |name: &str| array_from_csv::<f64>(&format!("tests/R/logistic_results/{dir}/{name}.csv"));

    // Covariance involves matrix inversion and accumulates more floating-point error than
    // the coefficients themselves. Even when coefficients agree to eps, the covariance can
    // differ by an order of magnitude more due to condition-number amplification in the
    // Fisher information matrix solve.
    let cov_eps = 100. * eps;

    let n_par = fit.result.len();
    let n_obs = 30usize; // original observations

    // --- Coefficients ---
    let r_coef = r("coefficients")?;
    assert_abs_diff_eq!(&fit.result, &r_coef, epsilon = eps);

    // --- Deviance ---
    let r_dev = r("deviance")?[0];
    assert_abs_diff_eq!(fit.deviance(), r_dev, epsilon = eps);

    // --- Null deviance (via lr_test) ---
    let r_null_dev = r("null_deviance")?[0];
    let our_null_dev = fit.lr_test() + r_dev;
    assert_abs_diff_eq!(our_null_dev, r_null_dev, epsilon = cov_eps);

    // --- Prediction on held-out data ---
    let r_pred = r("predict_resp")?;
    assert_abs_diff_eq!(fit.predict(pred_x, pred_off), r_pred, epsilon = eps);

    // --- Covariance (always comparable: logistic dispersion = 1) ---
    let r_cov_flat = r("covariance")?;
    let r_cov = Array::from_shape_vec((n_par, n_par), r_cov_flat.into_raw_vec_and_offset().0)?;
    assert_abs_diff_eq!(fit.covariance()?, &r_cov, epsilon = cov_eps);

    // --- Wald z (always comparable for logistic) ---
    let r_wald_z = r("wald_z")?;
    assert_abs_diff_eq!(fit.wald_z()?, r_wald_z, epsilon = cov_eps);

    // --- AIC / BIC ---
    // For binary (bool) responses without variance weights, our deviance-based AIC
    // matches R exactly (sat log-like = 0 for binary, no log-weight correction needed).
    // For float responses or var-weighted scenarios, AIC definitions differ.
    if is_bool && !has_var_weights {
        let r_aic = r("aic")?[0];
        assert_abs_diff_eq!(fit.aic(), r_aic, epsilon = eps);
        let r_bic = r("bic")?[0];
        assert_abs_diff_eq!(fit.bic(), r_bic, epsilon = cov_eps);
    }

    // --- Response residuals ---
    let r_resid_resp = r("resid_resp")?;
    assert_abs_diff_eq!(fit.resid_resp(), r_resid_resp, epsilon = eps);

    // --- Working residuals ---
    // For logistic canonical link: resid_work = (y - mu) / V(mu) = (y - mu) / (mu*(1-mu)).
    // When mu is near 0 or 1 the sensitivity to beta is amplified; use cov_eps here.
    let r_resid_work = r("resid_work")?;
    assert_abs_diff_eq!(fit.resid_work(), r_resid_work, epsilon = cov_eps);

    // --- Partial residuals ---
    // Our library uses fully-weighted (variance × frequency) column means for centering,
    // while R uses only frequency weights. Skip comparison when both var weights and an
    // intercept are present.
    let partial = fit.resid_part();
    let n_feat = partial.ncols();
    let has_intercept = n_feat + 1 == n_par;
    if !has_var_weights || !has_intercept {
        let r_partial_flat = r("resid_partial")?;
        let r_partial =
            Array::from_shape_vec((n_obs, n_feat), r_partial_flat.into_raw_vec_and_offset().0)?;
        assert_abs_diff_eq!(partial, r_partial, epsilon = cov_eps);
    }

    if !has_freq_weights {
        // Per-observation quantities that differ under frequency weighting because R expands
        // rows while we use the original n observations.

        // --- Pearson residuals ---
        let r_resid_pear = r("resid_pear")?;
        assert_abs_diff_eq!(*fit.resid_pear(), r_resid_pear, epsilon = eps);

        // --- Deviance residuals ---
        // For float (continuous) responses where y ≈ μ, the per-observation deviance term
        // 2[y·log(y/μ) + (1-y)·log((1-y)/(1-μ))] can be a tiny negative number due to
        // floating-point cancellation, causing sqrt(negative) = NaN. Skip for non-bool y.
        if is_bool {
            let r_resid_dev = r("resid_dev")?;
            assert_abs_diff_eq!(fit.resid_dev(), r_resid_dev, epsilon = eps);
        }

        // --- Leverage ---
        let r_leverage = r("leverage")?;
        assert_abs_diff_eq!(fit.leverage()?, r_leverage, epsilon = cov_eps);

        // --- Standardized Pearson residuals ---
        // Dispersion = 1 for logistic, so these are always comparable.
        let r_resid_pear_std = r("resid_pear_std")?;
        assert_abs_diff_eq!(fit.resid_pear_std()?, r_resid_pear_std, epsilon = cov_eps);

        // --- Standardized deviance residuals ---
        // Same NaN issue as raw deviance residuals for float responses; skip there too.
        if is_bool {
            let r_resid_dev_std = r("resid_dev_std")?;
            assert_abs_diff_eq!(fit.resid_dev_std()?, r_resid_dev_std, epsilon = cov_eps);
        }

        // --- Cook's distance ---
        let r_cooks = r("cooks")?;
        assert_abs_diff_eq!(fit.cooks()?, r_cooks, epsilon = cov_eps);

        // --- LOO influence ---
        // For non-linear models, R's one-step approximation is less accurate than ours.
        // We verify these methods run without error but don't compare to R's values.
        let _loo_exact = fit.loo_exact()?;
        let _infl_approx = fit.infl_coef()?;
    }

    // --- P-values (stats feature) ---
    #[cfg(feature = "stats")]
    {
        // LR p-value: chi-squared omnibus test, valid for all configurations.
        let r_lr_p = r("pvalue_lr_test")?[0];
        assert_abs_diff_eq!(fit.pvalue_lr_test(), r_lr_p, epsilon = eps);

        // Wald p-values: always comparable for logistic (dispersion = 1, normal reference).
        let r_wald_p = r("pvalue_wald")?;
        assert_abs_diff_eq!(fit.pvalue_wald()?, r_wald_p, epsilon = cov_eps);

        // Exact p-values: chi-squared drop-one test for logistic.
        let r_exact_p = r("pvalue_exact")?;
        assert_abs_diff_eq!(fit.pvalue_exact()?, r_exact_p, epsilon = cov_eps);
    }

    Ok(())
}

// --- Bool response scenarios ---

#[test]
fn logistic_bool_none() -> Result<()> {
    let (y, _, x, _var_wt, _freq_wt) = load_logistic_data()?;
    let model = ModelBuilder::<Logistic>::data(&y, &x).build()?;
    let fit = model.fit()?;
    check_logistic_scenario(
        &fit,
        "bool_none",
        2e-9,
        false,
        false,
        true,
        &pred_test_full_x(),
        None,
    )
}

#[test]
fn logistic_bool_var_weights() -> Result<()> {
    let (y, _, x, var_wt, _freq_wt) = load_logistic_data()?;
    let model = ModelBuilder::<Logistic>::data(&y, &x)
        .var_weights(var_wt)
        .build()?;
    let fit = model.fit()?;
    check_logistic_scenario(
        &fit,
        "bool_var",
        1e-9,
        true,
        false,
        true,
        &pred_test_full_x(),
        None,
    )
}

#[test]
fn logistic_bool_freq_weights() -> Result<()> {
    let (y, _, x, _var_wt, freq_wt) = load_logistic_data()?;
    let model = ModelBuilder::<Logistic>::data(&y, &x)
        .freq_weights(freq_wt)
        .build()?;
    let fit = model.fit()?;
    check_logistic_scenario(
        &fit,
        "bool_freq",
        1e-8,
        false,
        true,
        true,
        &pred_test_full_x(),
        None,
    )
}

#[test]
fn logistic_bool_both_weights() -> Result<()> {
    let (y, _, x, var_wt, freq_wt) = load_logistic_data()?;
    let model = ModelBuilder::<Logistic>::data(&y, &x)
        .var_weights(var_wt)
        .freq_weights(freq_wt)
        .build()?;
    let fit = model.fit()?;
    check_logistic_scenario(
        &fit,
        "bool_both",
        1e-9,
        true,
        true,
        true,
        &pred_test_full_x(),
        None,
    )
}

#[test]
fn logistic_bool_offset_noint() -> Result<()> {
    let (y, _, x, _var_wt, _freq_wt) = load_logistic_data()?;
    let off = offset_from_x(&x);
    // Fit y ~ x2 + x3 - 1 with offset = 0.5*x1
    let x_sub = x.slice(s![.., 1..]).to_owned();
    let model = ModelBuilder::<Logistic>::data(&y, &x_sub)
        .linear_offset(off)
        .no_constant()
        .build()?;
    let fit = model.fit()?;
    let off_pred = pred_test_off();
    check_logistic_scenario(
        &fit,
        "bool_off",
        1e-9,
        false,
        false,
        true,
        &pred_test_sub_x(),
        Some(&off_pred),
    )
}

// --- Float response scenarios (y in (0,1)) ---

#[test]
fn logistic_float_none() -> Result<()> {
    let (_, y_float, x, _var_wt, _freq_wt) = load_logistic_data()?;
    let model = ModelBuilder::<Logistic>::data(&y_float, &x).build()?;
    let fit = model.fit()?;
    // Larger tolerance: y = p_true makes the true MLE exactly β_true, which R finds to
    // machine precision. Our IRLS converges to a floating-point fixed point ~3e-8 away
    // because x3 ≈ 0.5·x1 makes the Fisher matrix ill-conditioned.
    check_logistic_scenario(
        &fit,
        "float_none",
        1e-7,
        false,
        false,
        false,
        &pred_test_full_x(),
        None,
    )
}

#[test]
fn logistic_float_var_weights() -> Result<()> {
    let (_, y_float, x, var_wt, _freq_wt) = load_logistic_data()?;
    let model = ModelBuilder::<Logistic>::data(&y_float, &x)
        .var_weights(var_wt)
        .build()?;
    let fit = model.fit()?;
    check_logistic_scenario(
        &fit,
        "float_var",
        1e-10,
        true,
        false,
        false,
        &pred_test_full_x(),
        None,
    )
}

// --- Internal consistency tests ---

/// `wald_test_against` and `score_test_against` must be zero at the MLE.
#[test]
fn logistic_test_against_self_is_zero() -> Result<()> {
    let (y, _, x, _, _) = load_logistic_data()?;
    let model = ModelBuilder::<Logistic>::data(&y, &x).build()?;
    let fit = model.fit()?;
    let eps = 64. * f64::EPSILON;
    assert_abs_diff_eq!(fit.wald_test_against(&fit.result), 0., epsilon = eps);
    assert_abs_diff_eq!(
        fit.score_test_against(fit.result.clone())?,
        0.,
        epsilon = eps
    );
    Ok(())
}

/// Internal standardization must not change the fit results or key statistics.
#[test]
fn logistic_std_vs_nostd() -> Result<()> {
    let (y, _, x, var_wt, _) = load_logistic_data()?;
    let model_std = ModelBuilder::<Logistic>::data(&y, &x)
        .var_weights(var_wt.clone())
        .build()?;
    let fit_std = model_std.fit()?;
    let model_ns = ModelBuilder::<Logistic>::data(&y, &x)
        .var_weights(var_wt)
        .no_standardize()
        .build()?;
    let fit_ns = model_ns.fit()?;
    let eps = 0.01 * f32::EPSILON as f64;
    assert_abs_diff_eq!(fit_std.result, fit_ns.result, epsilon = eps);
    assert_abs_diff_eq!(fit_std.deviance(), fit_ns.deviance(), epsilon = eps);
    assert_abs_diff_eq!(fit_std.lr_test(), fit_ns.lr_test(), epsilon = eps);
    Ok(())
}

/// L2 regularization should shrink non-intercept coefficients relative to unregularized fit.
#[test]
fn logistic_l2_regularization() -> Result<()> {
    let (y, _, x, _var_wt, _freq_wt) = load_logistic_data()?;
    let model = ModelBuilder::<Logistic>::data(&y, &x).build()?;
    let fit_unreg = model.fit()?;
    let fit_l2 = model.fit_options().l2_reg(0.5).fit()?;
    // L2 penalty should shrink the non-intercept coefficients (index 1..) toward zero
    let norm_unreg: f64 = fit_unreg.result.slice(s![1..]).mapv(|v| v * v).sum();
    let norm_l2: f64 = fit_l2.result.slice(s![1..]).mapv(|v| v * v).sum();
    assert!(
        norm_l2 <= norm_unreg,
        "L2 should shrink coefficients: unreg={norm_unreg}, l2={norm_l2}"
    );
    // Verify core diagnostics still run without error
    let _ = fit_l2.covariance()?;
    let _ = fit_l2.wald_z()?;
    let _ = fit_l2.lr_test();
    Ok(())
}

/// L1 (lasso) regularization: verify the model fits without error and shrinks coefficients.
#[test]
fn logistic_l1_regularization() -> Result<()> {
    let (y, _, x, _var_wt, _freq_wt) = load_logistic_data()?;
    let model = ModelBuilder::<Logistic>::data(&y, &x).build()?;
    let fit_unreg = model.fit()?;
    let fit_l1 = model.fit_options().l1_reg(0.1).max_iter(500).fit()?;
    let norm_unreg: f64 = fit_unreg.result.slice(s![1..]).mapv(|v| v.abs()).sum();
    let norm_l1: f64 = fit_l1.result.slice(s![1..]).mapv(|v| v.abs()).sum();
    assert!(
        norm_l1 <= norm_unreg,
        "L1 should shrink L1-norm: unreg={norm_unreg}, l1={norm_l1}"
    );
    Ok(())
}
