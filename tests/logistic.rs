//! test cases for logistic regression

use anyhow::Result;

use approx::assert_abs_diff_eq;
use ndarray::Array;
use ndarray_glm::{Logistic, ModelBuilder, error::RegressionError};
mod common;
use common::{array_from_csv, y_x_off_from_csv};

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

    // dbg!(fit.score_test()?);
    // dbg!(fit.lr_test());
    // dbg!(fit.wald_test());

    Ok(())
}

#[test]
// this data caused an infinite loop with step halving
fn log_termination_1() -> Result<()> {
    let (y, x, off) = y_x_off_from_csv::<bool, f32, 1>("tests/data/log_termination_1.csv")?;
    let model = ModelBuilder::<Logistic>::data(&y, &x)
        .linear_offset(off)
        .build()?;
    let fit = model.fit()?;
    dbg!(fit.result);
    dbg!(fit.n_iter);
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
    let fit = match model.fit_options().l2_reg(1e-5).max_iter(48).fit() {
        Ok(fit) => fit,
        Err(err) => {
            if let RegressionError::MaxIter { n_iter: _, history } = &err {
                dbg!(&history);
            }
            return Err(err.into());
        }
    };
    dbg!(fit.result);
    dbg!(fit.n_iter);
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
