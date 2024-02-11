use anyhow::Result;

use approx::assert_abs_diff_eq;
use ndarray::Array;
use ndarray_glm::{Logistic, ModelBuilder};
mod common;
use common::{array_from_csv, y_x_off_from_csv};

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
    let r_cov = Array::from_shape_vec((n_par, n_par), r_flat_cov.into_raw_vec())?;
    assert_abs_diff_eq!(fit.covariance()?, r_cov, epsilon = eps);

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
    assert_abs_diff_eq!(fit.resid_pear(), r_resid_pear, epsilon = eps);

    // Again we have a discrepancy between frequency and variance weights
    let r_resid_dev = array_from_csv::<f32>("tests/R/log_weights/dev_resid.csv")?;
    assert_abs_diff_eq!(fit.resid_dev(), r_resid_dev, epsilon = eps);

    // studentized residuals use the leverage, which depends on the weights
    let r_stand_resid_pear =
        array_from_csv::<f32>("tests/R/log_weights/standard_pearson_resid.csv")?;
    let r_stand_resid_dev =
        array_from_csv::<f32>("tests/R/log_weights/standard_deviance_resid.csv")?;
    let r_stud_resid = array_from_csv::<f32>("tests/R/log_weights/student_resid.csv")?;
    assert_abs_diff_eq!(fit.resid_pear_std()?, r_stand_resid_pear, epsilon = eps);
    assert_abs_diff_eq!(fit.resid_dev_std()?, r_stand_resid_dev, epsilon = eps);
    assert_abs_diff_eq!(fit.resid_student()?, r_stud_resid, epsilon = eps);

    Ok(())
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
