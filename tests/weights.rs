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
