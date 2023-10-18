use anyhow::Result;

use approx::assert_abs_diff_eq;
use ndarray::Array;
use ndarray_glm::{Logistic, ModelBuilder};
mod common;
use common::{array_from_csv, y_x_off_from_csv};

#[test]
fn logistic_weights() -> Result<()> {
    let (y, x, wts) = y_x_off_from_csv::<bool, f32, 2>("tests/data/log_weights.csv")?;
    let model = ModelBuilder::<Logistic>::data(&y, &x).weight(wts).build()?;
    let fit = model.fit()?;

    let r_result = array_from_csv::<f32>("tests/R/log_weights/coefficients.csv")?;
    // NOTE: R result only seems good to a few decimal points
    assert_abs_diff_eq!(&fit.result, &r_result, epsilon = 1e-4);
    assert!(
        fit.lr_test_against(&r_result) >= 0.,
        "make sure our fit is at least as good as R's"
    );

    let n_par = fit.result.len();

    // check parameter covariance function
    let r_flat_cov = array_from_csv::<f32>("tests/R/log_weights/covariance.csv")?;
    let r_cov = Array::from_shape_vec((n_par, n_par), r_flat_cov.into_raw_vec())?;
    assert_abs_diff_eq!(*fit.covariance()?, r_cov, epsilon = 1e-4);

    // total deviance uses the weights
    let r_dev = array_from_csv::<f32>("tests/R/log_weights/deviance.csv")?[0];
    assert_abs_diff_eq!(fit.deviance(), r_dev, epsilon = 1e-5);

    let r_null_dev = array_from_csv::<f32>("tests/R/log_weights/null_dev.csv")?[0];
    // assert_abs_diff_eq!(fit.null_like(), r_null_dev, epsilon = eps); // these are probably not
    // the same
    // likelihood ratio test should be positive
    assert_abs_diff_eq!(fit.lr_test(), r_null_dev - r_dev, epsilon = 1e-5);

    Ok(())
}
