use anyhow::Result;

use approx::assert_abs_diff_eq;
use ndarray_glm::{Logistic, ModelBuilder};
mod common;
use common::{array_from_csv, y_x_off_from_csv};

#[test]
fn logistic_weights() -> Result<()> {
    let (y, x, wts) = y_x_off_from_csv::<bool, f32, 2>("tests/data/log_weights.csv")?;
    dbg!(&y);
    dbg!(&x);
    let model = ModelBuilder::<Logistic>::data(&y, &x).weight(wts).build()?;
    let fit = model.fit()?;

    let r_result = array_from_csv::<f32>("tests/R/log_weights/coefficients.csv")?;
    // TODO: try different epsilons
    assert_abs_diff_eq!(&fit.result, &r_result, epsilon=1e-5);
    assert!(fit.lr_test_against(&r_result) >= 0., "make sure our fit is at least as good as R's");

    // TODO: deviance, covariance?
    Ok(())
}
