//! test cases for logistic regression

use anyhow::Result;

use approx::assert_abs_diff_eq;
use ndarray::Array;
use ndarray_glm::{Logistic, ModelBuilder};
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
    let r_cov = Array::from_shape_vec((n_par, n_par), r_flat_cov.into_raw_vec())?;
    assert_abs_diff_eq!(fit.covariance()?, r_cov, epsilon = 1e-5);

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
    // This can be terminated either by standardizing the data or by using
    // lambda = 2e-6 intead of 1e-6.
    // let x = ndarray_glm::standardize::standardize(x);
    let model = ModelBuilder::<Logistic>::data(&y, &x)
        .linear_offset(off)
        .build()?;
    let fit = model.fit_options().l2_reg(2e-6).max_iter(48).fit()?;
    dbg!(fit.result);
    dbg!(fit.n_iter);
    Ok(())
}
