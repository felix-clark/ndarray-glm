//! testing regularization
mod common;

use anyhow::Result;
use approx::assert_abs_diff_eq;
use common::{array_from_csv, y_x_from_iris};
use ndarray::{array, Array1, Array2};
use ndarray_glm::{utility::standardize, Linear, Logistic, ModelBuilder};

#[test]
/// Test that the intercept is not affected by regularization when the dependent
/// data is centered. This is only strictly true for linear regression.
fn same_lin_intercept() -> Result<()> {
    let y_data: Array1<f64> = array![0.3, 0.5, 0.8, 0.2];
    let x_data: Array2<f64> = array![[1.5, 0.6], [2.1, 0.8], [1.2, 0.7], [1.6, 0.3]];
    // standardize the data
    let x_data = standardize(x_data);

    let lin_model = ModelBuilder::<Linear>::data(&y_data, &x_data).build()?;
    let lin_fit = lin_model.fit()?;
    let lin_model_reg = ModelBuilder::<Linear>::data(&y_data, &x_data).build()?;
    // use a pretty large regularization term to make sure the effect is pronounced
    let lin_fit_reg = lin_model_reg.fit_options().l2_reg(1.0).fit()?;
    dbg!(&lin_fit.result);
    dbg!(&lin_fit_reg.result);
    // Ensure that the intercept terms are equal
    assert_abs_diff_eq!(
        lin_fit.result[0],
        lin_fit_reg.result[0],
        epsilon = 2.0 * f64::EPSILON
    );

    Ok(())
}

#[test]
/// Test the lasso regression on underconstrained data
fn lasso_underconstrained() -> Result<()> {
    let y_data: Array1<bool> = array![true, false, true];
    let x_data: Array2<f64> = array![[0.1, 1.5, 8.0], [-0.1, 1.0, -12.0], [0.2, 0.5, 9.5]];
    let model = ModelBuilder::<Logistic>::data(&y_data, &x_data).build()?;
    // The smoothing parameter needs to be relatively large in order to work
    let fit = model.fit_options().l1_reg(1.0).fit()?;
    dbg!(fit.result);
    let like = fit.model_like;
    // make sure the likelihood isn't NaN
    assert!(like.is_normal());
    Ok(())
}

#[test]
fn lasso_seperable() -> Result<()> {
    let (y_labels, x_data) = y_x_from_iris()?;
    let x_data = standardize(x_data);
    // let y_data: Array1<bool> = y_labels.mapv(|i| i == 0);
    // versicolor
    let y_data: Array1<bool> = y_labels.mapv(|i| i == 1);
    let target: Array1<f32> = array_from_csv("tests/R/log_regularization/iris_setosa_l1_1e-2.csv")?;
    dbg!(&target);
    let model = ModelBuilder::<Logistic>::data(&y_data, &x_data).build()?;
    // TODO: increasing the lambda seems to cause slow or failed convergence.
    // In particular, it finishes but fails to converge to the better value at lambda = 0.1
    // let fit = model.fit_options().l1_reg(0.1).max_iter(100).fit()?;
    let fit = model.fit_options().l1_reg(1e-2).fit()?;
    dbg!(&fit.result);
    // If this is negative then our alg hasn't converged to a good minimum
    dbg!(fit.lr_test_against(&target));
    assert!(fit.lr_test_against(&target) >= 0., "If it's not an exact match to the target, it should be a better result under our likelihood.");
    assert_abs_diff_eq!(&target, &fit.result, epsilon = 0.01);
    Ok(())
}

#[test]
fn ridge_seperable() -> Result<()> {
    let (y_labels, x_data) = y_x_from_iris()?;
    // let x_data = standardize(x_data);
    let y_data: Array1<bool> = y_labels.mapv(|i| i == 0);
    // versicolor
    // let y_data: Array1<bool> = y_labels.mapv(|i| i == 1);
    let target: Array1<f32> = array_from_csv("tests/R/log_regularization/iris_setosa_l2_1e-2.csv")?;
    let model = ModelBuilder::<Logistic>::data(&y_data, &x_data).build()?;
    // Temporarily try L2 for testing
    let fit = model.fit_options().l2_reg(1e-2).fit()?;
    // This still appears to be positive so our result is better
    dbg!(fit.lr_test_against(&target));
    // Ensure that our result is better, even if the parameters aren't epsilon-equivalent.
    assert!(fit.lr_test_against(&target) > -f32::EPSILON);
    // their result seems less precise, even when reducing the threshold.
    assert_abs_diff_eq!(&target, &fit.result, epsilon = 2e-3);
    Ok(())
}
