//! testing regularization
mod common;

use anyhow::Result;
use approx::assert_abs_diff_eq;
use common::{array_from_csv, load_linear_data, y_x_from_iris};
use ndarray::{Array1, Array2, Axis, array};
use ndarray_glm::{Linear, Logistic, ModelBuilder};

#[test]
/// Test that the intercept is not affected by regularization when the dependent
/// data is centered. This is only strictly true for linear regression.
fn same_lin_intercept() -> Result<()> {
    let y_data: Array1<f64> = array![0.3, 0.5, 0.8, 0.2];
    let x_data: Array2<f64> = array![[1.5, 0.6], [2.1, 0.8], [1.2, 0.7], [1.6, 0.3]];
    // Explicitly center the data, but don't scale it.
    let x_data = x_data.clone() - x_data.mean_axis(Axis(0)).unwrap();
    // Since we are explicitly centering the x-data, this should hold with or without
    // internal standardization.
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
    // Either standardization or 32-bit floats is needed to converge.
    // The data is now standardized by default so this should still work internally.
    let model = ModelBuilder::<Logistic>::data(&y_data, &x_data).build()?;
    // The smoothing parameter needs to be relatively large in order to test
    let fit = model.fit_options().max_iter(256).l1_reg(1.0).fit()?;
    let like: f64 = fit.model_like;
    // make sure the likelihood isn't NaN
    assert!(like.is_normal());
    Ok(())
}

#[test]
fn elnet_seperable() -> Result<()> {
    let (y_labels, x_data) = y_x_from_iris()?;
    // setosa
    let y_data: Array1<bool> = y_labels.mapv(|i| i == 0);
    let target: Array1<f32> =
        array_from_csv("tests/R/log_regularization/iris_setosa_l1_l2_1e-2.csv")?;
    let model = ModelBuilder::<Logistic>::data(&y_data, &x_data).build()?;
    // The X-data is now standardized by default.
    let fit = model.fit_options().l1_reg(1e-2).l2_reg(1e-2).fit()?;

    // If this is negative then our alg hasn't converged to a good minimum
    assert!(
        fit.lr_test_against(&target) >= 0.,
        "If it's not an exact match to the target, it should be a better result under our likelihood."
    );
    assert_abs_diff_eq!(&target, &fit.result, epsilon = 0.01);

    let target_nostd: Array1<f32> =
        array_from_csv("tests/R/log_regularization/iris_setosa_l1_l2_1e-2_nostd.csv")?;
    let model_std = ModelBuilder::<Logistic>::data(&y_data, &x_data)
        .no_standardize()
        .build()?;
    // The X-data is now standardized by default.
    let fit_nostd = model_std.fit_options().l1_reg(1e-2).l2_reg(1e-2).fit()?;

    // ADMM convergence is harder without standardization for separable data. We don't require
    // an exact match, just that the result is in the right neighborhood.
    assert_abs_diff_eq!(&target_nostd, &fit_nostd.result, epsilon = 0.05);
    Ok(())
}

#[test]
fn ridge_seperable() -> Result<()> {
    let (y_labels, x_data) = y_x_from_iris()?;
    let y_data: Array1<bool> = y_labels.mapv(|i| i == 0);
    // Compare against glmnet standardize=FALSE on raw data, since no_standardize() is used.
    let target: Array1<f32> =
        array_from_csv("tests/R/log_regularization/iris_setosa_l2_1e-2_nostd.csv")?;
    // Explicitly skip standardization to test the ability of ridge to converge.
    let model = ModelBuilder::<Logistic>::data(&y_data, &x_data)
        .no_standardize()
        .build()?;
    // Temporarily try L2 for testing
    let fit = model.fit_options().l2_reg(1e-2).fit()?;
    // This still appears to be positive so our result is better
    // Ensure that our result is better, even if the parameters aren't epsilon-equivalent.
    assert!(fit.lr_test_against(&target) > -f32::EPSILON);
    // their result seems less precise, even when reducing the threshold.
    assert_abs_diff_eq!(&target, &fit.result, epsilon = 2e-3);
    Ok(())
}

#[test]
fn lasso_versicolor() -> Result<()> {
    let (y_labels, x_data) = y_x_from_iris()?;
    // NOTE: It matches for versicolor, but not setosa (which is fully seperable).
    // versicolor
    let y_data: Array1<bool> = y_labels.mapv(|i| i == 1);
    let target: Array1<f32> =
        array_from_csv("tests/R/log_regularization/iris_versicolor_l1_1e-2.csv")?;
    // The data is standardized internally by default.
    let model = ModelBuilder::<Logistic>::data(&y_data, &x_data).build()?;
    // TODO: test more harshly by increasing lambda. It passes at l1 = 1 at time of writing but
    // taks longer.
    let fit = model.fit_options().l1_reg(1e-2).fit()?;
    // If this is negative then our alg hasn't converged to a good minimum
    assert!(
        fit.lr_test_against(&target) >= 0.,
        "If it's not an exact match to the target, it should be a better result under our likelihood."
    );
    // The epsilon tolerance doesn't need to be very low if we've found a better minimum
    assert_abs_diff_eq!(&target, &fit.result, epsilon = 0.01);
    Ok(())
}

#[test]
fn linear_ridge() -> Result<()> {
    let (y, x, _var_wt, _freq_wt) = load_linear_data()?;
    let model = ModelBuilder::<Linear>::data(&y, &x).build()?;
    let fit_unreg = model.fit()?;

    let model = ModelBuilder::<Linear>::data(&y, &x).build()?;
    let fit = model.fit_options().l2_reg(0.1).fit()?;

    // Ridge should converge
    assert!(fit.n_iter <= 32, "Ridge should converge");

    // Ridge likelihood should be >= unregularized when evaluated with the regularizer
    // (the regularized fit maximizes the penalized likelihood)
    assert!(
        fit.lr_test_against(&fit_unreg.result) >= 0.,
        "Ridge fit should be at least as good under its own penalized likelihood"
    );

    // Coefficients should be shrunk toward zero compared to OLS
    let ridge_norm: f64 = fit.result.mapv(|x| x * x).sum();
    let ols_norm: f64 = fit_unreg.result.mapv(|x| x * x).sum();
    assert!(
        ridge_norm < ols_norm,
        "Ridge coefficients should have smaller L2 norm"
    );

    Ok(())
}

#[test]
fn linear_ridge_var_weights() -> Result<()> {
    let (y, x, var_wt, _freq_wt) = load_linear_data()?;
    let model = ModelBuilder::<Linear>::data(&y, &x)
        .var_weights(var_wt.clone())
        .build()?;
    let fit_unreg = model.fit()?;

    let model = ModelBuilder::<Linear>::data(&y, &x)
        .var_weights(var_wt)
        .build()?;
    let fit = model.fit_options().l2_reg(0.1).fit()?;

    assert!(fit.n_iter <= 32, "Ridge with var weights should converge");

    let ridge_norm: f64 = fit.result.mapv(|x| x * x).sum();
    let ols_norm: f64 = fit_unreg.result.mapv(|x| x * x).sum();
    assert!(
        ridge_norm < ols_norm,
        "Ridge coefficients should have smaller L2 norm"
    );

    Ok(())
}

#[test]
fn linear_lasso() -> Result<()> {
    let (y, x, _var_wt, _freq_wt) = load_linear_data()?;
    let model = ModelBuilder::<Linear>::data(&y, &x).build()?;
    let fit_unreg = model.fit()?;

    let model = ModelBuilder::<Linear>::data(&y, &x).build()?;
    let fit = model.fit_options().l1_reg(0.1).fit()?;

    assert!(fit.n_iter <= 64, "Lasso should converge");

    // L1 should shrink coefficients
    let lasso_norm: f64 = fit.result.iter().map(|x| x.abs()).sum();
    let ols_norm: f64 = fit_unreg.result.iter().map(|x| x.abs()).sum();
    assert!(
        lasso_norm < ols_norm,
        "Lasso coefficients should have smaller L1 norm"
    );

    Ok(())
}

#[test]
fn linear_elastic_net() -> Result<()> {
    let (y, x, _var_wt, _freq_wt) = load_linear_data()?;
    let model = ModelBuilder::<Linear>::data(&y, &x).build()?;
    let fit_unreg = model.fit()?;

    let model = ModelBuilder::<Linear>::data(&y, &x).build()?;
    let fit = model.fit_options().l1_reg(0.05).l2_reg(0.05).fit()?;

    assert!(fit.n_iter <= 64, "Elastic net should converge");

    // Should shrink both norms
    let en_l1: f64 = fit.result.iter().map(|x| x.abs()).sum();
    let ols_l1: f64 = fit_unreg.result.iter().map(|x| x.abs()).sum();
    assert!(
        en_l1 < ols_l1,
        "Elastic net coefficients should have smaller L1 norm"
    );

    let en_l2: f64 = fit.result.mapv(|x| x * x).sum();
    let ols_l2: f64 = fit_unreg.result.mapv(|x| x * x).sum();
    assert!(
        en_l2 < ols_l2,
        "Elastic net coefficients should have smaller L2 norm"
    );

    Ok(())
}
