//! testing regularization

use anyhow::Result;
use approx::assert_abs_diff_eq;
use ndarray::{array, Array1, Array2};
use ndarray_glm::{linear::Linear, model::ModelBuilder, standardize::standardize};

#[test]
/// Test that the intercept is not affected by regularization when the dependent
/// data is centered. This is only strictly true for linear regression.
fn test_intercept() -> Result<()> {
    let y_data: Array1<f64> = array![0.3, 0.5, 0.8, 0.2];
    let x_data: Array2<f64> = array![[1.5, 0.6], [2.1, 0.8], [1.2, 0.7], [1.6, 0.3]];
    // standardize the data
    let x_data = standardize(x_data);

    let lin_model = ModelBuilder::<Linear, _>::new(&y_data, &x_data).build()?;
    let lin_fit = lin_model.fit()?;
    // use a pretty large regularization term to make sure the effect is pronounced
    let lin_model_reg = ModelBuilder::<Linear, _>::new(&y_data, &x_data)
        .l2_reg(0.1)
        .build()?;
    let lin_fit_reg = lin_model_reg.fit()?;
    dbg!(&lin_fit.result);
    dbg!(&lin_fit_reg.result);
    // Ensure that the intercept terms are equal
    assert_abs_diff_eq!(
        lin_fit.result[0],
        lin_fit_reg.result[0],
        epsilon = 2.0 * std::f64::EPSILON
    );

    Ok(())
}
