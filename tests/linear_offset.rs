//! testing closure with a linear offset

use anyhow::Result;
use approx::assert_abs_diff_eq;
use ndarray::{array, Array1, Array2};
use ndarray_glm::{linear::Linear, model::ModelBuilder};

#[test]
/// Check that the result is the same in linear regression when subtracting
/// offsets from the y values as it is when adding linear offsets to the model.
fn test_intercept() -> Result<()> {
    let y_data: Array1<f64> = array![0.6, 0.3, 0.5, 0.1];
    let offsets: Array1<f64> = array![0.1, -0.1, 0.2, 0.0];
    let x_data: Array2<f64> = array![[1.2, 0.7], [2.1, 0.8], [1.5, 0.6], [1.6, 0.3]];

    let lin_model = ModelBuilder::<Linear, _>::new(&y_data, &x_data)
        .linear_offset(offsets.clone())
        .build()?;
    let lin_fit = lin_model.fit()?;
    let y_offset = y_data - offsets;
    let lin_model_off = ModelBuilder::<Linear, _>::new(&y_offset, &x_data).build()?;
    let lin_fit_off = lin_model_off.fit()?;
    dbg!(&lin_fit.result);
    dbg!(&lin_fit_off.result);
    // Ensure that the two methods give consistent results
    assert_abs_diff_eq!(
        lin_fit.result,
        lin_fit_off.result,
        epsilon = 16.0 * std::f64::EPSILON
    );

    Ok(())
}
