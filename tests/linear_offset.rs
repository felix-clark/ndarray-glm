//! testing closure with a linear offset

use anyhow::Result;
use approx::assert_abs_diff_eq;
use ndarray::{array, Array1, Array2};
use ndarray_glm::{linear::Linear, model::ModelBuilder};

#[test]
/// Check that the result is the same in linear regression when subtracting
/// offsets from the y values as it is when adding linear offsets to the model.
fn lin_off_0() -> Result<()> {
    let y_data: Array1<f64> = array![0.6, 0.3, 0.5, 0.1];
    let offsets: Array1<f64> = array![0.1, -0.1, 0.2, 0.0];
    let x_data: Array2<f64> = array![[1.2, 0.7], [2.1, 0.8], [1.5, 0.6], [1.6, 0.3]];

    let lin_model = ModelBuilder::<Linear>::data(&y_data, &x_data)
        .linear_offset(offsets.clone())
        .build()?;
    let lin_fit = lin_model.fit()?;
    let y_offset = y_data - offsets;
    let lin_model_off = ModelBuilder::<Linear>::data(&y_offset, &x_data).build()?;
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

#[test]
// Ensure that the linear offset term adjusts all values sanely.
// TODO: similar test for all types of regression, to ensure they are using
// linear_predictor() properly.
fn lin_off_1() -> Result<()> {
    let data_x = array![
        [-0.23, 2.1, 0.7],
        [1.2, 4.5, 1.3],
        [0.42, 1.8, 0.97],
        [0.4, 3.2, -0.3]
    ];
    let data_y = array![1.23, 0.91, 2.34, 0.62];
    let model = ModelBuilder::<Linear>::data(&data_y, &data_x).build()?;
    let fit = model.fit()?;
    let result = fit.result;
    // a constant linear offset to add for easy checking
    let lin_off = 1.832;
    let lin_offsets = array![lin_off, lin_off, lin_off, lin_off];
    let model_off = ModelBuilder::<Linear>::data(&data_y, &data_x)
        .linear_offset(lin_offsets)
        .build()?;
    let off_fit = model_off.fit()?;
    dbg!(off_fit.n_iter);
    let off_result = off_fit.result;
    let mut compensated_offset_result = off_result.clone();
    compensated_offset_result[0] += lin_off;
    assert_abs_diff_eq!(
        result,
        compensated_offset_result,
        epsilon = 16. * std::f64::EPSILON
    );
    Ok(())
}
