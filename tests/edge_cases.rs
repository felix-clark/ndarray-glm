//! Handles edge cases that have caused trouble at times.
use anyhow::Result;
use ndarray::{array, Array2};
use ndarray_glm::{logistic::Logistic, model::ModelBuilder};
use num_traits::Float;

/// Ensure that a valid likelihood is returned when the initial guess is the
/// best one.
#[test]
fn start_zero() -> Result<()> {
    // Exactly half of the data are true, meaning the initial guess of beta = 0 will be the best.
    let data_y = array![true, false, false, true];
    let data_x: Array2<f64> = array![[], [], [], []];
    let model = ModelBuilder::<Logistic>::data(&data_y, &data_x).build()?;
    let fit = model.fit()?;
    assert_eq!(fit.model_like > -f64::infinity(), true);

    Ok(())
}
