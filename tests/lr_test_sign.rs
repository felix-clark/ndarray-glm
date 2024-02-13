//! Data that has resulted in a significantly negative LR test.
mod common;
use anyhow::Result;
use common::y_x_off_from_csv;
use ndarray_glm::{Logistic, ModelBuilder};

#[test]
fn lr_test_sign0() -> Result<()> {
    // TODO: this assumes the tests are run from the root directory of the
    // crate. This might not be true in general, but it often will be.
    let (y, x, off) = y_x_off_from_csv::<bool, f32, 1>("tests/data/lr_test_sign0.csv")?;
    let model = ModelBuilder::<Logistic>::data(&y, &x)
        .linear_offset(off)
        .build()?;
    let fit = model.fit_options().l2_reg(2e-6).fit()?;
    dbg!(&fit.result);
    assert!(fit.lr_test() >= 0.);
    Ok(())
}

// This test seems to have a first step that has a big jump but lands at exactly the same
// likelihood, so it's useful for testing the step halving and termination logic.
#[test]
fn lr_test_sign1() -> Result<()> {
    let (y, x, off) = y_x_off_from_csv::<bool, f32, 1>("tests/data/lr_test_sign1.csv")?;
    let model = ModelBuilder::<Logistic>::data(&y, &x)
        .linear_offset(off)
        .build()?;
    // This fit failed with regularization in the range of about 3e-7 to 3e-6.
    // Only a single iteration was performed in this case, because step halving was not being
    // engaged when it should have.
    let fit = model.fit_options().l2_reg(1e-6).fit()?;
    assert!(fit.lr_test() >= 0.);
    Ok(())
}
