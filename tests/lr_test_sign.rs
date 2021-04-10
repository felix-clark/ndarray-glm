//! Data that has resulted in a significantly negative LR test.
mod common;
use anyhow::Result;
use common::y_x_off_from_csv;
use ndarray_glm::{Logistic, ModelBuilder};

#[test]
fn lr_test_sign0() -> Result<()> {
    // TODO: this assumes the tests are run from the root directory of the
    // crate. This might not be true in general, but it often will be.
    let (y, x, off) = y_x_off_from_csv::<bool, f32>("tests/data/lr_test_sign0.csv")?;
    let model = ModelBuilder::<Logistic>::data(&y, &x)
        .linear_offset(off)
        .build()?;
    let fit = model.fit_options().l2_reg(2e-6).fit()?;
    dbg!(&fit.result);
    assert_eq!(fit.lr_test() >= 0., true);
    Ok(())
}
