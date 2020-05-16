//! test cases for logistic regression

use anyhow::Result;

use ndarray_glm::{Logistic, ModelBuilder};
mod common;
use common::y_x_off_from_csv;

#[test]
// this data caused an infinite loop with step halving
fn log_termination_0() -> Result<()> {
    let (y, x, off) = y_x_off_from_csv::<bool, f32>("tests/data/log_termination_0.csv")?;
    let model = ModelBuilder::<Logistic>::data(y.view(), x.view())
        .linear_offset(off)
        .build()?;
    let fit = model.fit()?;
    dbg!(fit.result);
    dbg!(fit.n_iter);
    Ok(())
}

#[test]
// this data caused an infinite loop with step halving
fn log_termination_1() -> Result<()> {
    let (y, x, off) = y_x_off_from_csv::<bool, f32>("tests/data/log_termination_1.csv")?;
    let model = ModelBuilder::<Logistic>::data(y.view(), x.view())
        .linear_offset(off)
        .build()?;
    let fit = model.fit()?;
    dbg!(fit.result);
    dbg!(fit.n_iter);
    Ok(())
}

#[test]
fn log_regularization() -> Result<()> {
    let (y, x, off) = y_x_off_from_csv::<bool, f32>("tests/data/log_regularization.csv")?;
    // This can be terminated either by standardizing the data or by using
    // lambda = 2e-6 intead of 1e-6.
    // let x = ndarray_glm::standardize::standardize(x);
    let model = ModelBuilder::<Logistic>::data(y.view(), x.view())
        .linear_offset(off)
        .l2_reg(2e-6)
        .build()?;
    let fit = model.fit()?;
    dbg!(fit.result);
    dbg!(fit.n_iter);
    Ok(())
}
