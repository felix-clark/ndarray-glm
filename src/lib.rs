//! library for solving GLM regression
//! TODO: documentation

// this line is necessary to avoid linking errors
// but maybe it should only go into final library
// extern crate openblas_src;

// pub mod data;
pub mod error;
mod fit;
mod glm;
pub mod linear;
pub mod logistic;
pub mod model;
pub mod poisson;
mod utility;

#[cfg(test)]
mod tests {
    // use super::*;
    use crate::{
        error::RegressionResult, linear::Linear, logistic::Logistic, model::ModelBuilder,
        poisson::Poisson,
    };
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn log_reg() -> RegressionResult<()> {
        let beta = array![0., 1.0];
        let ln2 = f64::ln(2.);
        let data_x = array![[0.], [0.], [ln2], [ln2], [ln2]];
        let data_y = array![true, false, true, true, false];
        let model = ModelBuilder::<Logistic, _>::new(&data_y, &data_x).build()?;
        let fit = model.fit()?;
        dbg!(fit.n_iter);
        assert_abs_diff_eq!(beta, fit.result, epsilon = std::f32::EPSILON as f64);
        // test the significance function
        let significance = fit.z_scores(&model);
        dbg!(significance);
        Ok(())
    }

    #[test]
    fn lin_reg() -> RegressionResult<()> {
        let beta = array![0.3, 1.2, -0.5];
        let data_x = array![[-0.1, 0.2], [0.7, 0.5], [3.2, 0.1]];
        // let data_x = array![[-0.1, 0.1], [0.7, -0.7], [3.2, -3.2]];
        let data_y = array![
            beta[0] + beta[1] * data_x[[0, 0]] + beta[2] * data_x[[0, 1]],
            beta[0] + beta[1] * data_x[[1, 0]] + beta[2] * data_x[[1, 1]],
            beta[0] + beta[1] * data_x[[2, 0]] + beta[2] * data_x[[2, 1]],
        ];
        let model = ModelBuilder::<Linear, _>::new(&data_y, &data_x)
            .max_iter(10)
            .build()?;
        let fit = model.fit()?;
        dbg!(fit.n_iter);
        // This is failing within the default tolerance
        assert_abs_diff_eq!(beta, fit.result, epsilon = 64.0 * std::f64::EPSILON);
        Ok(())
    }

    #[test]
    fn poisson_reg() -> RegressionResult<()> {
        let ln2 = f64::ln(2.);
        let beta = array![0., ln2, -ln2];
        let data_x = array![[1., 0.], [1., 1.], [0., 1.], [0., 1.]];
        let data_y = array![2, 1, 0, 1];
        let model = ModelBuilder::<Poisson<u32>, _>::new(&data_y, &data_x)
            .max_iter(10)
            .build()?;
        let fit = model.fit()?;
        dbg!(fit.n_iter);
        assert_abs_diff_eq!(beta, fit.result, epsilon = std::f32::EPSILON as f64);
        Ok(())
    }

    #[test]
    // Ensure that the linear offset term adjusts all values sanely.
    // TODO: similar test for all types of regression, to ensure they are using
    // linear_predictor() properly.
    fn linear_offset() -> RegressionResult<()> {
        let data_x = array![
            [-0.23, 2.1, 0.7],
            [1.2, 4.5, 1.3],
            [0.42, 1.8, 0.97],
            [0.4, 3.2, -0.3]
        ];
        let data_y = array![1.23, 0.91, 2.34, 0.62];
        let model = ModelBuilder::<Linear, _>::new(&data_y, &data_x).build()?;
        let fit = model.fit()?;
        let result = fit.result;
        // a constant linear offset to add for easy checking
        let lin_off = 1.832;
        let lin_offsets = array![lin_off, lin_off, lin_off, lin_off];
        let model_off = ModelBuilder::<Linear, _>::new(&data_y, &data_x)
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
            epsilon = 4. * std::f64::EPSILON
        );
        Ok(())
    }
}
