//! library for solving GLM regression
//! TODO: documentation

// this line is necessary to avoid linking errors
// but maybe it should only go into final library
// extern crate openblas_src;

pub mod data;
pub mod error;
pub mod fit;
pub mod glm;
pub mod linear;
pub mod logistic;
mod utility;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DataConfigBuilder;
    use crate::error::RegressionResult;
    use crate::glm::Glm;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn log_reg() -> RegressionResult<()> {
        let beta = array![0., 1.0];
        let ln2 = f64::ln(2.);
        let data_x = array![[0.], [0.], [ln2], [ln2], [ln2]];
        // let data_y = array![true, false, true, true, false];
        let data_y = array![1.0, 0.0, 1.0, 1.0, 0.0];
        let data = DataConfigBuilder::new(data_y, data_x).build()?;
        let fit = logistic::Logistic::regression(&data).expect("regression failed");
        dbg!(fit.n_iter);
        assert_abs_diff_eq!(beta, fit.result, epsilon = 32.0 * std::f64::EPSILON);
        // test the significance function
        let significance = fit.z_scores(&data);
        dbg!(significance);
        Ok(())
    }

    #[test]
    fn lin_reg() -> RegressionResult<()> {
        let beta = array![0.3, 1.2, -0.5];
        let data_x = array![[-0.1, 0.2], [0.7, 0.5], [3.2, 0.1]];
        let data_y = array![
            beta[0] + beta[1] * data_x[[0, 0]] + beta[2] * data_x[[0, 1]],
            beta[0] + beta[1] * data_x[[1, 0]] + beta[2] * data_x[[1, 1]],
            beta[0] + beta[1] * data_x[[2, 0]] + beta[2] * data_x[[2, 1]],
        ];
        let data = DataConfigBuilder::new(data_y, data_x)
            .max_iter(10)
            .build()?;
        // let result = linear::regression(&data_y, &data_x).expect("regression failed");
        let fit = linear::Linear::regression(&data).expect("regression failed");
        dbg!(fit.n_iter);
        // This is failing within the default tolerance
        assert_abs_diff_eq!(beta, fit.result, epsilon = 32.0 * std::f64::EPSILON);
        Ok(())
    }
}
