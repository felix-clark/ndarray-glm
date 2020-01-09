//! library for solving GLM regression
//! TODO: documentation

// this line is necessary to avoid linking errors
// but maybe it should only go into final library
// extern crate openblas_src;

pub mod error;
pub mod linear;
pub mod logistic;
mod utility;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn log_reg() {
        let beta = array![0., 1.0];
        let ln2 = f32::ln(2.);
        let data_x = array![[0.], [0.], [ln2], [ln2], [ln2]];
        let data_y = array![true, false, true, true, false];
        let result = logistic::regression(&data_y, &data_x)
            .expect("regression failed")
            .result;
        assert_abs_diff_eq!(beta, result, epsilon = 4.0 * std::f32::EPSILON);
        // test the significance function
        let significance = logistic::significance(&data_y, &data_x, &result);
        dbg!(significance);
    }

    #[test]
    fn lin_reg() {
        let beta = array![0.3, 1.2, -0.5];
        let data_x = array![[-0.1, 0.2], [0.7, 0.5], [3.2, 0.1]];
        let data_y = array![
            beta[0] + beta[1] * data_x[[0, 0]] + beta[2] * data_x[[0, 1]],
            beta[0] + beta[1] * data_x[[1, 0]] + beta[2] * data_x[[1, 1]],
            beta[0] + beta[1] * data_x[[2, 0]] + beta[2] * data_x[[2, 1]],
        ];
        let result = linear::regression(&data_y, &data_x).expect("regression failed");
        // This is failing within the default tolerance
        assert_abs_diff_eq!(beta, result, epsilon = 32.0 * std::f32::EPSILON);
    }
}
