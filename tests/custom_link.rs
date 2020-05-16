//! Test implementation of custom link functions

use anyhow::Result;
use approx::assert_abs_diff_eq;
use ndarray::{array, Array1, Axis};
use ndarray_glm::{
    link::{Link, Transform},
    num::Float,
    Linear, ModelBuilder,
};

#[test]
fn linear_with_lin_transform() -> Result<()> {
    // A linear transformation for simplicity.
    struct LinTran {}
    impl Link<Linear<LinTran>> for LinTran {
        fn func<F: Float>(y: F) -> F {
            F::from(2.5).unwrap() * y - F::from(3.4).unwrap()
        }
        fn func_inv<F: Float>(lin_pred: F) -> F {
            (lin_pred + F::from(3.4).unwrap()) * F::from(0.4).unwrap()
        }
    }
    assert_abs_diff_eq!(
        LinTran::func(LinTran::func_inv(0.45)),
        0.45,
        epsilon = 4. * f64::EPSILON
    );
    impl Transform for LinTran {
        fn nat_param<F: Float>(lin_pred: Array1<F>) -> Array1<F> {
            lin_pred.mapv_into(Self::func_inv)
        }
        fn d_nat_param<F: Float>(lin_pred: &Array1<F>) -> Array1<F> {
            Array1::<F>::from_elem(lin_pred.len(), F::from(0.4).unwrap())
        }
    }
    let beta = array![-0.2, 0.7];
    let data_x = array![-1.5, -1.2, -0.8, -0.8, -0.5, -0.2, -0.2, 0.3, 0.3, 0.7, 0.9, 1.2, 1.2];
    let mut data_y = data_x.mapv(|x| LinTran::func_inv(beta[0] + beta[1] * x));
    // some x points are redundant, and Gaussian errors are symmetric, so some
    // pairs of points can be moved off of the exact fit without affecting the
    // result.
    data_y[2] += 0.3;
    data_y[3] -= 0.3;
    data_y[5] -= 0.2;
    data_y[6] += 0.2;
    data_y[7] += 0.4;
    data_y[8] -= 0.4;
    data_y[11] -= 0.3;
    data_y[12] += 0.3;
    // Change X data into a 2D array
    let data_x = data_x.insert_axis(Axis(1));
    let model = ModelBuilder::<Linear<LinTran>>::data(data_y.view(), data_x.view()).build()?;
    let fit = model.fit()?;
    dbg!(fit.n_iter);
    dbg!(&fit.result);
    dbg!(&beta);
    assert_abs_diff_eq!(fit.result, beta, epsilon = 16.0 * f64::EPSILON);
    Ok(())
}

#[test]
fn linear_with_cubic() -> Result<()> {
    // An adjusted cube root link function to test on Linear regression. This
    // fits to y ~ (a + b*x)^3. If the starting guess is zero this fails to
    // converge because the derivative of the link function is zero at the
    // origin.
    struct Cbrt {}
    impl Link<Linear<Cbrt>> for Cbrt {
        fn func<F: Float>(y: F) -> F {
            y.cbrt()
        }
        fn func_inv<F: Float>(lin_pred: F) -> F {
            lin_pred.powi(3)
        }
    }
    assert_abs_diff_eq!(
        Cbrt::func(Cbrt::func_inv(0.45)),
        0.45,
        epsilon = 4. * f64::EPSILON
    );
    impl Transform for Cbrt {
        fn nat_param<F: Float>(lin_pred: Array1<F>) -> Array1<F> {
            lin_pred.mapv_into(|w| w.powi(3))
        }
        fn d_nat_param<F: Float>(lin_pred: &Array1<F>) -> Array1<F> {
            let three = F::from(3.).unwrap();
            lin_pred.mapv(|w| three * w.powi(2))
        }
    }

    type TestLink = Cbrt;
    let beta = array![-0.2, 0.7];
    let data_x = array![-1.5, -1.2, -0.8, -0.8, -0.5, -0.2, -0.2, 0.3, 0.3, 0.7, 0.9, 1.2, 1.2];
    let mut data_y = data_x.mapv(|x| TestLink::func_inv(beta[0] + beta[1] * x));
    // some x points are redundant, and Gaussian errors are symmetric, so some
    // pairs of points can be moved off of the exact fit without affecting the
    // result.
    data_y[2] += 0.3;
    data_y[3] -= 0.3;
    data_y[5] -= 0.2;
    data_y[6] += 0.2;
    data_y[7] += 0.4;
    data_y[8] -= 0.4;
    data_y[11] -= 0.3;
    data_y[12] += 0.3;
    // Change X data into a 2D array
    let data_x = data_x.insert_axis(Axis(1));
    let model = ModelBuilder::<Linear<TestLink>>::data(data_y.view(), data_x.view()).build()?;
    eprintln!("Built model");
    let fit = model.fit()?;
    dbg!(fit.n_iter);
    dbg!(&fit.result);
    dbg!(&beta);
    assert_abs_diff_eq!(fit.result, beta, epsilon = f32::EPSILON as f64);
    Ok(())
}
