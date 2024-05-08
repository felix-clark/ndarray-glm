//! utility functions for internal library use

use ndarray::{concatenate, Array1, Array2, ArrayView2, Axis};
use num_traits::{
    identities::One,
    {Float, FromPrimitive},
};

/// Prepend the input with a column of ones.
/// Used to incorporate a constant intercept term in a regression.
pub fn one_pad<T>(data: ArrayView2<T>) -> Array2<T>
where
    T: Copy + One,
{
    // create the ones column
    let ones: Array2<T> = Array2::ones((data.nrows(), 1));
    // This should be guaranteed to succeed since we are manually specifying the dimension
    concatenate![Axis(1), ones, data]
}

/// Returns a standardization of a design matrix where rows are seperate
/// observations and columns are different dependent variables. Each quantity
/// has its mean subtracted and is then divided by the standard deviation.
/// The normalization by the standard deviation is not performed if there is only 1
/// observation, since such an operation is undefined.
pub fn standardize<F>(mut design: Array2<F>) -> Array2<F>
where
    F: Float + FromPrimitive + std::ops::DivAssign,
{
    let n_obs: usize = design.nrows();
    if n_obs >= 1 {
        // subtract the mean
        design = &design - &design.mean_axis(Axis(0)).expect("mean should succeed here");
    }
    if n_obs >= 2 {
        // divide by the population standard deviation
        let std: Array1<F> = design.std_axis(Axis(0), F::zero());
        // design = &design / &std;
        design.zip_mut_with(&std, |x, &sig| {
            if sig > F::zero() {
                *x /= sig;
            }
        })
    }
    design
}
