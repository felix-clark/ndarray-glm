//! Standardization of a design matrix.
use ndarray::{Array1, Array2, Axis};
use num_traits::{Float, FromPrimitive};

/// Returns a standardization of a design matrix where rows are seperate
/// observations and columns are different dependent variables. Each quantity
/// has its mean subtracted and is then divided by the standard deviation.
pub fn standardize<F: Float>(mut design: Array2<F>) -> Array2<F>
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
