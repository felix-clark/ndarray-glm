//! utility functions for internal library use

use ndarray::{concatenate, Array2, ArrayView2, Axis};
use num_traits::identities::One;

/// prepend the input with a column of ones.
/// useful to describe a constant term in a regression in a general way with the data.
pub fn one_pad<T>(data: ArrayView2<T>) -> Array2<T>
where
    T: Copy + One,
{
    // create the ones column
    let ones: Array2<T> = Array2::ones((data.nrows(), 1));
    // This should be guaranteed to succeed since we are manually specifying the dimension
    concatenate![Axis(1), ones, data]
}
