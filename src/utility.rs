//! utility functions for internal library use

use ndarray::{stack, Array2, Axis};
use num_traits::identities::One;

/// prepend the input with a column of ones.
/// useful to describe a constant term in a regression in a general way with the data.
/// NOTE: This creates a copy, which may not be memory efficient. It can be used in such a way such that the old value is dropped.
pub fn one_pad<T>(data: &Array2<T>) -> Array2<T>
where
    T: Copy + One,
{
    // create the ones column
    let ones: Array2<T> = Array2::ones((data.nrows(), 1));
    // This should be guaranteed to succeed since we are manually specifying the dimension
    stack(Axis(1), &[ones.view(), data.view()]).unwrap()
}
