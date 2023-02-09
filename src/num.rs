//! numerical trait constraints
use std::cmp;

use ndarray::ScalarOperand;
use ndarray_linalg::Lapack;

pub trait Float: Sized + num_traits::Float + Lapack + ScalarOperand {
    // Return 1/2 = 0.5
    fn half() -> Self;

    /// A more conventional sign function, because the built-in signum treats signed zeros as
    /// positive and negative: https://github.com/rust-lang/rust/issues/57543
    fn sign(self) -> Self {
        if self == Self::zero() {
            Self::zero()
        } else {
            self.signum()
        }
    }

    /// total_cmp is not implemented in num_traits, so implement it here.
    fn total_cmp(&self, other: &Self) -> cmp::Ordering;
}

impl Float for f32 {
    fn half() -> Self {
        0.5
    }

    fn total_cmp(&self, other: &Self) -> cmp::Ordering {
        self.total_cmp(other)
    }
}
impl Float for f64 {
    fn half() -> Self {
        0.5
    }

    fn total_cmp(&self, other: &Self) -> cmp::Ordering {
        self.total_cmp(other)
    }
}
