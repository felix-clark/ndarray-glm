//! numerical trait constraints
use ndarray::ScalarOperand;
use ndarray_linalg::Lapack;

pub trait Float: Sized + num_traits::Float + Lapack + ScalarOperand {
    /// A more conventional sign function, because the built-in signum treats signed zeros as
    /// positive and negative: https://github.com/rust-lang/rust/issues/57543
    fn sign(self) -> Self {
        if self == Self::zero() {
            Self::zero()
        } else {
            self.signum()
        }
    }
}

impl Float for f32 {}
impl Float for f64 {}
