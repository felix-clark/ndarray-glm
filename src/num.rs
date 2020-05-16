//! numerical trait constraints
use ndarray_linalg::lapack::Lapack;
use num_traits;

pub trait Float: Sized + num_traits::Float + Lapack {}

impl Float for f32 {}
impl Float for f64 {}
