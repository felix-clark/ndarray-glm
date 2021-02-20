//! numerical trait constraints
use ndarray_linalg::Lapack;

pub trait Float: Sized + num_traits::Float + Lapack {}

impl Float for f32 {}
impl Float for f64 {}
