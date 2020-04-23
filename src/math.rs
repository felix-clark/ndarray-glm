//! Mathematical helper functions
use num_traits::Float;

/// The product-logarithm function (not the W function) x * log(x). If x == 0, 0 is returned.
pub fn prod_log<F>(x: F) -> F
where
    F: Float,
{
    if x == F::zero() {
        return F::zero();
    }
    x * x.ln()
}
