//! Mathematical helper functions
use crate::num::Float;

/// The product-logarithm function (not the W function) x * log(x). If x == 0, 0 is returned.
pub(crate) fn prod_log<F>(x: F) -> F
where
    F: Float,
{
    if x == F::zero() {
        return F::zero();
    }
    x * num_traits::Float::ln(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_prod_log() {
        assert_abs_diff_eq!(0., prod_log(0.));
        let e: f64 = std::f64::consts::E;
        assert_abs_diff_eq!(e, prod_log(e));
    }
}
