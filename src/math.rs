//! Mathematical helper functions
use crate::num::Float;
use ndarray::Array2;
use ndarray_linalg::QRSquareInto;

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

/// Returns true iff the matrix is rank deficient with tolerance `eps` using QR
/// decomposition.
// NOTE: SVD may be faster
pub fn is_rank_deficient<F>(matrix: Array2<F>, eps: F) -> ndarray_linalg::error::Result<bool>
where
    F: Float,
{
    if matrix.ncols() != matrix.nrows() {
        return Ok(true);
    }
    let (_, r) = matrix.qr_square_into()?;
    let diag = r.into_diag();
    for e in diag.into_iter() {
        if F::from(e.abs()).unwrap() < eps {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_prod_log() {
        assert_abs_diff_eq!(0., prod_log(0.));
        let e: f64 = std::f64::consts::E;
        assert_abs_diff_eq!(e, prod_log(e));
    }

    #[test]
    fn test_rank_def() {
        assert_eq!(true, is_rank_deficient(array![[0., 1.]], 0.).unwrap());
        assert_eq!(
            false,
            is_rank_deficient(array![[0., 1.], [2., 0.]], f32::EPSILON as f64).unwrap()
        );
        assert_eq!(
            true,
            is_rank_deficient(array![[0., 1.], [0., 2.342]], f64::EPSILON).unwrap()
        );
        assert_eq!(
            true,
            is_rank_deficient(
                array![[1., 1., 0.], [1., 0.5, 0.5], [1., 0.2, 0.8]],
                f64::EPSILON
            )
            .unwrap()
        );
    }
}
