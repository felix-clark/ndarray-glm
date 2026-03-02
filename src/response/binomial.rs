//! Regression with a binomial response function. The N parameter must be known ahead of time.
#[cfg(feature = "stats")]
use crate::response::Response;
use crate::{
    error::{RegressionError, RegressionResult},
    glm::{DispersionType, Glm},
    math::prod_log,
    num::Float,
    response::Yval,
};
#[cfg(feature = "stats")]
use statrs::distribution::Binomial as BinDist;

/// Use a fixed type of u16 for the domain of the binomial distribution.
type BinDom = u16;

/// Binomial regression with a fixed N. Non-canonical link functions are not
/// possible at this time due to the awkward ergonomics with the const trait
/// parameter N.
pub struct Binomial<const N: BinDom>;

impl<const N: BinDom> Yval<Binomial<N>> for BinDom {
    fn into_float<F: Float>(self) -> RegressionResult<F, F> {
        F::from(self).ok_or_else(|| RegressionError::InvalidY(self.to_string()))
    }
}

#[cfg(feature = "stats")]
impl<const N: BinDom> Response for Binomial<N> {
    type DistributionType = BinDist;

    fn get_distribution(mu: f64, _phi: f64) -> Self::DistributionType {
        use num_traits::ToPrimitive;

        let p = mu / N.to_f64().unwrap();
        BinDist::new(p, N.into()).unwrap()
    }
}

impl<const N: BinDom> Glm for Binomial<N> {
    /// Only the canonical link function is available for binomial regression.
    type Link = link::Logit;
    const DISPERSED: DispersionType = DispersionType::NoDispersion;

    /// The log-partition function for the binomial distribution is similar to
    /// that for logistic regression, but it is adjusted for the maximum value.
    fn log_partition<F: Float>(nat_par: F) -> F {
        let n: F = F::from(N).unwrap();
        n * num_traits::Float::exp(nat_par).ln_1p()
    }

    fn variance<F: Float>(mean: F) -> F {
        let n_float: F = F::from(N).unwrap();
        mean * (n_float - mean) / n_float
    }

    fn log_like_sat<F: Float>(y: F) -> F {
        let n: F = F::from(N).unwrap();
        prod_log(y) + prod_log(n - y) - prod_log(n)
    }
}

pub mod link {
    use super::*;
    use crate::link::{Canonical, Link};
    use num_traits::Float;

    pub struct Logit {}
    impl Canonical for Logit {}
    impl<const N: BinDom> Link<Binomial<N>> for Logit {
        fn func<F: Float>(y: F) -> F {
            let n_float: F = F::from(N).unwrap();
            Float::ln(y / (n_float - y))
        }
        fn func_inv<F: Float>(lin_pred: F) -> F {
            let n_float: F = F::from(N).unwrap();
            n_float / (F::one() + (-lin_pred).exp())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Binomial;
    use crate::{error::RegressionResult, model::ModelBuilder};
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn bin_reg() -> RegressionResult<(), f64> {
        const N: u16 = 12;
        let ln2 = f64::ln(2.);
        let beta = array![0., 1.];
        let data_x = array![[0.], [0.], [ln2], [ln2], [ln2]];
        // the first two data points should average to 6 and the last 3 should average to 8.
        let data_y = array![5, 7, 9, 6, 9];
        let model = ModelBuilder::<Binomial<N>>::data(&data_y, &data_x).build()?;
        let fit = model.fit()?;
        dbg!(&fit.result);
        dbg!(&fit.n_iter);
        assert_abs_diff_eq!(beta, fit.result, epsilon = 0.05 * f32::EPSILON as f64);
        Ok(())
    }

    #[test]
    // Confirm logit closure explicitly.
    fn logit_closure() {
        use super::link::Logit;
        use crate::link::TestLink;
        // as for logistic, floats lose precision on difference from 1 relative to 0, so we can't
        // use x values that are very large at all.
        let x = array![-400., -50., -2.0, -0.2, 0., 0.5, 20.];
        <Logit as TestLink<Binomial<10>>>::check_closure(&x);
        let y = array![0., 1e-5, 0.2, 0.5, 0.75, 0.9999, 1.0];
        <Logit as TestLink<Binomial<12>>>::check_closure_y(&y);
    }
}
