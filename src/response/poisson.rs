//! Model for Poisson regression

#[cfg(feature = "stats")]
use crate::response::Response;
use crate::{
    error::{RegressionError, RegressionResult},
    glm::{DispersionType, Glm},
    link::Link,
    math::prod_log,
    num::Float,
    response::Yval,
};
use num_traits::{ToPrimitive, Unsigned};
#[cfg(feature = "stats")]
use statrs::distribution::Poisson as PoisDist;
use std::marker::PhantomData;

/// Poisson regression over an unsigned integer type.
pub struct Poisson<L = link::Log>
where
    L: Link<Poisson<L>>,
{
    _link: PhantomData<L>,
}

/// Poisson variables can be any unsigned integer.
impl<U, L> Yval<Poisson<L>> for U
where
    U: Unsigned + ToPrimitive + ToString + Copy,
    L: Link<Poisson<L>>,
{
    fn into_float<F: Float>(self) -> RegressionResult<F, F> {
        F::from(self).ok_or_else(|| RegressionError::InvalidY(self.to_string()))
    }
}
// TODO: A floating point response for Poisson might also be do-able.

#[cfg(feature = "stats")]
impl<L> Response for Poisson<L>
where
    L: Link<Poisson<L>>,
{
    type DistributionType = PoisDist;

    fn get_distribution(mu: f64, _phi: f64) -> Self::DistributionType {
        PoisDist::new(mu).unwrap()
    }
}

impl<L> Glm for Poisson<L>
where
    L: Link<Poisson<L>>,
{
    type Link = L;
    const DISPERSED: DispersionType = DispersionType::NoDispersion;

    /// The logarithm of the partition function for Poisson is the exponential of the natural
    /// parameter, which is the logarithm of the mean.
    fn log_partition<F: Float>(nat_par: F) -> F {
        num_traits::Float::exp(nat_par)
    }

    /// The variance of a Poisson variable is equal to its mean.
    fn variance<F: Float>(mean: F) -> F {
        mean
    }

    /// The saturation likelihood of the Poisson distribution is non-trivial.
    /// It is equal to y * (log(y) - 1).
    fn log_like_sat<F: Float>(y: F) -> F {
        prod_log(y) - y
    }
}

pub mod link {
    //! Link functions for Poisson regression
    use super::Poisson;
    use crate::{
        link::{Canonical, Link},
        num::Float,
    };

    /// The canonical link function of the Poisson response is the logarithm.
    pub struct Log {}
    impl Canonical for Log {}
    impl Link<Poisson<Log>> for Log {
        fn func<F: Float>(y: F) -> F {
            num_traits::Float::ln(y)
        }
        fn func_inv<F: Float>(lin_pred: F) -> F {
            num_traits::Float::exp(lin_pred)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{error::RegressionResult, model::ModelBuilder};
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, array};

    #[test]
    fn poisson_reg() -> RegressionResult<(), f64> {
        let ln2 = f64::ln(2.);
        let beta = array![0., ln2, -ln2];
        let data_x = array![[1., 0.], [1., 1.], [0., 1.], [0., 1.]];
        let data_y: Array1<u32> = array![2, 1, 0, 1];
        let model = ModelBuilder::<Poisson>::data(&data_y, &data_x).build()?;
        let fit = model.fit_options().max_iter(10).fit()?;
        dbg!(fit.n_iter);
        assert_abs_diff_eq!(beta, fit.result, epsilon = f32::EPSILON as f64);
        Ok(())
    }

    #[test]
    // Confirm log closure explicitly.
    fn logit_closure() {
        use super::link::Log;
        use crate::link::TestLink;
        // Because floats lose precision on difference from 1 relative to 0, higher values get
        // mapped back to infinity under closure. This is sort of fundamental to the logit
        // function and I'm not sure there's a good way around it.
        let x = array![-500., -50., -2.0, -0.2, 0., 0.5, 20.];
        Log::check_closure(&x);
        let y = array![0., 1e-5, 0.25, 0.5, 0.8, 0.9999, 1.0];
        Log::check_closure_y(&y);
    }
}
