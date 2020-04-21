//! Model for Poisson regression

use crate::{
    glm::{Glm, Response},
    link::Link,
};
use ndarray::Array1;
use num_traits::{Float, ToPrimitive, Unsigned};
use std::marker::PhantomData;

/// Poisson regression over an unsigned integer type.
pub struct Poisson<L = link::Log>
where
    L: Link<Poisson<L>>,
{
    _link: PhantomData<L>,
}

/// Poisson variables can be any unsigned integer.
impl<U, L> Response<Poisson<L>> for U
where
    U: Unsigned + ToPrimitive,
    L: Link<Poisson<L>>,
{
    fn to_float<F: Float>(self) -> F {
        F::from(self).unwrap()
    }
}
// TODO: A floating point response for Poisson might also be do-able.

impl<L> Glm for Poisson<L>
where
    L: Link<Poisson<L>>,
{
    type Link = L;

    /// The logarithm of the partition function for Poisson is the exponential of the natural parameter, which is the logarithm of the mean.
    fn log_partition<F: Float>(nat_par: &Array1<F>) -> F {
        nat_par.mapv(F::exp).sum()
    }

    /// The variance of a Poisson variable is equal to its mean.
    fn variance<F: Float>(mean: F) -> F {
        mean
    }
}

pub mod link {
    //! Link functions for Poisson regression
    use super::Poisson;
    use crate::link::{Canonical, Link};
    use num_traits::Float;

    /// The canonical link function of the Poisson response is the logarithm.
    pub struct Log {}
    impl Canonical for Log {}
    impl Link<Poisson<Log>> for Log {
        fn func<F: Float>(y: F) -> F {
            y.ln()
        }
        fn func_inv<F: Float>(lin_pred: F) -> F {
            lin_pred.exp()
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{error::RegressionResult, model::ModelBuilder, poisson::Poisson};
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array1};

    #[test]
    fn poisson_reg() -> RegressionResult<()> {
        let ln2 = f64::ln(2.);
        let beta = array![0., ln2, -ln2];
        let data_x = array![[1., 0.], [1., 1.], [0., 1.], [0., 1.]];
        let data_y: Array1<u32> = array![2, 1, 0, 1];
        let model = ModelBuilder::<Poisson>::data(&data_y, &data_x)
            .max_iter(10)
            .build()?;
        let fit = model.fit()?;
        dbg!(fit.n_iter);
        assert_abs_diff_eq!(beta, fit.result, epsilon = std::f32::EPSILON as f64);
        Ok(())
    }
}
