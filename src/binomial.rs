//! Regression with a binomial response function. The N parameter must be known ahead of time.
//! This submodule uses const_generics, available only in nightly rust, and must
//! be activated with the "binomial" feature option.
use crate::{glm::Glm, model::Model};
// use cauchy::Scalar;
use ndarray::Array1;
use ndarray_linalg::Lapack;
use num_traits::Float;

/// Use a fixed type of u16 for the domain of the binomial distribution.
type BinDom = u16;

/// Binomial regression with a fixed N.
pub struct Binomial<const N: BinDom>;

impl<F, const N: BinDom> Glm<F> for Binomial<N>
where
    F: Float + Lapack,
{
    type Domain = BinDom;

    fn y_float(y: Self::Domain) -> F {
        F::from(y).unwrap()
    }

    fn link(y: F) -> F {
        Float::ln(y / (F::from(N).unwrap() - y))
    }

    fn mean(lin_pred: F) -> F {
        F::from(N).unwrap() * lin_pred
    }

    fn variance(mean: F) -> F {
        let n_float: F = F::from(N).unwrap();
        mean * (n_float - mean) / n_float
    }

    /// The binomial likelihood includes a BetaLn() term of N and y, which can
    /// be skipped for parameter minimization.
    fn quasi_log_likelihood(data: &Model<Self, F>, regressors: &Array1<F>) -> F {
        let lin_pred: Array1<F> = data.linear_predictor(&regressors);
        // in the canonical version, the natural parameter is logit(p)
        let log_like_sum = (&data.y * &lin_pred).sum()
            - F::from(N).unwrap() * lin_pred.mapv(Float::exp).mapv(F::ln_1p).sum();
        log_like_sum + data.l2_like_term(regressors)
    }
}

#[cfg(test)]
mod tests {
    use super::Binomial;
    use crate::{error::RegressionResult, model::ModelBuilder};
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn bin_reg() -> RegressionResult<()> {
        const N: u16 = 12;
        let ln2 = f64::ln(2.);
        let beta = array![0., 1.];
        let data_x = array![[0.], [0.], [ln2], [ln2], [ln2]];
        let data_y = array![5, 7, 9, 6, 9];
        let model = ModelBuilder::<Binomial<N>, _>::new(&data_y, &data_x).build()?;
        let fit = model.fit()?;
        dbg!(&fit.result);
        dbg!(&fit.n_iter);
        assert_abs_diff_eq!(beta, fit.result, epsilon = 32.0 * std::f64::EPSILON);
        Ok(())
    }
}
