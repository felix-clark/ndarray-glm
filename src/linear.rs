//! Functions for solving linear regression

use crate::{
    glm::Glm,
    link::{Canonical, Link},
    model::Model,
};
use ndarray::Array1;
use ndarray_linalg::Lapack;
use num_traits::Float;

/// Linear regression with constant variance.
pub struct Linear<F, L = Id>
where
    F: Float + Lapack,
    L: Link<F, Linear<F, L>>,
{
    _float: std::marker::PhantomData<F>,
    _link: std::marker::PhantomData<L>,
}

impl<F, L> Glm<F> for Linear<F, L>
where
    F: Float + Lapack,
    // L: LinLink,
    L: Link<F, Linear<F, L>>,
{
    type Domain = F;

    fn y_float(y: Self::Domain) -> F {
        y
    }

    // the link function, identity
    fn link(y: F) -> F {
        L::func(y)
    }

    // inverse link function, identity
    fn mean(lin_pred: F) -> F {
        L::inv_func(lin_pred)
    }

    // variance is not a function of the mean in OLS regression.
    fn variance(_mean: F) -> F {
        F::one()
    }

    // This version doesn't have the variances - either setting them to 1 or
    // 1/2pi to simplify the expression. It returns a simple sum of squares.
    // It also misses a factor of 0.5 in the squares.
    fn quasi_log_likelihood(data: &Model<Self, F>, regressors: &Array1<F>) -> F {
        let lin_pred = &data.linear_predictor(&regressors);
        // TODO: transform the linear predictors in the event of a non-identity link function
        let squares: Array1<F> = (&data.y - lin_pred).map(|&d| d * d);
        let l2_term = data.l2_like_term(regressors);
        -squares.sum() + l2_term
    }
}

// /// A trait describing link functions for linear regression.
// pub trait LinLink {
//     fn func<F: Float>(y: F) -> F;
//     fn inv_func<F: Float>(lin_pred: F) -> F;
//     // TODO: parameter transform function, its derivatives, ..., propagate this info to the likelihood
//     // the transformation function that takes the linear predictor to the
//     // canonical parameter. Should always be identify for canonical link
//     // functions.
//     fn canonical<F: Float>(lin_pred: Array1<F>) -> Array1<F>;
// }

/// The identity link function, which is canonical for linear regression.
pub struct Id;
impl Canonical for Id {}
impl<F: Float, M: Glm<F>> Link<F, M> for Id {
    fn func(y: F) -> F {
        y
    }
    fn inv_func(lin_pred: F) -> F {
        lin_pred
    }
}
