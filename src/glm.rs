//! trait defining a generalized linear model and providing common functionality
//! Models are fit such that E[Y] = g^-1(X*B) where g is the link function.

use num_traits::Float;

pub trait Glm {
    // the domain of the model
    // i.e. integer for Poisson, float for Linear, bool for logistic
    // TODO: perhaps create a custom Domain type or trait to deal with constraints
    // we typically work with floats as EVs, though.
    type Domain;

    /// a function to check if a Y-value is value

    /// the link function
    // fn link<F: 'static + Float>(y: Self::Domain) -> F;
    fn link<F: Float>(y: F) -> F;

    /// inverse link function which maps the linear predictors to the expected value of the prediction.
    fn mean<F: Float>(x: F) -> F;

    /// the variance as a function of the mean
    fn variance<F: Float>(mean: F) -> F;

    // /// logarithm of the likelihood given the data and fit parameters
    // fn log_likelihood<F: 'static + Float>(
    //     data_y: &Array1<Self::Domain>,
    //     data_x: &Array2<F>,
    //     regressors: &Array1<F>,
    // ) -> F;
}
