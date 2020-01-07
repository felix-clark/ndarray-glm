//! library for solving GLM regression
//! TODO: documentation

// extern crate ndarray_linalg;
// this line is necessary to avoid linking errors
// but maybe it should only go into final library
// extern crate openblas_src;

#[macro_use(array)]
extern crate ndarray;

pub mod linear;
pub mod utility;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let data_y = array![0.1, 0.2];
        let data_x = array![[1.2, 1.3], [1.9, 2.0]];
        eprintln!("{}", linear::regression(&data_y, &data_x));
        assert_eq!(2 + 2, 4);
    }
}
