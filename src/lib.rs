#[macro_use(array)]
extern crate ndarray;

pub mod linear;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn it_works() {
        let data_y = array![0.1, 0.2];
        let data_x = array![[1.2, 1.3], [1.9, 2.0]];
        eprintln!("{}", linear::regression(&data_y, &data_x));
        assert_eq!(2 + 2, 4);
    }
}
