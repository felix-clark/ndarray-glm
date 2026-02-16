//! Utility functions for testing
use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2};
use num_traits::Float;
use std::{
    error::Error,
    fs::File,
    io::{BufRead, BufReader},
    str::FromStr,
};

/// Read y, x pairs from a CSV. Right now it's assumed that there is only one covariate.
// This function isn't used yet, but it will be.
#[cfg(test)]
#[allow(dead_code)]
pub fn y_x_from_csv<Y, X>(file: &str) -> Result<(Array1<Y>, Array2<X>)>
where
    Y: FromStr,
    X: Float + FromStr,
    <Y as FromStr>::Err: 'static + Error + Send + Sync,
    <X as FromStr>::Err: 'static + Error + Send + Sync,
{
    let file = File::open(file)?;
    let reader = BufReader::new(file);
    let mut y_vec: Vec<Y> = Vec::new();
    let mut x_vec: Vec<X> = Vec::new();
    for line_result in reader.lines() {
        let line = line_result?;
        let split_line: Vec<&str> = line.split(',').collect();
        if split_line.len() != 2 {
            return Err(anyhow!("Expected two entries in CSV"));
        }
        let y_parsed: Y = split_line[0].parse()?;
        let x_parsed: X = split_line[1].parse()?;
        y_vec.push(y_parsed);
        x_vec.push(x_parsed);
    }
    let y = Array1::<Y>::from(y_vec);
    let x = Array2::<X>::from_shape_vec((y.len(), 1), x_vec)?;
    Ok((y, x))
}

/// Read y, x, and linear offsets from a CSV. NX indicates the number of X columns.
#[cfg(test)]
#[allow(dead_code)]
pub fn y_x_off_from_csv<Y, X, const NX: usize>(
    file: &str,
) -> Result<(Array1<Y>, Array2<X>, Array1<X>)>
where
    Y: FromStr,
    X: Float + FromStr,
    <Y as FromStr>::Err: 'static + Error + Send + Sync,
    <X as FromStr>::Err: 'static + Error + Send + Sync,
{
    let file = File::open(file)?;
    let reader = BufReader::new(file);
    let mut y_vec: Vec<Y> = Vec::new();
    let mut x_vec: Vec<X> = Vec::new();
    let mut off_vec: Vec<X> = Vec::new();
    let expected_cols = 2 + NX;
    for line_result in reader.lines() {
        let line = line_result?;
        let split_line: Vec<&str> = line.split(',').collect();
        let n_cols = split_line.len();
        if n_cols != expected_cols {
            return Err(anyhow!("Expected {expected_cols} entries in CSV"));
        }
        let y_parsed: Y = split_line[0].parse()?;
        y_vec.push(y_parsed);
        for x_str in split_line.iter().take(NX + 1).skip(1) {
            let x_parsed: X = x_str.parse()?;
            x_vec.push(x_parsed);
        }
        let off_parsed: X = split_line[NX + 1].parse()?;
        off_vec.push(off_parsed);
    }
    let y = Array1::<Y>::from(y_vec);
    let x = Array2::<X>::from_shape_vec((y.len(), x_vec.len() / y.len()), x_vec)?;
    let off = Array1::<X>::from(off_vec);
    Ok((y, x, off))
}

/// Read a flat array from a text file
#[cfg(test)]
// Silence an false warning about non-use
#[allow(dead_code)]
pub fn array_from_csv<X>(file: &str) -> Result<Array1<X>>
where
    X: Float + FromStr,
    <X as FromStr>::Err: 'static + Error + Send + Sync,
{
    let file = File::open(file)?;
    let reader = BufReader::new(file);
    let mut x_vec: Vec<X> = Vec::new();
    for line_result in reader.lines() {
        let line = line_result?;
        let x_parsed: X = line.parse()?;
        x_vec.push(x_parsed);
    }
    let x: Array1<X> = x_vec.into();
    Ok(x)
}

/// Load the linear_weights dataset: y, x1, x2, x3, var_wt, freq_wt (with header row)
#[cfg(test)]
#[allow(dead_code)]
#[allow(clippy::type_complexity)]
pub fn load_linear_weights_data() -> Result<(Array1<f64>, Array2<f64>, Array1<f64>, Array1<usize>)>
{
    let file = File::open("tests/data/linear_weights.csv")?;
    let reader = BufReader::new(file);
    let mut y_vec = Vec::new();
    let mut x_vec = Vec::new();
    let mut var_wt_vec = Vec::new();
    let mut freq_wt_vec = Vec::new();
    for (i, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        if i == 0 {
            continue; // skip header
        }
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() != 6 {
            return Err(anyhow!("Expected 6 columns in linear_weights.csv"));
        }
        y_vec.push(cols[0].parse::<f64>()?);
        x_vec.push(cols[1].parse::<f64>()?);
        x_vec.push(cols[2].parse::<f64>()?);
        x_vec.push(cols[3].parse::<f64>()?);
        var_wt_vec.push(cols[4].parse::<f64>()?);
        freq_wt_vec.push(cols[5].parse::<f64>()? as usize);
    }
    let n = y_vec.len();
    let y = Array1::from(y_vec);
    let x = Array2::from_shape_vec((n, 3), x_vec)?;
    let var_wt = Array1::from(var_wt_vec);
    let freq_wt = Array1::from(freq_wt_vec);
    Ok((y, x, var_wt, freq_wt))
}

/// Load data from the popular iris test dataset.
/// The class will be encoded as an integer in the y data.
#[allow(dead_code)]
pub fn y_x_from_iris() -> Result<(Array1<i32>, Array2<f32>)> {
    let file = File::open("tests/data/iris.csv")?;
    let reader = BufReader::new(file);
    let mut y_vec: Vec<i32> = Vec::new();
    let mut x_vec: Vec<f32> = Vec::new();
    for line_result in reader.lines() {
        let line = line_result?;
        if line == "sepal_length,sepal_width,petal_length,petal_width,class" {
            continue;
        }
        let split_line: Vec<&str> = line.split(',').collect();
        if split_line.len() != 5 {
            return Err(anyhow!("Expected five entries in CSV"));
        }
        for x_str in split_line.iter().take(4) {
            let x_val = x_str.parse()?;
            x_vec.push(x_val);
        }
        let y_parsed = match split_line[4] {
            "setosa" => 0,
            "versicolor" => 1,
            "virginica" => 2,
            _ => unreachable!("There should only be 3 classes of irises"),
        };
        y_vec.push(y_parsed);
    }
    let y = Array1::<i32>::from(y_vec);
    let x = Array2::<f32>::from_shape_vec((y.len(), 4), x_vec)?;
    Ok((y, x))
}
