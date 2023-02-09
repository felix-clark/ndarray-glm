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

/// Read y, x, and linear offsets from a CSV. Right now it's assumed that there is only one covariate.
#[cfg(test)]
#[allow(dead_code)]
pub fn y_x_off_from_csv<Y, X>(file: &str) -> Result<(Array1<Y>, Array2<X>, Array1<X>)>
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
    for line_result in reader.lines() {
        let line = line_result?;
        let split_line: Vec<&str> = line.split(',').collect();
        if split_line.len() != 3 {
            return Err(anyhow!("Expected three entries in CSV"));
        }
        let y_parsed: Y = split_line[0].parse()?;
        let x_parsed: X = split_line[1].parse()?;
        let off_parsed: X = split_line[2].parse()?;
        y_vec.push(y_parsed);
        x_vec.push(x_parsed);
        off_vec.push(off_parsed);
    }
    let y = Array1::<Y>::from(y_vec);
    let x = Array2::<X>::from_shape_vec((y.len(), 1), x_vec)?;
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
        for i in 0..4 {
            let x_val: f32 = split_line[i].parse()?;
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
