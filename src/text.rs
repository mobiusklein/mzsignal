//! Helpers for loading test data

use std::fs;
use std::io;
use std::io::prelude::*;
use std::path;

use crate::arrayops::ArrayPair;

/// A helper function to write an [`ArrayPair`] to a file on disk as plain text
/// The written format is a tab-separated file denoting an m/z intensity pair,
/// with one pair per line.
/// See [`arrays_to_writer`]
pub fn arrays_to_file<P: AsRef<path::Path>>(arrays: ArrayPair<'_>, path: P) -> io::Result<()> {
    let file = fs::File::create(path)?;
    let mut writer = io::BufWriter::new(file);
    arrays_to_writer(arrays, &mut writer)
}


/// A helper function to write an [`ArrayPair`] to a [`Write`] as plain text
/// The written format is a tab-separated file denoting an m/z intensity pair,
/// with one pair per line.
pub fn arrays_to_writer<W: io::Write>(arrays: ArrayPair<'_>, writer: &mut W) -> io::Result<()> {
    let n = arrays.len();
    for i in 0..n {
        let pt = arrays.get(i).unwrap();
        writer.write_all(format!("{}\t{}\n", pt.0, pt.1).as_bytes())?;
    }
    Ok(())
}


/// A helper function to read an [`ArrayPair`] from a [`Read`] type.
/// The expected format is a tab-separated file denoting an m/z intensity pair,
/// with one pair per line.
pub fn arrays_from_reader<'a, R: io::Read>(source: R) -> io::Result<ArrayPair<'a>> {
    let reader = io::BufReader::new(source);
    let mut mz_array: Vec<f64> = Vec::new();
    let mut intensity_array: Vec<f32> = Vec::new();

    for line in reader.lines() {
        let line = line.unwrap();
        let pref = line.trim();
        let chunks: Vec<&str> = pref.split('\t').collect();
        mz_array.push(chunks[0].parse::<f64>().expect("Expected number for m/z"));
        intensity_array.push(
            chunks[1]
                .parse::<f32>()
                .expect("Expected number for intensity"),
        );
    }
    Ok(ArrayPair::from((mz_array, intensity_array)))
}


/// A helper function to read an [`ArrayPair`] from a file on disk.
/// The expected format is a tab-separated file denoting an m/z intensity pair,
/// with one pair per line.
/// See [`arrays_from_reader`]
pub fn arrays_from_file<'a, P: AsRef<path::Path>>(path: P) -> io::Result<ArrayPair<'a>> {
    let reader = fs::File::open(path)?;
    arrays_from_reader(reader)
}


pub fn arrays_over_time_from_reader<'a, R: io::Read>(source: R) -> io::Result<Vec<(f64, ArrayPair<'a>)>> {
    let reader = io::BufReader::new(source);
    let mut mz_array: Vec<f64> = Vec::new();
    let mut intensity_array: Vec<f32> = Vec::new();
    let mut time: f64 = f64::NEG_INFINITY;

    let mut acc = Vec::new();

    for line in reader.lines() {
        let line = line.unwrap();
        let pref = line.trim();
        let chunks: Vec<&str> = pref.split('\t').collect();
        let mz = chunks[0].parse::<f64>().expect("Expected number for m/z");
        let inten = chunks[1]
                .parse::<f32>()
                .expect("Expected number for intensity");
        let t = chunks[2].parse::<f64>().expect("Expected number for time");

        if t != time {
            if !mz_array.is_empty() {
                acc.push((
                    time,
                    ArrayPair::from((mz_array, intensity_array))
                ));
                mz_array = Vec::new();
                intensity_array = Vec::new();
            }
            time = t;
        }
        mz_array.push(mz);
        intensity_array.push(inten);
    }

    if !mz_array.is_empty() {
        acc.push((
            time,
            ArrayPair::from((mz_array, intensity_array))
        ));
    }
    Ok(acc)
}


/// A helper function to read an [`ArrayPair`] from a file on disk.
/// The expected format is a tab-separated file denoting an m/z intensity pair,
/// with one pair per line.
pub fn arrays_over_time_from_file<'a, P: AsRef<path::Path>>(path: P) -> io::Result<Vec<(f64, ArrayPair<'a>)>> {
    let reader = fs::File::open(path)?;
    arrays_over_time_from_reader(reader)
}


#[cfg(test)]
mod test {
    use super::*;


    #[test]
    fn read_arrays_over_time() -> io::Result<()> {
        let time_arrays = arrays_over_time_from_file("./test/data/peaks_over_time.txt")?;
        assert_eq!(187, time_arrays.len());
        Ok(())
    }
}