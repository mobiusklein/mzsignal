//! Helpers for loading test data

use std::fs;
use std::io;
use std::io::prelude::*;
use std::path;

use crate::arrayops::{ArrayPair, ArrayPairLike};

/// A helper function to write an [`ArrayPair`] to a file on disk as plain text
/// The written format is a tab-separated file denoting an m/z intensity pair,
/// with one pair per line.
/// See [`arrays_to_writer`]
pub fn arrays_to_file<P: AsRef<path::Path>, A: ArrayPairLike>(
    arrays: A,
    path: P,
) -> io::Result<()> {
    let file = fs::File::create(path)?;
    let mut writer = io::BufWriter::new(file);
    arrays_to_writer(arrays, &mut writer)
}

/// A helper function to write an [`ArrayPair`] to a [`Write`] as plain text
/// The written format is a tab-separated file denoting an m/z intensity pair,
/// with one pair per line.
pub fn arrays_to_writer<W: io::Write, A: ArrayPairLike>(
    arrays: A,
    writer: &mut W,
) -> io::Result<()> {
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

pub fn arrays_over_time_from_reader<'a, R: io::Read>(
    source: R,
) -> io::Result<Vec<(f64, ArrayPair<'a>)>> {
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
                acc.push((time, ArrayPair::from((mz_array, intensity_array))));
                mz_array = Vec::new();
                intensity_array = Vec::new();
            }
            time = t;
        }
        mz_array.push(mz);
        intensity_array.push(inten);
    }

    if !mz_array.is_empty() {
        acc.push((time, ArrayPair::from((mz_array, intensity_array))));
    }
    Ok(acc)
}

/// A helper function to read an [`ArrayPair`] from a file on disk.
/// The expected format is a tab-separated file denoting an m/z intensity pair,
/// with one pair per line.
pub fn arrays_over_time_from_file<'a, P: AsRef<path::Path>>(
    path: P,
) -> io::Result<Vec<(f64, ArrayPair<'a>)>> {
    let reader = fs::File::open(path)?;
    arrays_over_time_from_reader(reader)
}

pub fn load_feature_table<P: AsRef<path::Path>>(
    path: P,
) -> io::Result<Vec<mzpeaks::feature::Feature<mzpeaks::MZ, mzpeaks::Time>>> {
    let fh = io::BufReader::new(fs::File::open(path)?);

    let mut features: Vec<mzpeaks::feature::Feature<mzpeaks::MZ, mzpeaks::Time>> = Vec::new();
    let mut feature: mzpeaks::feature::Feature<mzpeaks::MZ, mzpeaks::Time> =
        mzpeaks::feature::Feature::empty();
    let mut last_id = 0;
    for (_i, line) in fh.lines().enumerate().skip(1) {
        let line_raw = line?;
        let line = line_raw.trim();
        if line.is_empty() {
            continue;
        }
        // eprintln!("{i}: {line}");
        let mut tokenizer = line.split("\t");

        let feat_id = tokenizer.next().map(|s| s.parse::<i32>().unwrap()).unwrap();
        let mz = tokenizer.next().map(|s| s.parse::<f64>().unwrap()).unwrap();
        let rt = tokenizer.next().map(|s| s.parse::<f64>().unwrap()).unwrap();
        let inten = tokenizer.next().map(|s| s.parse::<f32>().unwrap()).unwrap();

        if feat_id != last_id {
            features.push(feature);
            feature = Default::default();
            last_id = feat_id;
        }
        feature.push_raw(mz, rt, inten);
    }
    features.push(feature);
    Ok(features)
}

pub fn write_feature_table<
    'a,
    P: AsRef<path::Path>,
    X: mzpeaks::coordinate::CoordinateSystem + 'a,
    Y: mzpeaks::coordinate::CoordinateSystem + 'a,
>(
    path: P,
    features: impl Iterator<Item=&'a mzpeaks::feature::Feature<X, Y>>,
) -> io::Result<()> {
    let mut writer = io::BufWriter::new(fs::File::create(path)?);

    let header = ["feature_id", X::name(), Y::name(), "intensity"].join("\t");
    writeln!(writer, "{}", header)?;
    for (i, f) in features.into_iter().enumerate() {
        for (x, y, inten) in f.iter() {
            writer.write_all(format!("{i}\t{x}\t{y}\t{inten}\n").as_bytes())?;
        }
    }
    Ok(())
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
