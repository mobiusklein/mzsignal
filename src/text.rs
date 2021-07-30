use std::io::prelude::*;
use std::io;
use std::path;
use std::fs;

use crate::arrayops::ArrayPair;

pub fn arrays_to_file<P: AsRef<path::Path>>(arrays: ArrayPair<'_>, path: P) -> io::Result<()> {
    let file = fs::File::create(path)?;
    let mut writer = io::BufWriter::new(file);
    let n = arrays.len();
    for i in 0..n {
        let pt = arrays.get(i).unwrap();
        writer.write_all(format!("{}\t{}\n", pt.0, pt.1).as_bytes())?;
    }
    Ok(())
}


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


pub fn arrays_from_file<'a, P: AsRef<path::Path>>(path: P) -> io::Result<ArrayPair<'a>> {
    let reader = fs::File::open(path)?;
    arrays_from_reader(reader)
}