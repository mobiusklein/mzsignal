use std::io::prelude::*;
use std::io;
use std::path;
use std::fs;

use crate::arrayops::ArrayPair;

pub fn to_file<P: AsRef<path::Path>>(arrays: ArrayPair<'_>, path: P) -> io::Result<()> {
    let file = fs::File::create(path)?;
    let mut writer = io::BufWriter::new(file);
    let n = arrays.len();
    for i in 0..n {
        let pt = arrays.get(i).unwrap();
        writer.write_all(format!("{}\t{}\n", pt.0, pt.1).as_bytes())?;
    }
    Ok(())
}