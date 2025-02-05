use std::{io, env};

use mzsignal::{text::{arrays_from_file, arrays_to_writer}, average};


fn main() -> io::Result<()> {
    let path = env::args().nth(1).unwrap();
    let dx: f64 = env::args().nth(2).unwrap_or_else(|| "0.002".to_string()).parse().unwrap();

    let arrays = arrays_from_file(&path)?;

    let rebinned = average::rebin(&arrays.mz_array, &arrays.intensity_array, dx);

    arrays_to_writer(rebinned,  &mut io::stdout().lock())?;
    Ok(())
}