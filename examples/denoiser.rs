use std::{io, env};

use mzsignal::{text::{arrays_from_file, arrays_to_writer}, denoise};


fn main() -> io::Result<()> {
    let path = env::args().skip(1).next().unwrap();
    let factor: f32 = env::args().skip(2).next().unwrap_or_else(|| "1".to_string()).parse().unwrap();

    let mut arrays = arrays_from_file(&path)?;
    let mut intensities = arrays.intensity_array.to_vec();

    denoise::denoise(&arrays.mz_array, &mut intensities, factor).unwrap();

    let diff = arrays.intensity_array.iter().zip(intensities.iter()).map(|(a, b)| (*a - *b)).sum::<f32>();
    let total = arrays.intensity_array.iter().copied().sum::<f32>();
    eprintln!("Difference: {diff}, Total: {total}, Ratio {}", diff / total);

    arrays.intensity_array = intensities.into();
    arrays_to_writer(arrays,  &mut io::stdout().lock())?;
    Ok(())
}