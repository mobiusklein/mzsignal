use std::path;
use std::io;
use std::io::prelude;
use plotters::prelude::*;

use mzpeaks::{CoordinateLike, IntensityMeasurement, MZ};

use crate::arrayops;

pub fn peaks_to_arrays<
    'lifespan,
    P: CoordinateLike<MZ> + IntensityMeasurement,
    I: IntoIterator<Item = P>,
>(
    peaks: I,
) -> arrayops::ArrayPair<'lifespan> {
    let mut mz_array: Vec<f64> = Vec::new();
    let mut intensity_array: Vec<f32> = Vec::new();

    for peak in peaks {
        let mz = peak.coordinate();
        let intens = peak.intensity();
        mz_array.push(mz - 0.0001);
        mz_array.push(mz);
        mz_array.push(mz + 0.0001);

        intensity_array.push(0.0);
        intensity_array.push(intens);
        intensity_array.push(0.0);
    }
    (mz_array, intensity_array).into()
}

pub fn draw_svg_file<P>(
    mz_array: &[f64],
    intensity_array: &[f32],
    path: P,
) -> Result<(), Box<dyn std::error::Error>>
where
    P: AsRef<path::Path>,
{
    let backend = SVGBackend::new(&path, (640, 480));
    draw_on_svg(mz_array, intensity_array, backend)
}

pub fn draw_png_file<P>(
    mz_array: &[f64],
    intensity_array: &[f32],
    path: P,
) -> Result<(), Box<dyn std::error::Error>>
where
    P: AsRef<path::Path>,
{
    let backend = BitMapBackend::new(&path, (640, 480));
    draw_on_png(mz_array, intensity_array, backend)
}

pub fn draw_on_png(
    mz_array: &[f64],
    intensity_array: &[f32],
    backend: BitMapBackend,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = backend.into_drawing_area();

    let (xmin, xmax) = arrayops::minmax(mz_array);
    let (_ymin, ymax) = arrayops::minmax(intensity_array);

    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Spectrum", ("sans-serif", 20).into_font())
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(xmin..xmax, 0.0..ymax)?;

    chart
        .configure_mesh()
        .disable_mesh()
        .x_desc("m/z")
        .axis_desc_style(("sans-serif", 16).into_font())
        .y_desc("Intensity")
        .draw()?;

    let points: Vec<(f64, f32)> = mz_array
        .iter()
        .zip(intensity_array.iter())
        .map(|(x, y)| (*x, *y))
        .collect();

    let series = LineSeries::new(
        points.into_iter().map(|xy| xy),
        ShapeStyle {
            color: BLACK.mix(1.0),
            filled: false,
            stroke_width: 1,
        },
    );

    chart.draw_series(series)?;

    Ok(())
}

pub fn draw_on_svg(
    mz_array: &[f64],
    intensity_array: &[f32],
    backend: SVGBackend,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = backend.into_drawing_area();

    let (xmin, xmax) = arrayops::minmax(mz_array);
    let (_ymin, ymax) = arrayops::minmax(intensity_array);

    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Spectrum", ("sans-serif", 20).into_font())
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(xmin..xmax, 0.0..ymax)?;

    chart
        .configure_mesh()
        .disable_mesh()
        .x_desc("m/z")
        .axis_desc_style(("sans-serif", 16).into_font())
        .y_desc("Intensity")
        .draw()?;

    let points: Vec<(f64, f32)> = mz_array
        .iter()
        .zip(intensity_array.iter())
        .map(|(x, y)| (*x, *y))
        .collect();

    let series = LineSeries::new(
        points.into_iter().map(|xy| xy),
        ShapeStyle {
            color: BLACK.mix(1.0),
            filled: false,
            stroke_width: 1,
        },
    );

    chart.draw_series(series)?;

    Ok(())
}

pub enum GraphicsFormat {
    PNG,
    SVG
}


pub fn draw_raw<P: AsRef<path::Path>>(arrays: arrayops::ArrayPair, format: GraphicsFormat, path: P) -> Result<(), Box<dyn std::error::Error>> {
    match format {
        GraphicsFormat::PNG => {
            draw_png_file(&arrays.mz_array, &arrays.intensity_array, path)
        },
        GraphicsFormat::SVG => {
            draw_svg_file(&arrays.mz_array, &arrays.intensity_array, path)
        }
    }
}

pub fn draw_peaks<C: CoordinateLike<MZ> + IntensityMeasurement, I: IntoIterator<Item = C>, P: AsRef<path::Path>>(peaks: I, format: GraphicsFormat, path: P) -> Result<(), Box<dyn std::error::Error>> {
    let pair = peaks_to_arrays(peaks).into();
    draw_raw(pair, format, path)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::arrayops::ArrayPair;
    use crate::average::average_signal;
    use crate::peak_picker::pick_peaks;
    use crate::reprofile::reprofile;
    use crate::test_data::{NOISE, X, Y};
    use crate::text;

    #[test]
    fn test_profile() -> Result<(), Box<dyn std::error::Error>> {
        let yhat: Vec<f32> = Y
            .iter()
            .zip(NOISE.iter())
            .map(|(y, e)| y * 5.0 + e)
            .collect();
        let peaks = pick_peaks(&X, &yhat).unwrap();
        let iterator = peaks.iter();

        let arrays = reprofile(iterator, 0.00001);

        draw_png_file(&arrays.mz_array, &arrays.intensity_array, "test/0.png")?;
        Ok(())
    }
}
