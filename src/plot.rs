use std::path;

use plotters::prelude::*;
use plotters::style::RGBAColor;

use crate::arrayops;

pub fn draw_svg_file<P>(mz_array: &[f64], intensity_array: &[f32], path: P) -> Result<(), Box<dyn std::error::Error>> where P: AsRef<path::Path> {
    let backend = SVGBackend::new(&path, (640, 480));
    draw_on_svg(mz_array, intensity_array, backend)
}


pub fn draw_png_file<P>(mz_array: &[f64], intensity_array: &[f32], path: P) -> Result<(), Box<dyn std::error::Error>> where P: AsRef<path::Path> {
    let backend = BitMapBackend::new(&path, (640, 480));
    draw_on_png(mz_array, intensity_array, backend)
}


pub fn draw_on_png(mz_array: &[f64], intensity_array: &[f32], backend: BitMapBackend) -> Result<(), Box<dyn std::error::Error>> {
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

    chart.configure_mesh()
        .disable_mesh()
        .x_desc("m/z").axis_desc_style(("sans-serif", 16).into_font())
        .y_desc("Intensity")
        .draw()?;

    let points: Vec<(f64, f32)> = mz_array.iter().zip(intensity_array.iter()).map(|(x, y)| (*x, *y)).collect();

    let series = LineSeries::new(points.into_iter().map(|xy| xy), ShapeStyle {
        color: BLACK.mix(1.0),
        filled: false,
        stroke_width: 1
    });

    chart.draw_series(series)?;

    Ok(())
}

pub fn draw_on_svg(mz_array: &[f64], intensity_array: &[f32], backend: SVGBackend) -> Result<(), Box<dyn std::error::Error>> {
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

    chart.configure_mesh()
        .disable_mesh()
        .x_desc("m/z").axis_desc_style(("sans-serif", 16).into_font())
        .y_desc("Intensity")
        .draw()?;

    let points: Vec<(f64, f32)> = mz_array.iter().zip(intensity_array.iter()).map(|(x, y)| (*x, *y)).collect();

    let series = LineSeries::new(points.into_iter().map(|xy| xy), ShapeStyle {
        color: BLACK.mix(1.0),
        filled: false,
        stroke_width: 1
    });

    chart.draw_series(series)?;

    Ok(())
}


#[cfg(test)]
mod test {
    use super::*;
    use crate::test_data::{NOISE, X, Y};
    use crate::arrayops::ArrayPair;
    use crate::peak_picker::pick_peaks;
    use crate::reprofile::reprofile;
    use crate::average::average_signal;
    use crate::text;

    #[test]
    fn test_profile() -> Result<(), Box<dyn std::error::Error>> {
        let yhat: Vec<f32> = Y
            .iter()
            .zip(NOISE.iter())
            .map(|(y, e)| y * 5.0 + e).collect();
        let peaks = pick_peaks(&X, &yhat).unwrap();
        let iterator = peaks.iter();

        let arrays = reprofile(iterator, 0.00001);

        draw_png_file(&arrays.mz_array, &arrays.intensity_array, "test/0.png")?;
        text::to_file(arrays, "test/0.txt")?;
        Ok(())
    }
}