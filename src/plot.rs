use std::borrow::Cow;
use std::io;
use std::io::prelude;
use std::ops::Range;
use std::path;

use plotters::coord::ranged1d::{
    AsRangedCoord, DefaultFormatting, KeyPointHint, NoDefaultFormatting, Ranged, ValueFormatter,
};
use plotters::coord::types::{RangedCoordf32, RangedCoordf64};
use plotters::prelude::*;
pub use plotters::prelude::RED;

use mzpeaks::{CoordinateLike, IntensityMeasurement, MZ};

use crate::arrayops;
use crate::average::rebin;

pub fn peaks_to_arrays<
    'transient,
    'lifespan: 'transient,
    P: CoordinateLike<MZ> + IntensityMeasurement + 'static,
    I: Iterator<Item = &'transient P>,
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

pub struct SciNotRangedCoordf32 {
    pub coord: RangedCoordf32,
}

impl Ranged for SciNotRangedCoordf32 {
    type ValueType = f32;
    type FormatOption = NoDefaultFormatting;

    fn map(&self, value: &Self::ValueType, limit: (i32, i32)) -> i32 {
        self.coord.map(value, limit)
    }

    fn key_points<Hint: KeyPointHint>(&self, hint: Hint) -> Vec<f32> {
        self.coord.key_points(hint)
    }

    fn range(&self) -> Range<f32> {
        self.coord.range()
    }
}

impl ValueFormatter<f32> for SciNotRangedCoordf32 {
    fn format(value: &f32) -> String {
        plotters::data::float::FloatPrettyPrinter {
            allow_scientific: true,
            min_decimal: 0,
            max_decimal: 3,
        }
        .print(*value as f64)
    }
}

pub type CoordinateSpace =
    Cartesian2d<plotters::coord::types::RangedCoordf64, SciNotRangedCoordf32>;

macro_rules! error_t {
    ($t:ty) => {
        plotters::drawing::DrawingAreaErrorKind<<$t as plotters::prelude::DrawingBackend>::ErrorType>
    };
}

#[derive(Default, Clone)]
pub struct SpectrumSeries<'a> {
    pub mz_array: Cow<'a, [f64]>,
    pub intensity_array: Cow<'a, [f32]>,
    pub name: String,
    pub color: Option<plotters::style::RGBAColor>,
    pub width: u32,
    pub downsample: f64,
}

impl<'a> SpectrumSeries<'a> {
    pub fn draw<B: DrawingBackend>(
        &self,
        chart: &mut ChartContext<B, CoordinateSpace>,
    ) -> Result<(), Box<error_t!(B)>> {
        let pair = if self.downsample > 0.0 {
            rebin(&self.mz_array, &self.intensity_array, self.downsample)
        } else {
            arrayops::ArrayPair::new(
                Cow::Borrowed(&self.mz_array),
                Cow::Borrowed(&self.intensity_array),
            )
        };

        let points: Vec<(f64, f32)> = pair
            .mz_array
            .iter()
            .zip(pair.intensity_array.iter())
            .map(|(x, y)| (*x, *y))
            .collect();

        let color = match self.color {
            Some(c) => c,
            None => BLACK.mix(1.0),
        };

        let series = LineSeries::new(
            points.into_iter().map(|xy| xy),
            ShapeStyle {
                color: color,
                filled: false,
                stroke_width: self.width,
            },
        );

        match chart.draw_series(series) {
            Ok(_) => Ok(()),
            Err(err) => Err(Box::new(err)),
        }
    }

    pub fn extrema(&self) -> (f64, f64, f32, f32) {
        let (xmin, xmax) = arrayops::minmax(&self.mz_array);
        let (ymin, ymax) = arrayops::minmax(&self.intensity_array);
        (xmin, xmax, ymin, ymax)
    }

    pub fn color(&mut self, color: plotters::style::RGBAColor) -> &mut Self {
        self.color = Some(color);
        self
    }

    pub fn name(&mut self, name: String) -> &mut Self {
        self.name = name;
        self
    }

    pub fn downsample(&mut self, spacing: f64) -> &mut Self {
        self.downsample = spacing;
        self
    }
}

impl<'a> From<&arrayops::ArrayPair<'a>> for SpectrumSeries<'a> {
    fn from(pair: &arrayops::ArrayPair<'a>) -> SpectrumSeries<'a> {
        let mut inst = SpectrumSeries::default();
        inst.mz_array = match &pair.mz_array {
            Cow::Borrowed(array) => Cow::Borrowed(array),
            Cow::Owned(array) => Cow::Owned(array.clone()),
        };
        inst.intensity_array = match &pair.intensity_array {
            Cow::Borrowed(array) => Cow::Borrowed(array),
            Cow::Owned(array) => Cow::Owned(array.clone()),
        };
        inst.color = None;
        inst.width = 1;
        inst.downsample = 0.0;
        inst
    }
}

impl<
        'b,
        'a: 'b,
        P: CoordinateLike<MZ> + IntensityMeasurement + 'static,
        I: IntoIterator<Item = &'b P>,
    > From<I> for SpectrumSeries<'a>
{
    fn from(iterator: I) -> SpectrumSeries<'a> {
        let mut inst = SpectrumSeries::default();
        let mut mz_array = Vec::new();
        let mut intensity_array = Vec::new();

        for p in iterator {
            let c = p.coordinate();
            mz_array.push(c - 0.0001);
            mz_array.push(c);
            mz_array.push(c + 0.0001);

            intensity_array.push(0.0);
            intensity_array.push(p.intensity());
            intensity_array.push(0.0);
        }

        inst.mz_array = Cow::Owned(mz_array);
        inst.intensity_array = Cow::Owned(intensity_array);
        inst.color = None;
        inst.width = 1;
        inst
    }
}

pub struct SpectrumPlot<'lifespan, 'a, 'b, B: DrawingBackend> {
    pub chart: ChartBuilder<'a, 'b, B>,
    pub series: Vec<SpectrumSeries<'lifespan>>,
    pub xlim: Option<(f64, f64)>,
}

impl<'lifespan, 'a, 'b, B: DrawingBackend> SpectrumPlot<'lifespan, 'a, 'b, B> {
    pub fn new(chart: ChartBuilder<'a, 'b, B>) -> Self {
        Self {
            chart,
            series: Vec::new(),
            xlim: None,
        }
    }

    pub fn add_series<P: Into<SpectrumSeries<'lifespan>>>(&mut self, series: P) -> &mut Self {
        self.series.push(series.into());
        self
    }

    pub fn xlim(&mut self, xlow: f64, xhigh: f64) -> &mut Self {
        self.xlim = Some((xlow, xhigh));
        self
    }

    pub fn make_coordinate_ranges(&self) -> (RangedCoordf64, SciNotRangedCoordf32) {
        let mut xmin = f64::INFINITY;
        let mut xmax = 0f64;
        let mut ymax = 0f32;

        for series in self.series.iter() {
            let (s_xmin, s_xmax, _ymin, s_ymax) = series.extrema();
            xmin = if s_xmin < xmin { s_xmin } else { xmin };
            xmax = if s_xmax > xmax { s_xmax } else { xmax };
            ymax = if s_ymax > ymax { s_ymax } else { ymax };
        }
        if let Some(xlim) = self.xlim {
            xmin = xlim.0;
            xmax = xlim.1;
        }
        let xrange = RangedCoordf64::from(xmin..xmax);
        let yrange = SciNotRangedCoordf32 {
            coord: RangedCoordf32::from(0.0..ymax),
        };
        (xrange, yrange)
    }

    pub fn draw(&mut self) -> Result<(), Box<error_t!(B)>> {
        let (xrange, yrange) = self.make_coordinate_ranges();
        let mut chart = self
            .chart
            .margin(15)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(xrange, yrange)?;

        let mut mesh = chart.configure_mesh();
        mesh.disable_mesh();
        mesh.x_desc("m/z");
        mesh.axis_desc_style(("sans-serif", 16).into_font());
        mesh.y_desc("Intensity");
        mesh.draw()?;

        for series in self.series.iter() {
            series.draw(&mut chart)?;
        }
        Ok(())
    }
}

pub trait PlotBuilder<'a> {
    type BackendType: DrawingBackend;

    fn size(&mut self, width: u32, height: u32) -> &mut Self;
    fn path<P: Into<path::PathBuf>>(&mut self, path: P) -> &mut Self;
    fn add_series<S: Into<SpectrumSeries<'a>>>(&mut self, series: S) -> &mut Self;
    fn xlim(&mut self, xlow: f64, xhigh: f64) -> &mut Self;
    fn draw(&mut self) -> Result<(), Box<error_t!(Self::BackendType)>>;
}

pub struct SVGBuilder<'lifespan> {
    pub size: (u32, u32),
    pub path: path::PathBuf,
    pub series: Vec<SpectrumSeries<'lifespan>>,
    pub xlim: Option<(f64, f64)>,
}

impl<'lifespan> Default for SVGBuilder<'lifespan> {
    fn default() -> SVGBuilder<'lifespan> {
        SVGBuilder {
            size: (640, 480),
            path: path::PathBuf::default(),
            series: Vec::new(),
            xlim: None,
        }
    }
}

impl<'lifespan> PlotBuilder<'lifespan> for SVGBuilder<'lifespan> {
    type BackendType = SVGBackend<'lifespan>;

    fn size(&mut self, width: u32, height: u32) -> &mut Self {
        self.size = (width, height);
        self
    }

    fn path<P: Into<path::PathBuf>>(&mut self, path: P) -> &mut Self {
        self.path = path.into();
        self
    }

    fn add_series<S: Into<SpectrumSeries<'lifespan>>>(&mut self, series: S) -> &mut Self {
        self.series.push(series.into());
        self
    }

    fn xlim(&mut self, xlow: f64, xhigh: f64) -> &mut Self {
        self.xlim = Some((xlow, xhigh));
        self
    }

    fn draw(&mut self) -> Result<(), Box<error_t!(Self::BackendType)>> {
        let backend = SVGBackend::new(&self.path, self.size);
        let root = backend.into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = SpectrumPlot::new(ChartBuilder::on(&root));
        if let Some(xlim) = self.xlim {
            chart.xlim(xlim.0, xlim.1);
        }
        for series in self.series.iter().cloned() {
            chart.add_series(series);
        }

        chart.draw()
    }
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

    let xrange = RangedCoordf64::from(xmin..xmax);
    let yrange = SciNotRangedCoordf32 {
        coord: RangedCoordf32::from(0.0..ymax),
    };

    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Spectrum", ("sans-serif", 20).into_font())
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(xrange, yrange)?;

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

    let xrange = RangedCoordf64::from(xmin..xmax);
    let yrange = SciNotRangedCoordf32 {
        coord: RangedCoordf32::from(0.0..ymax),
    };

    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Spectrum", ("sans-serif", 20).into_font())
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(xrange, yrange)?;

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
    SVG,
}

pub fn draw_raw<P: AsRef<path::Path>>(
    arrays: arrayops::ArrayPair,
    format: GraphicsFormat,
    path: P,
) -> Result<(), Box<dyn std::error::Error>> {
    match format {
        GraphicsFormat::PNG => draw_png_file(&arrays.mz_array, &arrays.intensity_array, path),
        GraphicsFormat::SVG => draw_svg_file(&arrays.mz_array, &arrays.intensity_array, path),
    }
}

pub fn draw_peaks<
    'a,
    C: CoordinateLike<MZ> + IntensityMeasurement + 'static,
    I: Iterator<Item = &'a C>,
    P: AsRef<path::Path>,
>(
    peaks: I,
    format: GraphicsFormat,
    path: P,
) -> Result<(), Box<dyn std::error::Error>> {
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

    #[test]
    fn test_builder() -> Result<(), Box<dyn std::error::Error>> {
        let yhat: Vec<f32> = Y
            .iter()
            .zip(NOISE.iter())
            .map(|(y, e)| y * 50.0 + e * 20.0)
            .collect();
        let peaks = pick_peaks(&X, &yhat).unwrap();
        let iterator = peaks.iter();

        let arrays = reprofile(iterator, 0.001);
        let mut chart = SVGBuilder::default();
        let mut ser = SpectrumSeries::from(peaks.iter());
        ser.color(RED.mix(1.0));
        chart
            .path("test/0.svg")
            .size(1028, 512)
            .add_series(SpectrumSeries::from(&arrays))
            .add_series(ser);
        chart.xlim(179.0, 181.0);
        chart.draw()?;
        Ok(())
    }
}
