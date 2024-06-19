//! Convert picked peaks into a profile spectrum.
//!
use std::borrow;
use std::borrow::Cow;
use std::cmp;
use std::iter;

use mzpeaks::{
    CentroidLike, CoordinateLike, IndexType, IndexedCoordinate, IntensityMeasurement, MZ,
};

use crate::arrayops::{gridspace, trapz, ArrayPair, MZGrid};
use crate::peak::FittedPeak;

#[derive(Debug, Clone, Copy)]
/// A statistical model for peak shapes
pub enum PeakShape {
    Gaussian,
}

#[derive(Debug, Clone)]
/// A model for predicting the signal shape given a fitted peak as a set
/// of model parameters
pub struct PeakShapeModel<'lifespan> {
    pub peak: Cow<'lifespan, FittedPeak>,
    pub shape: PeakShape,
}

impl<'lifespan> CoordinateLike<MZ> for PeakShapeModel<'lifespan> {
    fn coordinate(&self) -> f64 {
        self.peak.coordinate()
    }
}

impl<'lifespan> IntensityMeasurement for PeakShapeModel<'lifespan> {
    fn intensity(&self) -> f32 {
        self.peak.intensity()
    }
}

impl<'lifespan> IndexedCoordinate<MZ> for PeakShapeModel<'lifespan> {
    fn get_index(&self) -> IndexType {
        self.peak.get_index()
    }

    fn set_index(&mut self, _index: IndexType) {}
}

impl<'lifespan> PartialEq<PeakShapeModel<'lifespan>> for PeakShapeModel<'lifespan> {
    fn eq(&self, other: &PeakShapeModel<'lifespan>) -> bool {
        (self.peak.mz - other.peak.mz).abs() < 1e-6
            && (self.peak.intensity - other.peak.intensity).abs() < 1e-6
    }
}

impl<'a> Eq for PeakShapeModel<'a> {}

impl<'lifespan> PartialOrd<PeakShapeModel<'lifespan>> for PeakShapeModel<'lifespan> {
    fn partial_cmp(&self, other: &PeakShapeModel<'lifespan>) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<'lifespan> Ord for PeakShapeModel<'lifespan> {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.peak.mz.partial_cmp(&other.peak.mz).unwrap()
    }
}

impl<'lifespan> PeakShapeModel<'lifespan> {
    pub fn from_centroid(
        mz: f64,
        intensity: f32,
        full_width_at_half_max: f32,
        shape: PeakShape,
    ) -> PeakShapeModel<'lifespan> {
        PeakShapeModel {
            peak: Cow::Owned(FittedPeak {
                mz,
                intensity,
                full_width_at_half_max,
                ..FittedPeak::default()
            }),
            shape,
        }
    }

    pub fn new(peak: Cow<'lifespan, FittedPeak>, shape: PeakShape) -> Self {
        Self { peak, shape }
    }

    /// Create a [`PeakShape::Gaussian`] [`PeakShapeModel`]
    pub fn gaussian(peak: &FittedPeak) -> PeakShapeModel {
        PeakShapeModel {
            peak: Cow::Borrowed(peak),
            shape: PeakShape::Gaussian,
        }
    }

    /// Estimate the intensity of this peak at `mz`, relative to the
    /// position of the model peak
    pub fn predict(&self, mz: &f64) -> f32 {
        match self.shape {
            PeakShape::Gaussian => {
                let spread = self.peak.full_width_at_half_max / 2.35482;
                let scaler = (-(f64::powf(mz - self.peak.mz, 2.0))
                    / (2.0 * f64::powf(spread as f64, 2.0)))
                .exp();
                self.peak.intensity * scaler as f32
            }
        }
    }

    /// Generate a theoretical peak shape signal with m/z and intensity arrays
    pub fn shape(&self, dx: f64) -> (Vec<f64>, Vec<f32>) {
        let (start, end) = self.extremes();
        let mz_array = gridspace(start, end, dx);
        let intensity_array = mz_array.iter().map(|x| self.predict(x)).collect();
        (mz_array, intensity_array)
    }

    /// Generate a theoretical peak shape signal with m/z arrays in `mz_array`
    /// and adds the theoretical intensity to `intensity_array`
    pub fn shape_in(&self, mz_array: &[f64], intensity_array: &mut [f32]) {
        for (i, val) in mz_array.iter().map(|x| self.predict(x)).enumerate() {
            intensity_array[i] += val;
        }
    }

    /// Calculate the area of the peak shape estimated with an m/z spacing of `dx`
    pub fn area(&self, dx: f64) -> f32 {
        let (x, y) = self.shape(dx);
        trapz(&x, &y)
    }

    pub fn center(&self) -> f64 {
        self.peak.mz
    }

    /// Approximate the lower and upper m/zs at which this peak no longer detectable
    pub fn extremes(&self) -> (f64, f64) {
        (
            self.peak.mz - self.peak.full_width_at_half_max as f64 - 0.02,
            self.peak.mz + self.peak.full_width_at_half_max as f64 + 0.02,
        )
    }
}

impl<'lifespan> From<&'lifespan FittedPeak> for PeakShapeModel<'lifespan> {
    fn from(peak: &'lifespan FittedPeak) -> PeakShapeModel<'lifespan> {
        PeakShapeModel::gaussian(peak)
    }
}

impl From<FittedPeak> for PeakShapeModel<'static> {
    fn from(value: FittedPeak) -> Self {
        PeakShapeModel {
            peak: Cow::Owned(value),
            shape: PeakShape::Gaussian,
        }
    }
}

/// Convert something into a [`PeakShapeModel`] with a given width parameter
pub trait AsPeakShapeModel<'a, 'b: 'a> {
    /// Convert something into a [`PeakShapeModel`] with a given width parameter `fwhm`
    /// and a specific [`PeakShape`]
    fn as_peak_shape_model(&'b self, fwhm: f32, shape: PeakShape) -> PeakShapeModel<'a>;
}

impl<'a, 'b: 'a, T: CentroidLike> AsPeakShapeModel<'a, 'b> for &T {
    fn as_peak_shape_model(&'b self, fwhm: f32, shape: PeakShape) -> PeakShapeModel<'a> {
        PeakShapeModel::from_centroid(self.coordinate(), self.intensity(), fwhm, shape)
    }
}

/// A probabilistic peak shape re-construction spectrum intensity averager over a
/// shared m/z axis.
#[derive(Debug, Default, Clone)]
pub struct PeakSetReprofiler {
    /// The evenly spaced m/z axis over which peaks are re-estimated
    pub mz_grid: Vec<f64>,
    /// The lowest m/z in the spectrum. If an input spectrum has lower m/z values, they will be ignored.
    pub mz_start: f64,
    /// The highest m/z in the spectrum. If an input spectrum has higher m/z values, they will be ignored.
    pub mz_end: f64,
}

impl<'passing, 'transient: 'passing, 'lifespan: 'transient> PeakSetReprofiler {
    pub fn new(mz_start: f64, mz_end: f64, dx: f64) -> PeakSetReprofiler {
        PeakSetReprofiler {
            mz_grid: gridspace(mz_start, mz_end, dx),
            mz_start,
            mz_end,
        }
    }

    /// Create an array of [`PeakShapeModel`]s from an array of structs that can convert
    /// into them, using `shape` for the type of peak shape.
    pub fn build_peak_shape_models<T>(
        &self,
        peaks: &'lifespan [T],
        shape: PeakShape,
    ) -> Vec<PeakShapeModel<'lifespan>>
    where
        &'lifespan T: Into<PeakShapeModel<'lifespan>>,
    {
        let mut result: Vec<PeakShapeModel<'lifespan>> = Vec::with_capacity(peaks.len());
        for mut model in peaks
            .iter()
            .map(|x| -> PeakShapeModel<'lifespan> { x.into() })
        {
            model.shape = shape;
            result.push(model);
        }
        result.sort_unstable();
        result
    }

    /// Create a new spectrum from `models` over the shared m/z axis
    pub fn reprofile_from_models(
        &'lifespan self,
        models: &[PeakShapeModel<'transient>],
    ) -> ArrayPair<'lifespan> {
        if models.is_empty() {
            return ArrayPair::new(
                Cow::Borrowed(&self.mz_grid()[0..0]),
                Cow::Owned(self.create_intensity_array_of_size(0)),
            );
        }
        let mz_start = models.first().unwrap().center();
        let mz_end = models.last().unwrap().center();

        let _start_index = self.find_offset(mz_start);
        let _end_index = self.find_offset(mz_end);
        let mz_view = &self.mz_grid();

        // let n_points = self.points_between(mz_start, mz_end);
        let mut result = self.create_intensity_array_of_size(self.mz_grid.len());

        for model in models.iter() {
            let (mz_start, mz_end) = model.extremes();
            let start_index = self.find_offset(mz_start);
            let end_index = self.find_offset(mz_end);
            model.shape_in(
                &mz_view[start_index..end_index],
                &mut result[start_index..end_index],
            )
        }
        ArrayPair::new(Cow::Borrowed(mz_view), Cow::Owned(result))
    }

    /// Create a new spectrum from `peaks` after creating [`PeakShapeModel`]s of them
    /// over the shared m/z axis
    pub fn reprofile<T: Into<PeakShapeModel<'transient>> + Clone>(
        &'lifespan self,
        peaks: &'lifespan [T],
    ) -> ArrayPair<'lifespan> {
        let models: Vec<_> = peaks.iter().cloned().map(|p| p.into()).collect();
        self.reprofile_from_models(&models)
    }

    /// Create a new spectrum from `peaks` after creating [`PeakShapeModel`]s of them
    /// over the shared m/z axis using a uniform peak width parameter `fwhm`
    pub fn reprofile_from_centroids<T>(
        &'lifespan self,
        peaks: &'lifespan [T],
        fwhm: f32,
    ) -> ArrayPair<'lifespan>
    where
        T: AsPeakShapeModel<'passing, 'passing>,
    {
        let mut models = Vec::with_capacity(peaks.len());
        for p in peaks.iter() {
            let pm = p.as_peak_shape_model(fwhm, PeakShape::Gaussian);
            models.push(pm);
        }
        self.reprofile_from_models(&models)
    }
}

/// Convert an iterator of peak-like objects into an `ArrayPair` with spacing `dx`
pub fn reprofile<'transient, 'lifespan: 'transient, T: Iterator<Item = &'lifespan P>, P>(
    peaks: T,
    dx: f64,
) -> ArrayPair<'static>
where
    &'lifespan P: Into<PeakShapeModel<'transient>>,
    P: 'static,
{
    let models: Vec<PeakShapeModel<'transient>> = peaks.map(|p| p.into()).collect();
    if models.is_empty() {
        return ArrayPair::from((Vec::new(), Vec::new()));
    }
    let mz_start = models.first().unwrap().extremes().0 - 1.0;
    let mz_end = models.last().unwrap().extremes().1 + 1.0;
    let reprofiler = PeakSetReprofiler::new(mz_start, mz_end, dx);
    let arrays = reprofiler.reprofile_from_models(&models);
    ArrayPair::from((
        reprofiler.copy_mz_array(),
        arrays.intensity_array.into_owned(),
    ))
}

impl MZGrid for PeakSetReprofiler {
    fn mz_grid(&self) -> &[f64] {
        &self.mz_grid
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::arrayops::ArrayPair;
    use crate::peak_picker::pick_peaks;
    use crate::reprofile::reprofile;
    use crate::test_data::{NOISE, X, Y};

    #[test]
    fn test_builder() -> () {
        let yhat: Vec<f32> = Y
            .iter()
            .zip(NOISE.iter())
            .map(|(y, e)| y * 50.0 + e * 20.0)
            .collect();
        let mut peaks = pick_peaks(&X, &yhat).unwrap();
        peaks.sort_by(|a, b| a.mz.total_cmp(&b.mz));
        assert_eq!(peaks.len(), 37);
        let iterator = peaks.iter();
        let pair = reprofile(iterator, 0.01);
        eprintln!("{} {}", pair.min_mz, pair.max_mz);
        let peaks2 = pick_peaks(&pair.mz_array, &pair.intensity_array).unwrap();
        assert_eq!(peaks2.len(), 32);
        let p1 = peaks
            .iter()
            .max_by(|a, b| a.intensity.total_cmp(&b.intensity))
            .unwrap();
        let p2 = peaks2
            .iter()
            .max_by(|a, b| a.intensity.total_cmp(&b.intensity))
            .unwrap();

        assert!(
            (p1.mz - p2.mz).abs() < 1e-3,
            "{} - {} = {}",
            p1.mz,
            p2.mz,
            p1.mz - p2.mz
        )
    }
}
