//! Convert picked peaks into a profile spectrum.
//!
use std::borrow;
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
    pub peak: borrow::Cow<'lifespan, FittedPeak>,
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
        return self.peak.mz.partial_cmp(&other.peak.mz);
    }
}

impl<'lifespan> Ord for PeakShapeModel<'lifespan> {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.partial_cmp(other).unwrap()
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
            peak: borrow::Cow::Owned(FittedPeak {
                mz,
                intensity,
                full_width_at_half_max,
                ..FittedPeak::default()
            }),
            shape,
        }
    }

    pub fn gaussian(peak: &FittedPeak) -> PeakShapeModel {
        PeakShapeModel {
            peak: borrow::Cow::Borrowed(peak),
            shape: PeakShape::Gaussian,
        }
    }

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

    pub fn shape(&self, dx: f64) -> (Vec<f64>, Vec<f32>) {
        let (start, end) = self.extremes();
        let mz_array = gridspace(start, end, dx);
        let intensity_array = mz_array.iter().map(|x| self.predict(x)).collect();
        (mz_array, intensity_array)
    }

    pub fn shape_in(&self, mz_array: &[f64], intensity_array: &mut [f32]) {
        for (i, val) in mz_array.iter().map(|x| self.predict(x)).enumerate() {
            intensity_array[i] += val;
        }
    }

    pub fn area(&self, dx: f64) -> f32 {
        let (x, y) = self.shape(dx);
        trapz(&x, &y)
    }

    pub fn center(&self) -> f64 {
        self.peak.mz
    }

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

pub trait AsPeakShapeModel<'a, 'b: 'a> {
    fn as_peak_shape_model(self: &'a Self, fwhm: f32, shape: PeakShape) -> PeakShapeModel<'b>;
}

impl<'a, 'b: 'a, T: CentroidLike> AsPeakShapeModel<'a, 'b> for &T {
    fn as_peak_shape_model(self: &'a Self, fwhm: f32, shape: PeakShape) -> PeakShapeModel<'b> {
        PeakShapeModel::from_centroid(self.coordinate(), self.intensity(), fwhm, shape)
    }
}

#[derive(Debug, Default, Clone)]
pub struct PeakSetReprofiler {
    pub mz_grid: Vec<f64>,
    pub mz_start: f64,
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

    pub fn build_peak_shape_models<T>(
        &self,
        peaks: &'lifespan Vec<T>,
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

    pub fn reprofile_from_models(
        &'lifespan self,
        models: Vec<PeakShapeModel<'transient>>,
    ) -> ArrayPair<'lifespan> {
        if models.is_empty() {
            return ArrayPair::new(
                borrow::Cow::Borrowed(&self.mz_grid()[0..0]),
                borrow::Cow::Owned(self.create_intensity_array_of_size(0)),
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
        ArrayPair::new(borrow::Cow::Borrowed(&mz_view), borrow::Cow::Owned(result))
    }

    pub fn reprofile<T>(&'lifespan self, peaks: &'lifespan Vec<T>) -> ArrayPair<'lifespan>
    where
        &'lifespan T: Into<PeakShapeModel<'transient>> + 'static,
    {
        let models = peaks.iter().map(|p| p.into()).collect();
        self.reprofile_from_models(models)
    }

    pub fn reprofile_from_centroids<T>(
        &'lifespan self,
        peaks: &'lifespan Vec<T>,
        fwhm: f32,
    ) -> ArrayPair<'lifespan>
    where
        T: AsPeakShapeModel<'passing, 'passing> + 'static,
    {
        let mut models = Vec::with_capacity(peaks.len());
        for p in peaks.iter() {
            let pm = p.as_peak_shape_model(fwhm, PeakShape::Gaussian);
            models.push(pm);
        }
        self.reprofile_from_models(models)
    }
}

/// Convert an iterator of peak-like objects into an `ArrayPair` with spacing `dx`
pub fn reprofile<'transient, 'lifespan: 'transient, T: Iterator<Item = &'lifespan P>, P>(
    peaks: T,
    dx: f64,
) -> ArrayPair<'lifespan>
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
    let arrays = reprofiler.reprofile_from_models(models);
    let result = ArrayPair::from((
        reprofiler.copy_mz_array(),
        arrays.intensity_array.into_owned(),
    ));
    result
}

impl MZGrid for PeakSetReprofiler {
    fn mz_grid(&self) -> &[f64] {
        &self.mz_grid
    }
}

#[cfg(test)]
mod test {
    use super::*;
}
