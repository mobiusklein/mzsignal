use std::borrow;

use crate::peak::FittedPeak;

use crate::arrayops::{gridspace, trapz};


#[derive(Debug, Clone, Copy)]
pub enum PeakShape {
    Gaussian
}


#[derive(Debug, Clone)]
pub struct PeakShapeModel<'lifespan> {
    pub peak: borrow::Cow<'lifespan, FittedPeak>,
    pub shape: PeakShape
}

impl<'lifespan> PeakShapeModel<'lifespan> {
    pub fn from_centroid(mz: f64, intensity: f32, full_width_at_half_max: f32, shape: PeakShape) -> PeakShapeModel<'lifespan> {
        PeakShapeModel {
            peak: borrow::Cow::Owned(FittedPeak {
                mz, intensity, full_width_at_half_max, .. FittedPeak::default()
            }), shape
        }
    }

    pub fn gaussian(peak: &FittedPeak) -> PeakShapeModel {
        PeakShapeModel {
            peak: borrow::Cow::Borrowed(peak), shape: PeakShape::Gaussian
        }
    }

    pub fn predict(&self, mz: &f64) -> f32 {
        match self.shape {
            PeakShape::Gaussian => {
                let spread = self.peak.full_width_at_half_max / 2.35482;
                let scaler = (-(f64::powf(mz - self.peak.mz, 2.0)) / (2.0 * f64::powf(spread as f64, 2.0))).exp();
                self.peak.intensity * scaler as f32
            }
        }
    }

    pub fn shape(&self, dx: f64) -> (Vec<f64>, Vec<f32>) {
        let mz_array = gridspace(self.peak.mz - self.peak.full_width_at_half_max as f64 - 0.02,
                                 self.peak.mz - self.peak.full_width_at_half_max as f64 - 0.02, dx);
        let intensity_array = mz_array.iter().map(|x| self.predict(x)).collect();
        (mz_array, intensity_array)
    }

    pub fn shape_in(&self, mz_array: &[f64], intensity_array: &mut [f32]) {
        for (i, val) in mz_array.iter().map(|x| self.predict(x)).enumerate() {
            intensity_array[i] = val;
        }
    }

    pub fn area(&self, dx: f64) -> f32 {
        let (x, y) = self.shape(dx);
        trapz(&x, &y)
    }
}
