use std::cmp;
use std::fmt;
use std::hash;

use mzpeaks;
use mzpeaks::peak::MZPoint;
use mzpeaks::prelude::*;
use mzpeaks::CentroidPeak;
use mzpeaks::{
    CentroidLike, CoordinateLike, IndexType, IndexedCoordinate, IntensityMeasurement, MZ,
};

#[derive(Debug, Clone, Copy, Default)]
/// A [`FittedPeak`] implements the [`CentroidLike`](https://docs.rs/mzpeaks/latest/mzpeaks/peak/trait.CentroidLike.html) trait
/// with an m/z coordinate, but also a shape attribute `full_width_at_half_max` and a
/// intensity uncertainty, `signal_to_noise_ratio`.
pub struct FittedPeak {
    pub mz: f64,
    pub intensity: f32,
    pub index: u32,

    /// A measure of the difference between the intensity of this peak and the
    /// surrounding data
    pub signal_to_noise: f32,
    /// A symmetric average peak shape parameter
    pub full_width_at_half_max: f32,
}

impl FittedPeak {
    pub fn new(
        mz: f64,
        intensity: f32,
        index: u32,
        signal_to_noise: f32,
        full_width_at_half_max: f32,
    ) -> Self {
        Self {
            mz,
            intensity,
            index,
            signal_to_noise,
            full_width_at_half_max,
        }
    }
}

// Implement the CentroidLike interface
mzpeaks::implement_centroidlike!(FittedPeak, true);

impl From<MZPoint> for FittedPeak {
    fn from(value: MZPoint) -> Self {
        Self {
            mz: value.mz,
            intensity: value.intensity,
            index: value.get_index(),
            ..Default::default()
        }
    }
}

impl fmt::Display for FittedPeak {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "FittedPeak({}, {}, {}, {}, {})",
            self.mz, self.intensity, self.index, self.full_width_at_half_max, self.signal_to_noise
        )
    }
}
