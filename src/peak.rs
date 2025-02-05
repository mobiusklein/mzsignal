
use std::fmt;

use mzpeaks::peak::MZPoint;
use mzpeaks::{
    CentroidLike, CoordinateLike, IndexedCoordinate, IntensityMeasurement,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

const DEFAULT_FWHM: f32 = 0.005;

#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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

mzpeaks::implement_mz_coord!(FittedPeak);

impl mzpeaks::IndexedCoordinate<mzpeaks::MZ> for FittedPeak {
    #[inline]
    fn get_index(&self) -> mzpeaks::IndexType {
        self.index
    }
    #[inline]
    fn set_index(&mut self, index: mzpeaks::IndexType) {
        self.index = index
    }
}

impl From<FittedPeak> for mzpeaks::CentroidPeak {
    fn from(peak: FittedPeak) -> Self {
        peak.as_centroid()
    }
}

impl From<FittedPeak> for mzpeaks::peak::MZPoint {
    fn from(peak: FittedPeak) -> Self {
        Self {
            mz: peak.coordinate(),
            intensity: peak.intensity(),
        }
    }
}

/// Conversion from a [`mzpeaks::CentroidPeak`]
impl From<mzpeaks::CentroidPeak> for FittedPeak {
    fn from(peak: mzpeaks::CentroidPeak) -> Self {
        let mut inst = Self {
            mz: peak.coordinate(),
            intensity: peak.intensity(),
            full_width_at_half_max: DEFAULT_FWHM,
            ..Self::default()
        };
        inst.set_index(peak.index);
        inst
    }
}

impl From<MZPoint> for FittedPeak {
    fn from(value: MZPoint) -> Self {
        Self {
            mz: value.mz,
            intensity: value.intensity,
            index: value.get_index(),
            full_width_at_half_max: DEFAULT_FWHM,
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


#[cfg(test)]
mod test {
    use super::*;
    use mzpeaks::prelude::*;
    use mzpeaks::CentroidPeak;


    #[test]
    fn test_conversion() {
        let peak = CentroidPeak::new(1500.0, 6e3, 0);
        let fpeak = FittedPeak::from(peak.clone());
        assert_eq!(fpeak.full_width_at_half_max, DEFAULT_FWHM);
        assert_eq!(fpeak.mz(), peak.mz);
        assert_eq!(CentroidPeak::from(fpeak), peak);
        assert_eq!(MZPoint::from(fpeak), MZPoint::from(peak.clone()));
    }
}