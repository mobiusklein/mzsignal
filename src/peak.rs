use std::cmp;
use std::fmt;
use std::hash;

#[derive(Debug, Clone, Default)]
pub struct FittedPeak {
    pub mz: f64,
    pub intensity: f32,
    pub index: u32,

    pub signal_to_noise: f32,
    pub full_width_at_half_max: f32,
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

impl hash::Hash for FittedPeak {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        let mz_val: i64 = self.mz.round() as i64;
        mz_val.hash(state);
    }
}

impl cmp::PartialOrd<FittedPeak> for FittedPeak {
    fn partial_cmp(&self, other: &FittedPeak) -> Option<cmp::Ordering> {
        self.mz.partial_cmp(&other.mz)
    }
}

impl cmp::PartialEq<FittedPeak> for FittedPeak {
    fn eq(&self, other: &FittedPeak) -> bool {
        if (self.mz - other.mz).abs() > 1e-3 || (self.intensity - other.intensity).abs() > 1e-3 {
            return false;
        }
        true
    }
}
