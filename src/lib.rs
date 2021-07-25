#![allow(unused_imports)]
pub mod peak;
pub mod peak_picker;
pub mod peak_statistics;
pub mod search;
pub mod test_data;
pub mod average;

use crate::peak::FittedPeak;
use crate::peak_picker::{PeakFitType, PeakPicker, PeakPickerError, pick_peaks};
