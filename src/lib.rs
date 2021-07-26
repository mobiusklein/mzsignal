//! `mzsignal` is a library for performing low-level signal processing on
//! mass spectra en-route to converting a continuous profile-mode spectrum
//! into a centroided peak list.
//!
//! The peak picking facility can be used directly with [`PeakPicker`] which
//! implements a simple gaussian peak shape fitter. There are a some threshold
//! criteria that can be manipulated to control which fits are reported, see its
//! documentation for more details.
//!
//! When one spectrum is insufficient, averaging the signal from multiple spectra
//! together can be better. The [`crate::average`] sub-module includes components
//! for
#![allow(unused_imports)]
pub mod peak;
pub mod peak_picker;
pub mod peak_statistics;
pub mod search;
pub mod average;

#[cfg(test)]
mod test_data;

pub use crate::peak::FittedPeak;
pub use crate::peak_picker::{PeakFitType, PeakPicker, PeakPickerError, pick_peaks};
