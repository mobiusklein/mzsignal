//! `mzsignal` is a library for performing low-level signal processing on
//! mass spectra en-route to converting a continuous profile-mode spectrum
//! into a centroided peak list.
//!
//! The peak picking facility can be used directly with [`PeakPicker`] which
//! implements a simple gaussian peak shape fitter. There are a some threshold
//! criteria that can be manipulated to control which fits are reported, see its
//! documentation for more details.
//!
//! When one spectrum is too noisy to accurately assign peaks, averaging the signal from multiple spectra
//! together can be better. The [`average`] submodule includes components
//! for merging together multiple profile spectra.
//!
//! The [`denoise`] submodule includes an algorithm for local background
//! noise removal, reducing the intensity of a region according to its noisiest window.
//!
//! The [`reprofile`] submodule provides an algorithm to convert
//! a centroid peak back into a profile spectrum, using either fitted or assumed peak
//! shape parameters.
//!
//! # Usage
//! ```
//! use std::fs;
//! use std::io;
//! use std::io::prelude::*;
//!
//! use mzsignal;
//!
//! let mut mz_array: Vec<f64> = Vec::new();
//! let mut intensity_array: Vec<f32> = Vec::new();
//!
//! // Read in signal arrays from a text file
//! let reader = io::BufReader::new(fs::File::open("./test/data/test.txt").unwrap());
//! for line in reader.lines() {
//!     let line = line.unwrap();
//!     let pref = line.trim();
//!     let chunks: Vec<&str> = pref.split("\t").collect();
//!     mz_array.push(chunks[0].parse::<f64>().expect("Expected number for m/z"));
//!     intensity_array.push(chunks[1].parse::<f32>().expect("Expected number for intensity"));
//! }
//!
//! // Create a peak picker
//! let picker = mzsignal::PeakPicker::default();
//!
//! // Create an accumulator
//! let mut acc = Vec::new();
//!
//! // Pick peaks
//! let peak_count = picker.discover_peaks(&mz_array, &intensity_array, &mut acc).unwrap();
//! assert_eq!(peak_count , 4);
//! for peak in acc.iter() {
//!     println!("{}", peak);
//! }
//! ```
//!
//! ## Data Ownership
//! Most algorithms in `mzsignal` require ownly a borrowed view of the m/z dimension of a
//! spectrum, creating a new array of different size if appropriate. The intensity array is
//! almost always re-allocated to fit the new m/z array shape.
//!
//! When possible, the data structures in [`average`],
//! [`SignalAverager`](crate::average::SignalAverager), and [`reprofile`],
//! [`PeakSetReprofiler`](crate::reprofile::PeakSetReprofiler) will own their high density
//! m/z arrays and can be re-used from spectrum to spectrum to avoid repeatedly allocating that
//! large array, and leave it to be borrowed again later. The high level functions in these modules
//! create new data structures to exactly fit the m/z ranges they are given on every call, slowing
//! them down by comparison if used over and over again. In both cases, even if the reference m/z
//! range is much larger than the region needed for a particular input's, only the required region
//! will be used for any given computation.
//!
//! ## Building
//! This library needs a small amount of linear algebra, so it depends on either `nalgebra` or `ndarray`+`ndarray-linalg`.
//!
//! If the you wish to use `ndarray-linalg`, it needs a LAPACK implementation, controlled by the following features:
//!   - `intel-mkl`
//!   - `openblas`
//!   - `netlib`
//!
//! otherwise, the default `nalgebra` backend will be used.

#![allow(unused_imports)]
pub mod arrayops;
pub mod average;
pub mod denoise;
pub(crate) mod histogram;
pub(crate) mod peak;
pub mod peak_picker;
pub mod peak_statistics;
pub mod smooth;
pub mod feature_mapping;
pub mod reprofile;
pub mod search;
pub mod text;

#[cfg(test)]
mod test_data;

pub use crate::arrayops::ArrayPair;
pub use crate::peak::FittedPeak;
pub use crate::histogram::{Histogram, percentile};
pub use crate::peak_picker::{pick_peaks, PeakFitType, PeakPicker, PeakPickerError};