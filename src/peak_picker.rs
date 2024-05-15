//! Algorithm for finding simple symmetric peaks in a 1D array iteratively.
//!
//!
use std::cmp;
use std::ops;

use log::{debug, info};

use thiserror::Error;

use num_traits::Float;
#[cfg(feature = "parallelism")]
use rayon::prelude::*;

use crate::peak::FittedPeak;
use crate::peak_statistics::isclose;
use crate::peak_statistics::lorentzian_fit;
use crate::peak_statistics::{
    approximate_signal_to_noise, full_width_at_half_max, quadratic_fit, WidthFit,
};
use crate::search::{nearest, nearest_binary, nearest_left, nearest_right};
use std::collections::btree_map::{BTreeMap, Entry};

/// The type of peak picking to perform, defining the expected
/// peak shape fitting function.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum PeakFitType {
    /// Fit a Gaussian peak shape using a closed form quadratic function to
    /// determine the true peak centroid from the digitized signal.
    #[default]
    Quadratic,
    /// A simple fit which assumes that the highest point is the
    /// centroid. If the digitized signal doesn't directly strike
    /// the apex, there will be a small amount of error.
    Apex,
    /// Fit a Lorentzian peak shape using an iterative least squares solution
    /// searching an approximate local optimum.
    Lorentzian
}

/// Check if the value in `it` are monotonically ascending or flat
pub fn is_increasing<F: Float + PartialOrd>(it: &[F]) -> bool {
    let (ascending, _) = it
        .iter()
        .fold((true, F::zero()), |(ascending, last_val), val| {
            if !ascending {
                (false, last_val)
            } else {
                ((last_val <= *val), *val)
            }
        });
    ascending
}

/// Hold a partial peak shape fit for [`PeakPicker`]
#[derive(Debug, Clone, Default)]
pub struct PartialPeakFit {
    /// Whether the fit has been initialized for the current peak or not
    pub set: bool,

    /// The signal to noise ratio for the current peak fit
    pub signal_to_noise: f32,

    /// The left width at half max
    pub left_width: f32,
    /// The right width at half max
    pub right_width: f32,
    /// The average width at half max
    pub full_width_at_half_max: f32,
}

impl PartialPeakFit {
    /// Reset the peak fit, invalidating all attributes and setting [`PartialPeakFit::set`] to `false`
    pub fn reset(&mut self) {
        self.set = false;
        self.left_width = -1.0;
        self.right_width = -1.0;
        self.full_width_at_half_max = -1.0;
        self.signal_to_noise = -1.0;
    }

    /// Copy the shape information from a [`WidthFit`]
    pub fn update(&mut self, fit: &WidthFit) {
        self.full_width_at_half_max = fit.full_width_at_half_max as f32;
        self.left_width = fit.left_width as f32;
        self.right_width = fit.right_width as f32;
        self.set = true;
    }
}

/// All the ways peak picking can fail
#[derive(Debug, Clone, Error)]
pub enum PeakPickerError {
    #[error("Unknown error occurred")]
    Unknown,
    #[error("The m/z and intensity arrays do not match in length")]
    MZIntensityMismatch,
    #[error("The peak picking interval is too narrow")]
    IntervalTooSmall,
    #[error("The m/z array is not sorted")]
    MZNotSorted,
}

/// A peak picker for mass spectra
#[derive(Debug, Clone, Default)]
pub struct PeakPicker {
    pub background_intensity: f32,
    pub intensity_threshold: f32,
    pub signal_to_noise_threshold: f32,
    pub fit_type: PeakFitType,
}

/// A builder for configuring [`PeakPicker`]
#[derive(Debug, Clone, Default)]
pub struct PeakPickerBuilder {
    background_intensity: f32,
    intensity_threshold: f32,
    signal_to_noise_threshold: f32,
    fit_type: PeakFitType,
}

impl PeakPickerBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn background_intensity(&mut self, background_intensity: f32) -> &mut Self {
        self.background_intensity = background_intensity;
        self
    }

    pub fn intensity_threshold(&mut self, intensity_threshold: f32) -> &mut Self {
        self.intensity_threshold = intensity_threshold;
        self
    }

    pub fn signal_to_noise_threshold(&mut self, signal_to_noise_threshold: f32) -> &mut Self {
        self.signal_to_noise_threshold = signal_to_noise_threshold;
        self
    }

    pub fn fit_type(&mut self, fit_type: PeakFitType) -> &mut Self {
        self.fit_type = fit_type;
        self
    }

    pub fn build(self) -> PeakPicker {
        PeakPicker::new(
            self.background_intensity,
            self.intensity_threshold,
            self.signal_to_noise_threshold,
            self.fit_type,
        )
    }
}

impl From<PeakPickerBuilder> for PeakPicker {
    fn from(value: PeakPickerBuilder) -> Self {
        value.build()
    }
}

impl PeakPicker {
    /// Create a new peak picker
    pub fn new(
        background_intensity: f32,
        intensity_threshold: f32,
        signal_to_noise_threshold: f32,
        fit_type: PeakFitType,
    ) -> Self {
        Self {
            background_intensity,
            intensity_threshold,
            signal_to_noise_threshold,
            fit_type,
        }
    }

    /// Pick peaks from `mz_array` and `intensity_array`, pushing new peaks into `peak_accumulator`.
    ///
    /// Returns the number of peaks picked if successful.
    ///
    /// This is a thin wrapper around [`PeakPicker::discover_peaks_in_interval`].
    pub fn discover_peaks(
        &self,
        mz_array: &[f64],
        intensity_array: &[f32],
        peak_accumulator: &mut Vec<FittedPeak>,
    ) -> Result<usize, PeakPickerError> {
        match mz_array.first() {
            Some(start_mz) => match mz_array.last() {
                Some(stop_mz) => self.discover_peaks_in_interval(
                    mz_array,
                    intensity_array,
                    peak_accumulator,
                    *start_mz,
                    *stop_mz,
                ),
                None => Ok(0),
            },
            None => Ok(0),
        }
    }

    fn is_prominent(&self, prev: f32, cur: f32, next: f32) -> bool {
        (prev <= cur) && (cur >= next) && (cur > 0.0)
    }

    /// Fit a peak at position `index` in `mz_array` and `intensity_array`.
    ///
    /// `index` is assumed to be the most intense point along the putative
    /// peak, the "apex", thus for [`PeakFitType::Apex`], this function is
    /// simply `mz_array[index]`.
    ///
    /// Returns the estimated m/z of the "true" peak.
    pub fn fit_peak(
        &self,
        index: usize,
        mz_array: &[f64],
        intensity_array: &[f32],
        partial_fit: &PartialPeakFit,
    ) -> f64 {
        match self.fit_type {
            PeakFitType::Quadratic => quadratic_fit(mz_array, intensity_array, index),
            PeakFitType::Apex => mz_array[index],
            PeakFitType::Lorentzian => lorentzian_fit(mz_array, intensity_array, index, partial_fit),
        }
    }

    /// Pick peaks from `mz_array` and `intensity_array` between `start_mz` and `stop_mz`,
    /// pushing new peaks into `peak_accumulator`.
    ///
    /// Returns the number of peaks picked if successful
    pub fn discover_peaks_in_interval(
        &self,
        mz_array: &[f64],
        intensity_array: &[f32],
        peak_accumulator: &mut Vec<FittedPeak>,
        start_mz: f64,
        stop_mz: f64,
    ) -> Result<usize, PeakPickerError> {
        let intensity_threshold = self.intensity_threshold;
        let signal_to_noise_threshold = self.signal_to_noise_threshold;

        let n = mz_array.len() - 1;
        let m = peak_accumulator.len();

        let start_index = cmp::max(nearest(mz_array, start_mz, 0), 1);
        let stop_index = cmp::min(nearest(mz_array, stop_mz, n), n - 1);

        if intensity_array[start_index..stop_index].len() != mz_array[start_index..stop_index].len()
        {
            return Err(PeakPickerError::MZIntensityMismatch);
        }

        let mut partial_fit_state = PartialPeakFit::default();

        let mut index = start_index;

        while index <= stop_index {
            partial_fit_state.reset();
            let current_intensity = intensity_array[index];
            let current_mz = mz_array[index];

            let last_intensity = intensity_array[index - 1];
            let next_intensity = intensity_array[index + 1];
            if self.is_prominent(last_intensity, current_intensity, next_intensity)
                && (current_intensity >= intensity_threshold)
            {
                let mut fwhm = 0.0;
                let mut signal_to_noise =
                    approximate_signal_to_noise(current_intensity, intensity_array, index);

                partial_fit_state.signal_to_noise = signal_to_noise;

                // Run Full-Width Half-Max algorithm to try to improve SNR
                if signal_to_noise < signal_to_noise_threshold {
                    let shape_fit =
                        full_width_at_half_max(mz_array, intensity_array, index, signal_to_noise);
                    partial_fit_state.update(&shape_fit);

                    fwhm = partial_fit_state.full_width_at_half_max;
                    if (0.0 < fwhm) && (fwhm < 0.5) {
                        // TODO: Try to use local searches here instead of full range searches
                        let ilow = nearest_left(
                            mz_array,
                            current_mz - partial_fit_state.left_width as f64,
                            index,
                        );
                        let ihigh = nearest_right(
                            mz_array,
                            current_mz + partial_fit_state.right_width as f64,
                            index,
                        );

                        let low_intensity = intensity_array[ilow];
                        let high_intensity = intensity_array[ihigh];
                        let sum_intensity = low_intensity + high_intensity;

                        if sum_intensity > 0.0 {
                            signal_to_noise = (2.0 * current_intensity) / sum_intensity;
                        } else {
                            signal_to_noise = 10.0;
                        }

                        partial_fit_state.signal_to_noise = signal_to_noise;
                    }
                }

                // Found a putative peak, fit it
                if signal_to_noise >= signal_to_noise_threshold {
                    let fitted_mz =
                        self.fit_peak(index, mz_array, intensity_array, &partial_fit_state);
                    if !partial_fit_state.set {
                        let shape_fit = full_width_at_half_max(
                            mz_array,
                            intensity_array,
                            index,
                            signal_to_noise,
                        );
                        partial_fit_state.update(&shape_fit);
                        fwhm = partial_fit_state.full_width_at_half_max;
                    }
                    if fwhm > 0.0 {
                        if fwhm > 1.0 {
                            fwhm = 1.0;
                        }

                        if signal_to_noise > current_intensity {
                            signal_to_noise = current_intensity;
                        }

                        let peak = FittedPeak::new(
                            fitted_mz,
                            current_intensity,
                            index as u32,
                            signal_to_noise,
                            fwhm,
                        );
                        // eprintln!("Storing peak with mz {:0.3}/{:0.3} with FWHM {:0.3}", fitted_mz, current_mz, fwhm);
                        peak_accumulator.push(peak);
                        partial_fit_state.reset();
                        while index < stop_index
                            && isclose(intensity_array[index + 1], current_intensity)
                        {
                            // eprintln!("Advancing over equal intensity point {} @ {}", index, current_intensity);
                            index += 1;
                        }
                    } else {
                        // eprintln!("Skipping peak with FWHM {:03} and SNR {:03}", fwhm, signal_to_noise)
                    }
                } else {
                    // eprintln!("Skipping peak with FWHM {:03} and SNR {:03}", fwhm, signal_to_noise)
                }
            }
            index += 1;
        }
        Ok(peak_accumulator.len() - m)
    }

    #[cfg(feature = "parallelism")]
    #[allow(dead_code)]
    fn discover_peaks_parallel_with_overhang(
        &self,
        mz_array: &[f64],
        intensity_array: &[f32],
        peak_accumulator: &mut Vec<FittedPeak>,
        n_chunks: usize,
        overhang: usize,
    ) -> Result<usize, PeakPickerError> {
        let n = mz_array.len();
        let chunk_size = n / n_chunks;
        eprintln!("Chunk size: {}, Overhang: {}", chunk_size, overhang);
        if chunk_size <= overhang {
            return self.discover_peaks(mz_array, intensity_array, peak_accumulator);
        }
        let windows: Vec<ops::Range<usize>> = (0..n_chunks)
            .map(|i| {
                (if i == 0 { 0 } else { i * chunk_size - overhang })
                    ..cmp::min((i + 1) * chunk_size + overhang, n)
            })
            .collect();
        eprintln!("Windows: {:?}", windows);
        let peaks_or_errors: Vec<Result<(Vec<FittedPeak>, ops::Range<usize>), PeakPickerError>> =
            windows
                .into_par_iter()
                .map(|iv| {
                    let mut local_acc: Vec<FittedPeak> = Vec::new();
                    let start_idx = iv.start;
                    let end_idx = iv.end;

                    match self.discover_peaks(
                        &mz_array[start_idx..end_idx],
                        &intensity_array[start_idx..end_idx],
                        &mut local_acc,
                    ) {
                        Ok(_i) => {
                            let res: Vec<FittedPeak> = local_acc
                                .into_iter()
                                .map(|mut p| {
                                    // Shift the indices forwards to match the "real" coordinates
                                    p.index += start_idx as u32;
                                    p
                                })
                                // If the peak's index falls within either overhang, drop it
                                .filter(|p| {
                                    (p.index - start_idx as u32 > overhang as u32 || start_idx == 0)
                                        && (end_idx == n || p.index < (end_idx - overhang) as u32)
                                })
                                .collect();

                            Ok((res, iv))
                        }
                        Err(e) => Err(e),
                    }
                })
                .collect();

        let mut ivmap: BTreeMap<usize, Vec<FittedPeak>> = BTreeMap::new();

        // Iterate over each chunk, inserting it into an ordered map. If
        // there are multiple intervals with the same start that both have
        // peaks, an error is returned. If any of the chunks return an error,
        // an error is returned.
        for por in peaks_or_errors {
            match por {
                Ok((peaks, iv)) => {
                    let s = iv.start;
                    match ivmap.entry(s) {
                        Entry::Vacant(entry) => {
                            entry.insert(peaks);
                        }
                        Entry::Occupied(mut entry) => {
                            if entry.get().is_empty() {
                                entry.insert(peaks);
                            } else if peaks.is_empty() {
                            } else {
                                return Err(PeakPickerError::IntervalTooSmall);
                            };
                        }
                    }
                }
                Err(err) => return Err(err),
            };
        }

        for (_start, peaks) in ivmap {
            peak_accumulator.extend(peaks);
        }

        Ok(peak_accumulator.len())
    }

    #[cfg(feature = "parallelism")]
    #[allow(dead_code)]
    fn discover_peaks_parallel(
        &self,
        mz_array: &[f64],
        intensity_array: &[f32],
        peak_accumulator: &mut Vec<FittedPeak>,
        n_chunks: usize,
    ) -> Result<usize, PeakPickerError> {
        let n = mz_array.len();
        let chunk_size = n / n_chunks;
        let overhang = cmp::max(chunk_size / 2, 15);
        self.discover_peaks_parallel_with_overhang(
            mz_array,
            intensity_array,
            peak_accumulator,
            n_chunks,
            // This value needs to reflect ~twice the number of data points on a peak
            overhang,
        )
    }
}


/// A convenience function that uses a default peak picking configuration to pick peaks from paired
/// m/z and intensity arrays.
///
/// Performs a minimal m/z sorting check with [`is_increasing`]
pub fn pick_peaks(
    mz_array: &[f64],
    intensity_array: &[f32],
) -> Result<Vec<FittedPeak>, PeakPickerError> {
    let picker = PeakPicker::default();
    let mut acc = Vec::new();
    if !is_increasing(mz_array) {
        return Err(PeakPickerError::MZNotSorted);
    }
    match picker.discover_peaks(mz_array, intensity_array, &mut acc) {
        Ok(_) => Ok(acc),
        Err(err) => Err(err),
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::arrayops::ArrayPair;
    use crate::average::SignalAverager;
    use crate::test_data::{NOISE, X, Y};
    use rstest::rstest;

    #[test]
    fn test_peak_picker_no_noise() {
        let picker = PeakPicker::default();
        let mut acc = Vec::new();
        let result = picker.discover_peaks(&X, &Y, &mut acc);
        let z = result.expect("Should not encounter an error");
        assert_eq!(z, 4);
        let xs = [
            180.0633881,
            181.06578875772615,
            182.06662482711,
            183.06895705008014,
        ];
        for (peak, x) in acc.iter().zip(xs.iter()) {
            assert!((peak.mz - x).abs() < 1e-6);
        }
    }

    #[test]
    fn test_peak_picker_no_noise_snr_threshold() {
        let picker = PeakPicker {
            signal_to_noise_threshold: 10.0,
            ..PeakPicker::default()
        };
        let mut acc = Vec::new();
        let result = picker.discover_peaks(&X, &Y, &mut acc);
        let z = result.expect("Should not encounter an error");
        assert_eq!(z, 2);
        let xs = [180.0633881, 181.06578875772615];
        for (peak, x) in acc.iter().zip(xs.iter()) {
            assert!((peak.mz - x).abs() < 1e-6);
        }
    }

    #[test]
    fn test_peak_picker_noisy_snr_threshold() {
        let picker = PeakPicker {
            signal_to_noise_threshold: 10.0,
            ..PeakPicker::default()
        };
        let mut acc = Vec::new();
        let yhat: Vec<f32> = Y
            .iter()
            .zip(NOISE.iter())
            .map(|(y, e)| y * 5.0 + e)
            .collect();
        let result = picker.discover_peaks(&X, &yhat, &mut acc);
        let z = result.expect("Should not encounter an error");
        assert_eq!(z, 8);
    }

    #[test]
    #[cfg(feature = "parallelism")]
    fn test_peak_picker_no_noise_parallel() {
        let picker = PeakPicker::default();
        let mut acc = Vec::new();
        let result = picker.discover_peaks_parallel(&X, &Y, &mut acc, 6);
        let z = result.expect("Should not encounter an error");
        assert_eq!(z, 4);
        let xs = [
            180.0633881,
            181.06578875772615,
            182.06662482711,
            183.06895705008014,
        ];
        for (peak, x) in acc.iter().zip(xs.iter()) {
            assert!((peak.mz - x).abs() < 1e-6);
        }
    }

    // This test is expected to fail because the hacky logic used in the tested method
    // does not work on it at the desired precision. The overhangs need to be computed
    // dynamically.
    #[rstest]
    #[should_panic]
    #[cfg(feature = "parallelism")]
    fn test_peak_picker_no_noise_parallel_rebinned() {
        let picker = PeakPicker::default();
        let mut acc = Vec::new();

        let mut averager = SignalAverager::new(X[0], X[X.len() - 1], 0.0001);
        averager.push(ArrayPair::wrap(&X, &Y));
        let yhat = averager.interpolate();

        let result = picker.discover_peaks_parallel(&averager.mz_grid, &yhat, &mut acc, 6);
        let z = result.expect("Should not encounter an error");
        assert_eq!(z, 4);
        let xs = [
            180.0633881,
            181.06578875772615,
            182.06662482711,
            183.06895705008014,
        ];
        for (peak, x) in acc.iter().zip(xs.iter()) {
            println!("{} ? {}", peak, x);
            assert!((peak.mz - x).abs() < 1e-6);
        }
    }
}
