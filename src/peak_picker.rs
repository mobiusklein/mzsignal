use std::cmp;
use std::ops;

#[cfg(feature = "parallelism")]
use rayon::prelude::*;

use crate::peak::FittedPeak;
use crate::peak_statistics::{
    approximate_signal_to_noise, full_width_at_half_max, quadratic_fit, WidthFit,
};
use crate::search::{nearest, nearest_binary};
use std::collections::btree_map::{BTreeMap, Entry};

#[derive(Debug, Clone, Copy)]
pub enum PeakFitType {
    Quadratic,
    Apex,
}

impl Default for PeakFitType {
    fn default() -> PeakFitType {
        PeakFitType::Quadratic
    }
}

#[derive(Debug, Clone, Default)]
pub struct PartialPeakFit {
    pub set: bool,
    pub signal_to_noise: f32,

    /// Shape holders
    pub left_width: f32,
    pub right_width: f32,
    pub full_width_at_half_max: f32,
}

impl PartialPeakFit {
    pub fn reset(&mut self) {
        self.set = false;
        self.left_width = -1.0;
        self.right_width = -1.0;
        self.full_width_at_half_max = -1.0;
        self.signal_to_noise = -1.0;
    }

    pub fn update(&mut self, fit: &WidthFit) {
        self.full_width_at_half_max = fit.full_width_at_half_max as f32;
        self.left_width = fit.left_width as f32;
        self.right_width = fit.right_width as f32;
        self.set = true;
    }
}

#[derive(Debug, Clone)]
pub enum PeakPickerError {
    Unknown,
    MZIntensityMismatch,
    IntervalTooSmall,
}

#[derive(Debug, Clone, Default)]
pub struct PeakPicker {
    pub background_intensity: f32,
    pub intensity_threshold: f32,
    pub signal_to_noise_threshold: f32,

    pub partial_fit_state: PartialPeakFit,
    pub fit_type: PeakFitType,
}

impl PeakPicker {
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

    #[allow(unused_variables)]
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
        }
    }

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

        for index in start_index..=stop_index {
            partial_fit_state.reset();
            let current_intensity = intensity_array[index];
            let current_mz = mz_array[index];

            let last_intensity = intensity_array[index - 1];
            let next_intensity = intensity_array[index + 1];
            if (current_intensity >= last_intensity)
                && (current_intensity != 0.0)
                && (current_intensity >= next_intensity)
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
                        let ilow = nearest_binary(
                            mz_array,
                            current_mz - partial_fit_state.left_width as f64,
                            0,
                            index,
                        );
                        let ihigh = nearest_binary(
                            mz_array,
                            current_mz + partial_fit_state.right_width as f64,
                            index,
                            stop_index,
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

                        let peak = FittedPeak {
                            mz: fitted_mz,
                            intensity: current_intensity,
                            index: index as u32,
                            full_width_at_half_max: fwhm,
                            signal_to_noise,
                        };
                        peak_accumulator.push(peak);
                    }
                }
            }
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
        println!("Chunk size: {}, Overhang: {}", chunk_size, overhang);
        if chunk_size <= overhang {
            return self.discover_peaks(mz_array, intensity_array, peak_accumulator);
        }
        let windows: Vec<ops::Range<usize>> = (0..n_chunks)
            .map(|i| {
                (if i == 0 { 0 } else { i * chunk_size - overhang })
                    ..cmp::min((i + 1) * chunk_size + overhang, n)
            })
            .collect();
        println!("Windows: {:?}", windows);
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

pub fn pick_peaks(
    mz_array: &[f64],
    intensity_array: &[f32],
) -> Result<Vec<FittedPeak>, PeakPickerError> {
    let picker = PeakPicker::default();
    let mut acc = Vec::new();
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
        assert_eq!(z, 5);
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
