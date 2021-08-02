//! Remove local noise from a spectrum using the denoising algorithm from MasSpike.
//!
use std::ops;
use std::slice;

use num_traits::Float;

use crate::histogram::Histogram;
use crate::search;

struct Window<'lifespan> {
    pub mz_array: &'lifespan [f64],
    pub intensity_array: &'lifespan mut [f32],
    pub start_index: usize,
    pub end_index: usize,
    pub center_mz: f64,
    pub mean_intensity: f32,
    pub size: usize,
    pub is_empty: bool,
    pub histogram: Histogram<f32>,
    bins: usize,
}

impl<'transient, 'lifespan: 'transient> Window<'lifespan> {
    pub fn new(
        mz_array: &'lifespan [f64],
        intensity_array: &'lifespan mut [f32],
        start_index: usize,
        end_index: usize,
        center_mz: f64,
        bins: usize,
        is_empty: bool,
    ) -> Window<'lifespan> {
        let histogram = Histogram::new(intensity_array, bins);
        let mut window = Window {
            mz_array,
            intensity_array,
            start_index,
            end_index,
            center_mz,

            mean_intensity: 0.0,
            size: mz_array.len(),

            histogram,
            bins,
            is_empty,
        };
        window.mean_intensity =
            window.intensity_array.iter().sum::<f32>() / (window.intensity_array.len() as f32);
        window
    }

    pub fn deduct_intensity(&'transient mut self, value: f32) {
        let n = self.intensity_array.len();
        let mut total = 0.0;
        for i in 0..n {
            self.intensity_array[i] -= value;
            if self.intensity_array[i] < 0.0 {
                self.intensity_array[i] = 0.0;
            }
            total += self.intensity_array[i];
        }
        self.mean_intensity = total / n as f32;
    }

    pub fn rebin_intensities(&mut self) {
        self.histogram.clear();
        self.histogram.populate(self.intensity_array, self.bins)
    }

    pub fn truncated_mean_at(&mut self, threshold: f32) -> f32 {
        if self.size == 0 {
            return 1e-6;
        }
        self.rebin_intensities();
        let n_count = self.bins;

        let mut mask_level = 0;
        for i_count in 0..n_count {
            if self.histogram.bin_count[i_count] > mask_level {
                mask_level = self.histogram.bin_count[i_count];
            }
        }
        mask_level = (mask_level as f32 * (1.0 - threshold)) as usize;
        let mut total = 0.0;
        let mut weight = 0.0;
        for i_count in 0..n_count {
            if mask_level < self.histogram.bin_count[i_count] {
                total += self.histogram.bin_edges[i_count + 1]
                    * self.histogram.bin_count[i_count] as f32;
                weight += self.histogram.bin_count[i_count] as f32
            }
        }
        total / weight
    }

    pub fn truncated_mean(&mut self) -> f32 {
        self.truncated_mean_at(0.95)
    }
}

struct NoiseRegion<'lifespan> {
    pub windows: &'lifespan mut [Window<'lifespan>],
    pub width: u32,
    pub start_index: usize,
    pub end_index: usize,
    pub size: usize,
}

impl<'transient, 'lifespan: 'transient> NoiseRegion<'lifespan> {
    pub fn new(windows: &'lifespan mut [Window<'lifespan>], width: u32) -> NoiseRegion<'lifespan> {
        let mut inst = NoiseRegion {
            windows,
            width,
            start_index: 0,
            end_index: 0,
            size: 0,
        };
        if let Some(first) = inst.windows.first() {
            inst.start_index = first.start_index;
        }
        if let Some(last) = inst.windows.last() {
            inst.end_index = last.end_index;
        }
        inst.size = inst.windows.len();
        inst
    }

    pub fn noise_window(&mut self) -> Option<&mut Window<'lifespan>> {
        let i = 0;
        let n = self.size;

        if n == 0 {
            return None;
        }
        let mut minimum_window_index = i;
        let mut minimum = self[i].mean_intensity;
        for i in 1..n {
            let window = &self[i];
            if window.mean_intensity < minimum {
                minimum_window_index = i;
                minimum = window.mean_intensity;
            }
        }
        Some(&mut self[minimum_window_index])
    }

    pub fn noise_mean(&'transient mut self, scale: f32) -> f32 {
        match self.noise_window() {
            Some(noise_window) => noise_window.truncated_mean() * scale,
            None => 0.0,
        }
    }

    fn deduct_intensity_from_all_windows(&'transient mut self, noise: f32) {
        self.windows
            .iter_mut()
            .for_each(|w| w.deduct_intensity(noise));
    }

    pub fn denoise(&'lifespan mut self, scale: f32, maxiter: u32) -> f32 {
        if scale == 0.0 {
            return 0.0;
        }

        let mut noise_mean = self.noise_mean(scale);
        let first_mean = noise_mean;
        self.deduct_intensity_from_all_windows(noise_mean);
        let mut last_mean = noise_mean;
        noise_mean = self.noise_mean(scale);
        let mut niter = 1;
        while (last_mean - noise_mean).abs() > 1e-3 && niter < maxiter {
            niter += 1;
            last_mean = noise_mean;
            noise_mean = self.noise_mean(scale);
            self.deduct_intensity_from_all_windows(noise_mean);
        }
        first_mean - noise_mean
    }
}

impl<'lifespan> ops::Index<usize> for NoiseRegion<'lifespan> {
    type Output = Window<'lifespan>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.windows[index]
    }
}

impl<'lifespan> ops::IndexMut<usize> for NoiseRegion<'lifespan> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.windows[index]
    }
}

fn windowed_spectrum<'lifespan>(
    mz_array: &'lifespan [f64],
    intensity_array: &'lifespan mut [f32],
    window_size: f64,
) -> Vec<Window<'lifespan>> {
    let n = mz_array.len();
    let mut windows: Vec<Window<'lifespan>> = Vec::new();

    if n < 2 {
        return windows;
    }
    let mz_min = mz_array.first().unwrap();
    let mz_max = *mz_array.last().unwrap();

    let step_size = window_size / 2.0;
    let mut center_mz = mz_min + step_size;

    let mut partition = intensity_array;
    while center_mz < mz_max {
        let lo_mz = center_mz - step_size;
        let hi_mz = center_mz + step_size;
        let (lo_i, hi_i) = search::find_between(mz_array, lo_mz, hi_mz);
        let mid_point = (mz_array[lo_i] + mz_array[hi_i]) / 2.0;
        let offset = {
            let mid = (hi_i + 1) - lo_i;
            if mid > partition.len() {
                partition.len()
            } else {
                mid
            }
        };
        let (chunk, rest) = partition.split_at_mut(offset);
        partition = rest;
        if lo_mz <= mid_point && mid_point <= hi_mz {
            windows.push(Window::new(
                &mz_array[lo_i..hi_i + 1],
                chunk,
                lo_i,
                hi_i,
                center_mz,
                10,
                false,
            ));
        } else {
            windows.push(Window::new(
                &mz_array[lo_i..hi_i + 1],
                chunk,
                lo_i,
                hi_i,
                center_mz,
                0,
                true,
            ))
        }
        center_mz += window_size;
    }
    windows
}

fn group_windows_by_width<'lifespan>(
    windows: &'lifespan mut [Window<'lifespan>],
    width: u32,
) -> Vec<NoiseRegion<'lifespan>> {
    let step = if width > 2 { width / 2 } else { 1 };

    let mut result = Vec::new();

    let mut i = step;
    let n = windows.len();

    let mut partition = windows;
    while i < n as u32 {
        let lo = i - step;
        let hi = i + step;
        let mid = {
            let mid = hi - lo;
            if mid > partition.len() as u32 {
                partition.len()
            } else {
                mid as usize
            }
        };
        let pair = partition.split_at_mut(mid);
        let rest = pair.1;
        partition = rest;
        let subset: &'lifespan mut [Window<'lifespan>] = pair.0;
        let region = NoiseRegion::new(subset, width);
        result.push(region);
        i += 2 * step;
    }
    result
}

#[derive(Debug, Clone, Copy)]
pub enum DenoisingError {}

pub struct DenoisingArrayPair<'lifespan> {
    pub mz_array: &'lifespan [f64],
    pub intensity_array: &'lifespan mut [f32],
    pub scale: f32,
}

#[derive(Clone, Debug)]
pub struct SignalBackgroundDenoiser {
    pub window_size: f64,
    pub region_size: u32,
}

impl Default for SignalBackgroundDenoiser {
    fn default() -> SignalBackgroundDenoiser {
        SignalBackgroundDenoiser {
            window_size: 1.0,
            region_size: 10,
        }
    }
}

impl<'transient, 'lifespan: 'transient> SignalBackgroundDenoiser {
    pub fn prepare_spectrum(
        &self,
        mz_array: &'lifespan [f64],
        intensity_array: &'lifespan mut [f32],
        scale: f32,
    ) -> DenoisingArrayPair<'lifespan> {
        DenoisingArrayPair {
            mz_array,
            intensity_array,
            scale,
        }
    }

    pub fn denoise_inplace(
        &self,
        pair: &'lifespan mut DenoisingArrayPair,
    ) -> Result<f32, DenoisingError> {
        let mut windows = windowed_spectrum(pair.mz_array, pair.intensity_array, self.window_size);
        let mut regions = group_windows_by_width(&mut windows, self.region_size);
        let mut total = 0.0;
        let n = regions.len();
        for region in regions.iter_mut() {
            total += region.denoise(pair.scale, 10);
        }
        let average_noise_reduction = total / n as f32;
        Ok(average_noise_reduction)
    }

    pub fn denoise(
        &self,
        mz_array: &[f64],
        intensity_array: &mut [f32],
        scale: f32,
    ) -> Result<f32, DenoisingError> {
        let mut pair = self.prepare_spectrum(mz_array, intensity_array, scale);
        // average noise reduction
        self.denoise_inplace(&mut pair)
    }
}

/// Remove background noise from a spectrum **in-place**, returning the same slice of memory.
/// # Arguments
/// * `mz_array` - The m/z array for the spectrum. This _should_ be relatively evenly spaced for the
///                assumptions of this algorithm to work, so a profile spectrum is recommended.
/// * `intensity_array` - The intensity for each m/z in the spectrum. This buffer will be modified
///                       in place, removing background noise using the `MasSpike` algorithm and
///                       is carried forward as the return value.
/// * `scale` - The multiplicity of the noise to remove. When `scale` is small, local noise levels may
///             be exhausted in one window before the noise is appreciably depleted in the region, leading
///             to still-noisy spectra.
///
pub fn denoise<'a>(
    mz_array: &'a [f64],
    intensity_array: &'a mut [f32],
    scale: f32,
) -> Result<&'a [f32], DenoisingError> {
    let denoiser = SignalBackgroundDenoiser::default();
    match denoiser.denoise(mz_array, intensity_array, scale) {
        Ok(_noise) => Ok(intensity_array),
        Err(err) => Err(err),
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::peak_picker::PeakPicker;
    use crate::test_data::{NOISE, X, Y};

    use std::fs;
    use std::io;
    use std::io::prelude::*;

    #[test]
    fn test_denoise() -> io::Result<()> {
        let mut yhat: Vec<f32> = Y
            .iter()
            .zip(NOISE.iter())
            .map(|(y, e)| y * 1.0 + e)
            .collect();

        let mut acc = Vec::new();
        let mut picker = PeakPicker::default();
        picker.signal_to_noise_threshold = 3.0;
        picker.discover_peaks(&X, &yhat, &mut acc).unwrap();
        assert_eq!(acc.len(), 10);

        let denoiser = SignalBackgroundDenoiser::default();
        denoiser.denoise(&X, &mut yhat, 5.0).unwrap();
        let mut acc2 = Vec::new();
        picker.discover_peaks(&X, &yhat, &mut acc2).unwrap();
        assert_eq!(acc2.len(), 2);
        Ok(())
    }
}
