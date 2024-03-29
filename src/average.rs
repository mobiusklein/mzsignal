//! Re-bin a single spectrum, or average together multiple spectra using
//! interpolation.
//!
use std::borrow::Cow;
use std::cmp;
use std::collections::VecDeque;
use std::ops::{Add, Index};

#[cfg(feature = "parallelism")]
use rayon::prelude::*;
#[cfg(feature = "parallelism")]
use std::sync::Mutex;

use cfg_if;

use crate::arrayops::{gridspace, ArrayPair, MZGrid};
use crate::search;
use num_traits::{Float, ToPrimitive};

/// A linear interpolation spectrum intensity averager over a shared m/z axis.
#[derive(Debug, Default, Clone)]
pub struct SignalAverager<'lifespan> {
    /// The evenly spaced m/z axis over which spectra are averaged.
    pub mz_grid: Vec<f64>,
    /// The lowest m/z in the spectrum. If an input spectrum has lower m/z values, they will be ignored.
    pub mz_start: f64,
    /// The highest m/z in the spectrum. If an input spectrum has higher m/z values, they will be ignored.
    pub mz_end: f64,
    /// The spacing between m/z values in `mz_grid`. This value should be chosen relative to the sharpness
    /// of the peak shape of the mass analyzer used, but the smaller it is, the more computationally intensive
    /// the averaging process is, and the more memory it consumes.
    pub dx: f64,
    /// The current set of spectra to be averaged together. This uses a deque because the usecase of
    /// pushing spectra into an averaging window while removing them from the other side fits with the
    /// way one might average spectra over time.
    pub array_pairs: VecDeque<ArrayPair<'lifespan>>,
}

impl<'a, 'b: 'a> SignalAverager<'a> {
    pub fn new(mz_start: f64, mz_end: f64, dx: f64) -> SignalAverager<'a> {
        SignalAverager {
            mz_grid: gridspace(mz_start, mz_end, dx),
            mz_start,
            mz_end,
            dx,
            array_pairs: VecDeque::new(),
        }
    }

    /// Put `pair` into the queue of arrays being averaged together.
    pub fn push(&mut self, pair: ArrayPair<'b>) {
        self.array_pairs.push_back(pair)
    }

    /// Remove the least recently added array pair from the queue.
    pub fn pop(&mut self) -> Option<ArrayPair<'a>> {
        self.array_pairs.pop_front()
    }

    pub fn len(&self) -> usize {
        self.array_pairs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.array_pairs.is_empty()
    }

    /// Linear interpolation between two control points to find the intensity
    /// at a third point between them.
    ///
    /// # Arguments
    /// - `mz_j` - The first control point's m/z
    /// - `mz_x` - The interpolated m/z
    /// - `mz_j1` - The second control point's m/z
    /// - `inten_j` - The first control point's intensity
    /// - `inten_j1` - The second control point's intensity
    #[inline]
    pub fn interpolate_point(
        &self,
        mz_j: f64,
        mz_x: f64,
        mz_j1: f64,
        inten_j: f64,
        inten_j1: f64,
    ) -> f64 {
        ((inten_j * (mz_j1 - mz_x)) + (inten_j1 * (mz_x - mz_j))) / (mz_j1 - mz_j)
    }

    /// A linear interpolation across all spectra between `start_mz` and `end_mz`, with
    /// their intensities written into `out`.
    pub fn interpolate_into(&self, out: &mut [f32], start_mz: f64, end_mz: f64) -> usize {
        let offset = self.find_offset(start_mz);
        let stop_index = self.find_offset(end_mz);
        // debug_assert!(stop_index - offset == out.len());

        for block in self.array_pairs.iter() {
            for i in offset..stop_index {
                let x = self.mz_grid[i];
                let j = block.find(x);
                let mz_j = block.mz_array[j];

                let (mz_j, inten_j, mz_j1, inten_j1) = if (mz_j <= x) && ((j + 1) < block.len()) {
                    (
                        mz_j,
                        block.intensity_array[j],
                        block.mz_array[j + 1],
                        block.intensity_array[j + 1],
                    )
                } else if mz_j > x && j > 0 {
                    (
                        block.mz_array[j - 1],
                        block.intensity_array[j - 1],
                        block.mz_array[j],
                        block.intensity_array[j],
                    )
                } else {
                    continue;
                };
                let interp =
                    self.interpolate_point(mz_j, x, mz_j1, inten_j as f64, inten_j1 as f64);
                out[i - offset] += interp as f32;
            }
        }
        if self.array_pairs.len() > 1 {
            let normalizer = self.array_pairs.len() as f32;
            out.iter_mut().for_each(|y| *y /= normalizer);
        }
        stop_index - offset
    }

    pub fn interpolate_chunks(&self, n_chunks: usize) -> Vec<f32> {
        let mut result = self.create_intensity_array();
        if self.array_pairs.is_empty() {
            return result;
        }
        let n_points = self.points_between(self.mz_start, self.mz_end);

        let points_per_chunk = n_points / n_chunks;
        for i in 0..n_chunks {
            let offset = i * points_per_chunk;
            let (size, start_mz, end_mz) = if i == n_chunks - 1 {
                (n_points - offset, self.mz_grid[offset], self.mz_end)
            } else {
                (
                    points_per_chunk,
                    self.mz_grid[offset],
                    self.mz_grid[offset + points_per_chunk],
                )
            };
            let mut sub = self.create_intensity_array_of_size(size);
            self.interpolate_into(&mut sub, start_mz, end_mz);
            (result[offset..offset + size]).copy_from_slice(&sub);
        }
        result
    }

    #[cfg(feature = "parallelism")]
    #[allow(unused)]
    pub(crate) fn interpolate_chunks_parallel_locked(&'a self, n_chunks: usize) -> Vec<f32> {
        let result = self.create_intensity_array();
        if self.array_pairs.is_empty() {
            return result;
        }
        let n_points = self.points_between(self.mz_start, self.mz_end);
        let locked_result = Mutex::new(result);
        let points_per_chunk = n_points / n_chunks;
        (0..n_chunks).into_par_iter().for_each(|i| {
            let offset = i * points_per_chunk;
            let (size, start_mz, end_mz) = if i == n_chunks - 1 {
                (n_points - offset, self.mz_grid[offset], self.mz_end)
            } else {
                (
                    points_per_chunk,
                    self.mz_grid[offset],
                    self.mz_grid[offset + points_per_chunk],
                )
            };
            let mut sub = self.create_intensity_array_of_size(size);
            self.interpolate_into(&mut sub, start_mz, end_mz);

            let mut out = locked_result.lock().unwrap();
            (out[offset..offset + size]).copy_from_slice(&sub);
        });
        locked_result.into_inner().unwrap()
    }

    #[cfg(feature = "parallelism")]
    #[allow(unused)]
    pub(crate) fn interpolate_chunks_parallel(&'a self, n_chunks: usize) -> Vec<f32> {
        let mut result = self.create_intensity_array();
        if self.array_pairs.is_empty() {
            return result;
        }
        let n_points = self.points_between(self.mz_start, self.mz_end);
        let points_per_chunk = n_points / n_chunks;
        let mz_chunks: Vec<&[f64]> = self.mz_grid.chunks(points_per_chunk).collect();
        let mut intensity_chunks: Vec<&mut [f32]> = result.chunks_mut(points_per_chunk).collect();

        intensity_chunks[..]
            .par_iter_mut()
            .zip(mz_chunks[..].par_iter())
            .for_each(|(mut intensity_chunk, mz_chunk)| {
                let start_mz = mz_chunk.first().unwrap();
                // The + 1e-6 is just a gentle push to get interpolate_into to roll over to the last position in the chunk
                let end_mz = mz_chunk.last().unwrap() + 1e-6;
                self.interpolate_into(intensity_chunk, *start_mz, end_mz);
            });
        result
    }

    /// Allocate a new intensity array, [`interpolate_into`](SignalAverager::interpolate_into) it, and return it.
    pub fn interpolate(&'a self) -> Vec<f32> {
        let mut result = self.create_intensity_array();
        self.interpolate_into(&mut result, self.mz_start, self.mz_end);
        result
    }
}

impl<'lifespan> MZGrid for SignalAverager<'lifespan> {
    fn mz_grid(&self) -> &[f64] {
        &self.mz_grid
    }
}

impl<'lifespan> Extend<ArrayPair<'lifespan>> for SignalAverager<'lifespan> {
    fn extend<T: IntoIterator<Item = ArrayPair<'lifespan>>>(&mut self, iter: T) {
        self.array_pairs.extend(iter)
    }
}

// Can't inline cfg-if
cfg_if::cfg_if! {
    if #[cfg(feature = "parallelism")] {
        fn average_signal_inner(averager: &SignalAverager, n: usize) -> Vec<f32> {
            averager.interpolate_chunks_parallel(3 + n)
        }
    } else {
        fn average_signal_inner(averager: &SignalAverager, n: usize) -> Vec<f32> {
            averager.interpolate()
        }
    }
}

/// Average together signal from the slice of `ArrayPair`s with spacing `dx` and create
/// a new `ArrayPair` from it
pub fn average_signal<'lifespan, 'owned: 'lifespan>(signal: &[ArrayPair<'lifespan>], dx: f64) -> ArrayPair<'owned> {
    let (mz_min, mz_max) = signal.iter().fold((f64::infinity(), 0.0), |acc, x| {
        (
            if acc.0 < x.min_mz { acc.0 } else { x.min_mz },
            if acc.1 > x.max_mz { acc.1 } else { x.max_mz },
        )
    });
    let mut averager = SignalAverager::new(mz_min, mz_max, dx);
    averager
        .array_pairs
        .extend(signal.iter().map(|a| a.borrow()));
    let signal = average_signal_inner(&averager, signal.len());
    ArrayPair::new(Cow::Owned(averager.copy_mz_array()), Cow::Owned(signal))
}

pub fn rebin<'transient, 'lifespan: 'transient>(
    mz_array: &'lifespan [f64],
    intensity_array: &'lifespan [f32],
    dx: f64,
) -> ArrayPair<'transient> {
    let pair = [ArrayPair::from((mz_array, intensity_array))];
    average_signal(&pair, dx)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::peak_picker::PeakPicker;
    use crate::test_data::{X, Y};

    #[test]
    fn test_rebin_one() {
        let mut averager = SignalAverager::new(X[0], X[X.len() - 1], 0.00001);
        averager.push(ArrayPair::wrap(&X, &Y));
        let yhat = averager.interpolate();
        let picker = PeakPicker::default();
        let mut acc = Vec::new();
        picker
            .discover_peaks(&averager.mz_grid, &yhat, &mut acc)
            .expect("Signal can be picked");
        let mzs = [180.0633881, 181.06338858024316, 182.06338874740308];
        for (peak, mz) in acc.iter().zip(mzs.iter()) {
            assert!((peak.mz - mz).abs() < 1e-6);
            assert!(peak.intensity > 0.0);
        }
    }

    #[test]
    fn test_rebin_chunked() {
        let mut averager = SignalAverager::new(X[0], X[X.len() - 1], 0.00001);
        averager.push(ArrayPair::wrap(&X, &Y));
        let yhat = averager.interpolate_chunks(3);
        let picker = PeakPicker::default();
        let mut acc = Vec::new();
        picker
            .discover_peaks(&averager.mz_grid, &yhat, &mut acc)
            .expect("Signal can be picked");
        let mzs = [180.0633881, 181.06338858024316, 182.06338874740308];
        for (peak, mz) in acc.iter().zip(mzs.iter()) {
            assert!((peak.mz - mz).abs() < 1e-6);
        }
    }

    #[test]
    #[cfg(feature = "parallelism")]
    fn test_rebin_parallel_locked() {
        let mut averager = SignalAverager::new(X[0], X[X.len() - 1], 0.00001);
        averager.push(ArrayPair::wrap(&X, &Y));
        let yhat = averager.interpolate_chunks_parallel_locked(6);
        let picker = PeakPicker::default();
        let mut acc = Vec::new();
        picker
            .discover_peaks(&averager.mz_grid, &yhat, &mut acc)
            .expect("Signal can be picked");
        let mzs = [180.0633881, 181.06338858024316, 182.06338874740308];
        for (peak, mz) in acc.iter().zip(mzs.iter()) {
            assert!((peak.mz - mz).abs() < 1e-6);
        }
    }

    #[test]
    #[cfg(feature = "parallelism")]
    fn test_rebin_parallel() {
        let mut averager = SignalAverager::new(X[0], X[X.len() - 1], 0.00001);
        averager.push(ArrayPair::wrap(&X, &Y));
        let yhat = averager.interpolate_chunks_parallel(6);
        let picker = PeakPicker::default();
        let mut acc = Vec::new();
        picker
            .discover_peaks(&averager.mz_grid, &yhat, &mut acc)
            .expect("Signal can be picked");
        let mzs = [180.0633881, 181.06338858024316, 182.06338874740308];
        for (peak, mz) in acc.iter().zip(mzs.iter()) {
            assert!((peak.mz - mz).abs() < 1e-6);
        }
    }
}
