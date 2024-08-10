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

use mzpeaks::coordinate::{CoordinateLike, Time};

use crate::arrayops::{gridspace, ArrayPair, ArrayPairSplit, MZGrid};
use crate::search;
use num_traits::{Float, Saturating, ToPrimitive};

trait MZInterpolator: MZGrid {
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
    fn interpolate_point(
        &self,
        mz_j: f64,
        mz_x: f64,
        mz_j1: f64,
        inten_j: f64,
        inten_j1: f64,
    ) -> f64 {
        ((inten_j * (mz_j1 - mz_x)) + (inten_j1 * (mz_x - mz_j))) / (mz_j1 - mz_j)
    }
}

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
        let grid_size = self.mz_grid.len();
        assert!(offset < grid_size || grid_size == 0);
        assert!(stop_index <= grid_size);
        assert!((stop_index - offset) == out.len());

        for block in self.array_pairs.iter() {
            if block.is_empty() {
                continue;
            }
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

    pub fn interpolate_between(&'a self, mz_start: f64, mz_end: f64) -> (Vec<f32>, (usize, usize)) {
        let (n_points, (start, end)) = self.points_between_with_indices(mz_start, mz_end);
        let mut result = self.create_intensity_array_of_size(n_points);
        self.interpolate_into(&mut result, mz_start, mz_end);
        (result, (start, end))
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
impl<'lifespan> MZInterpolator for SignalAverager<'lifespan> {}

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
        fn average_signal_inner(averager: &SignalAverager, _n: usize) -> Vec<f32> {
            averager.interpolate()
        }
    }
}

/// Average together signal from the slice of `ArrayPair`s with spacing `dx` and create
/// a new `ArrayPair` from it
pub fn average_signal<'lifespan, 'owned: 'lifespan>(
    signal: &[ArrayPair<'lifespan>],
    dx: f64,
) -> ArrayPair<'owned> {
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

/// A segment over a signal array pair
#[derive(Debug, Default, Clone, Copy)]
pub struct Segment {
    start: usize,
    end: usize,
}

/// An [`ArrayPair`] with an associated list of [`Segment`], an associated cached interpolated intensity array
/// and a time index
#[derive(Debug, Default, Clone)]
pub struct ArrayPairWithSegments<'a> {
    /// The original signal
    pub array_pair: ArrayPair<'a>,
    /// The segments over the interpolated coordinate system that there was signal for
    pub segments: Vec<Segment>,
    /// The interpolated signal
    pub intensity_array: Vec<f32>,
    /// The time point associated with this signal
    pub time: f64,
}

impl<'a> PartialEq for ArrayPairWithSegments<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.time == other.time && self.intensity_array == other.intensity_array
    }
}

impl<'a> PartialOrd for ArrayPairWithSegments<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(&other))
    }
}

impl<'a> Eq for ArrayPairWithSegments<'a> {}

impl<'a> Ord for ArrayPairWithSegments<'a> {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.time.total_cmp(&other.time)
    }
}

impl<'a> CoordinateLike<Time> for ArrayPairWithSegments<'a> {
    fn coordinate(&self) -> f64 {
        self.time
    }
}

/// A linear interpolation spectrum intensity averager over a shared m/z axis that pre-computes
/// a segmented intensity grid.
#[derive(Debug, Default, Clone)]
pub struct SegmentGridSignalAverager<'lifespan> {
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
    /// The current set of spectra to be averaged over
    pub array_pairs: Vec<ArrayPairWithSegments<'lifespan>>,
}

impl<'lifespan> Extend<(f64, ArrayPair<'lifespan>)> for SegmentGridSignalAverager<'lifespan> {
    fn extend<T: IntoIterator<Item = (f64, ArrayPair<'lifespan>)>>(&mut self, iter: T) {
        for (time, block) in iter {
            self.push(time, block)
        }
    }
}

impl<'a, 'lifespan: 'a> SegmentGridSignalAverager<'lifespan> {
    pub fn new(mz_start: f64, mz_end: f64, dx: f64) -> Self {
        Self {
            mz_grid: gridspace(mz_start, mz_end, dx),
            mz_start,
            mz_end,
            dx,
            array_pairs: Vec::new(),
        }
    }

    pub fn from_iter<I: Iterator<Item=(f64, ArrayPair<'lifespan>)>>(mz_start: f64, mz_end: f64, dx: f64, iter: I) -> Self {
        let mut inst = Self::new(mz_start, mz_end, dx);
        inst.extend(iter);
        inst
    }

    pub fn find_time(&self, time: f64) -> Option<usize> {
        let i = self
            .array_pairs
            .binary_search_by(|block| block.time.total_cmp(&time));
        match i {
            Ok(i) => Some(i),
            Err(i) => (i.saturating_sub(2)..(i + 2).min(self.array_pairs.len()))
                .into_iter()
                .min_by(|i, j| {
                    let err_i = self
                        .array_pairs
                        .get(*i)
                        .map(|block| (block.time - time).abs())
                        .unwrap_or(f64::INFINITY);
                    let err_j = self
                        .array_pairs
                        .get(*j)
                        .map(|block| (block.time - time).abs())
                        .unwrap_or(f64::INFINITY);
                    err_i.total_cmp(&err_j)
                }),
        }
    }

    pub fn time_at(&self, index: usize) -> Option<f64> {
        self.array_pairs.get(index).map(|block| block.time)
    }

    pub fn push(&mut self, time: f64, block: ArrayPair<'lifespan>) {
        let block = self.populate_intensity_axis(time, block);
        self.push_block(block);
    }

    fn push_block(&mut self, block: ArrayPairWithSegments<'lifespan>) {
        if let Some(time) = self.array_pairs.last().map(|block| block.time) {
            if time < block.time {
                self.array_pairs.push(block)
            } else {
                self.array_pairs.push(block);
                self.array_pairs.sort();
            }
        } else {
            self.array_pairs.push(block)
        }
    }

    pub fn len(&self) -> usize {
        self.array_pairs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.array_pairs.is_empty()
    }

    fn populate_intensity_axis(
        &self,
        time: f64,
        block: ArrayPair<'lifespan>,
    ) -> ArrayPairWithSegments<'lifespan> {
        let mut segments = Vec::default();
        if block.is_empty() {
            return ArrayPairWithSegments {
                array_pair: block,
                segments: segments,
                intensity_array: Vec::new(),
                time,
            }
        }
        let mut segment = Segment::default();
        let mut opened = false;

        let n = self.mz_grid.len();

        let mut intensity_axis_ = self.create_intensity_array();
        let intensity_axis = &mut intensity_axis_[0..n];

        for (i, x) in self.mz_grid.iter().copied().enumerate() {
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
                    mz_j,
                    block.intensity_array[j],
                )
            } else {
                continue;
            };
            let interp = self.interpolate_point(mz_j, x, mz_j1, inten_j as f64, inten_j1 as f64);
            intensity_axis[i] = interp as f32;
            if interp > 0.0 {
                if opened {
                    segment.end = i;
                } else {
                    segment.start = i;
                    opened = true;
                }
            } else {
                if opened {
                    segment.end = i;
                    opened = false;
                    segments.push(segment);
                    segment = Segment::default();
                }
            }
        }
        if opened {
            segment.end = n;
            segments.push(segment);
        }
        ArrayPairWithSegments {
            array_pair: block,
            segments: segments,
            intensity_array: intensity_axis_,
            time,
        }
    }

    pub fn iter(&'a self, width: usize) -> SegmentGridSignalAveragerIter<'a> {
        SegmentGridSignalAveragerIter {
            averager: self,
            index: 0,
            width,
        }
    }

    pub fn average_over(&'a self, time: f64, width: usize) -> ArrayPairSplit<'a, 'static> {
        if let Some(i) = self.find_time(time) {
            self.average_over_index(i, width)
        } else {
            ArrayPairSplit::default()
        }
    }

    pub fn average_over_index(&'a self, index: usize, width: usize) -> ArrayPairSplit<'a, 'static> {
        let blocks = &self.array_pairs
            [index.saturating_sub(width)..(index + width).min(self.array_pairs.len())];
        self.average_segments(blocks)
    }

    pub fn average_segments(
        &'a self,
        segments: &[ArrayPairWithSegments],
    ) -> ArrayPairSplit<'a, 'static> {
        let (offset, end) = segments.iter().fold((usize::MAX, 0), |(start, end), seg| {
            let start = seg
                .segments
                .first()
                .map(|s| start.min(s.start))
                .unwrap_or(start);
            let end = seg.segments.last().map(|s| end.max(s.end)).unwrap_or(end);
            (start, end)
        });

        if offset >= end {
            return (Vec::new(), Vec::new()).into();
        }

        let mut intensity_array = self.create_intensity_array_of_size(end - offset);

        for seg in segments.iter() {
            for sg in seg.segments.iter() {
                for i in sg.start..sg.end {
                    intensity_array[i.saturating_sub(offset)] += seg.intensity_array[i];
                }
            }
        }

        let mz_array = Cow::Borrowed(&self.mz_grid[offset..end]);

        ArrayPairSplit::new(mz_array, Cow::Owned(intensity_array))
    }
}

impl<'lifespan> MZGrid for SegmentGridSignalAverager<'lifespan> {
    fn mz_grid(&self) -> &[f64] {
        &self.mz_grid
    }
}
impl<'lifespan> MZInterpolator for SegmentGridSignalAverager<'lifespan> {}

#[derive(Debug)]
pub struct SegmentGridSignalAveragerIter<'lifespan> {
    averager: &'lifespan SegmentGridSignalAverager<'lifespan>,
    width: usize,
    index: usize,
}

impl<'lifespan> Iterator for SegmentGridSignalAveragerIter<'lifespan> {
    type Item = (f64, ArrayPairSplit<'lifespan, 'static>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.averager.len() {
            let block = self.averager.average_over_index(self.index, self.width);
            let time = self.averager.time_at(self.index).unwrap();
            self.index += 1;
            Some((time, block))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test {
    use std::io;

    use mzpeaks::MZPeakSetType;

    use super::*;
    use crate::peak_picker::PeakPicker;
    use crate::test_data::{X, Y};
    use crate::FittedPeak;

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
        let mzs = [180.0633881, 181.06387399204235, 182.06404644991485];
        for (i, (peak, mz)) in acc.iter().zip(mzs.iter()).enumerate() {
            let diff = peak.mz - mz;
            assert!((peak.mz - mz).abs() < 1e-6, "Diff {} on peak {i}", diff);
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
        let mzs = [180.0633881, 181.06387399204235, 182.06404644991485];
        for (i, (peak, mz)) in acc.iter().zip(mzs.iter()).enumerate() {
            let diff = peak.mz - mz;
            assert!((peak.mz - mz).abs() < 1e-6, "Diff {} on peak {i}", diff);
            assert!(peak.intensity > 0.0);
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
        let mzs = [180.0633881, 181.06387399204235, 182.06404644991485];
        for (i, (peak, mz)) in acc.iter().zip(mzs.iter()).enumerate() {
            let diff = peak.mz - mz;
            assert!((peak.mz - mz).abs() < 1e-6, "Diff {} on peak {i}", diff);
            assert!(peak.intensity > 0.0);
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
        let mzs = [180.0633881, 181.06387399204235, 182.06404644991485];
        for (i, (peak, mz)) in acc.iter().zip(mzs.iter()).enumerate() {
            let diff = peak.mz - mz;
            assert!((peak.mz - mz).abs() < 1e-6, "Diff {} on peak {i}", diff);
            assert!(peak.intensity > 0.0);
        }
    }

    #[test_log::test]
    fn test_segment_grid() -> io::Result<()> {
        use crate::text::arrays_over_time_from_file;
        let time_arrays = arrays_over_time_from_file("./test/data/peaks_over_time.txt")?;

        let mut averager = SegmentGridSignalAverager::new(200.0, 2000.0, 0.001);
        let reprofiler = crate::reprofile::PeakSetReprofiler::new(200.0, 2000.0, 0.001);

        for (_, (t, row)) in time_arrays.into_iter().enumerate().take(5) {
            // log::info!("{i}: {t} with {} peaks", row.len());
            let peaks: MZPeakSetType<FittedPeak> = row
                .mz_array
                .into_iter()
                .zip(row.intensity_array.into_iter())
                .map(|(mz, i)| FittedPeak::new(*mz, *i, 0, *i, 0.005))
                .collect();

            // log::info!("Reprofiling");
            let peak_models =
                reprofiler.build_peak_shape_models(&peaks.as_slice(), crate::reprofile::PeakShape::Gaussian);
            let block = reprofiler.reprofile_from_models(&peak_models);

            averager.push(t, block);
        }

        // log::info!("Start averaging");
        let views: Vec<_> = averager.iter(1).collect();
        assert_eq!(views.len(), 5);

        views.iter().for_each(|(_, block)| {
            assert!(block.intensity_array.iter().all(|i| *i >= 0.0));
        });
        Ok(())
    }
}
