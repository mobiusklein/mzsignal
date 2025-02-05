//! Re-bin a single spectrum, or average together multiple spectra using
//! interpolation.
//!
use std::borrow::Cow;
use std::cmp;
use std::collections::VecDeque;

#[cfg(target_arch = "x86")]
use std::arch::x86::__m256d;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::__m256d;
#[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
struct __m256d();

#[cfg(feature = "parallelism")]
use rayon::prelude::*;
#[cfg(feature = "parallelism")]
use std::sync::Mutex;

use cfg_if;

use mzpeaks::coordinate::{CoordinateLike, Time};

use crate::arrayops::{gridspace, ArrayPair, ArrayPairIter, ArrayPairLike, ArrayPairSplit, MZGrid};
use num_traits::Float;

trait MZInterpolator {
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
        // ((inten_j * (mz_j1 - mz_x)) + (inten_j1 * (mz_x - mz_j))) / (mz_j1 - mz_j)
        let step_a = mz_j1 - mz_x;
        let step_b = mz_x - mz_j;
        let step_ab = mz_j1 - mz_j;
        let vb = inten_j1 * step_b;
        let vab = inten_j.mul_add(step_a, vb);
        vab / step_ab
    }

    // A version of [`MZInterpolator::interpolate_point`] that uses AVX 256-bit register operations
    #[cfg(feature = "avx")]
    #[cfg(target_arch = "x86_64")]
    fn interpolate_avx(
        &self,
        mz_j: __m256d,
        mz_x: __m256d,
        mz_j1: __m256d,
        inten_j: __m256d,
        inten_j1: __m256d,
    ) -> __m256d {
        unsafe {
            use std::arch::x86_64::*;
            let step_a = _mm256_sub_pd(mz_j1, mz_x);
            let step_b = _mm256_sub_pd(mz_x, mz_j);
            let step_ab = _mm256_sub_pd(mz_j1, mz_j);
            let vb = _mm256_mul_pd(inten_j1, step_b);
            let vab = _mm256_fmadd_pd(inten_j, step_a, vb);
            _mm256_div_pd(vab, step_ab)
        }
    }
}


struct Interpolator {}
impl MZInterpolator for Interpolator {}

pub fn interpolate(xj: f64, x: f64, xj1: f64, yj: f64, yj1: f64) -> f64 {
    Interpolator{}.interpolate_point(xj, x, xj1, yj, yj1)
}

struct MonotonicBlockSearcher<'a> {
    data: &'a ArrayPair<'a>,
    next_value: Option<f64>,
    last_index: usize,
}

impl<'a> MonotonicBlockSearcher<'a> {
    fn new(data: &'a ArrayPair<'a>) -> Self {
        Self {
            data,
            next_value: None,
            last_index: 0,
        }
    }

    fn find_update(&mut self, mz: f64) -> usize {
        let i = self.data.find(mz);
        self.last_index = i;
        self.next_value = self.data.mz_array.get(i).copied();
        i
    }

    /// This assumes that the next value will be suitable, but this is not actually
    /// true. The algorithm this component is used in though does not make the distinction
    #[allow(unused)]
    fn peek(&self, mz: f64) -> usize {
        if let Some(next_value) = self.next_value {
            if mz < next_value {
                self.last_index
            } else {
                (self.last_index + 1).min(self.data.len().saturating_sub(1))
            }
        } else {
            self.last_index
        }
    }

    fn find(&mut self, mz: f64) -> usize {
        if let Some(next_value) = self.next_value {
            if mz < next_value {
                self.last_index
            } else {
                self.find_update(mz)
            }
        } else {
            self.find_update(mz)
        }
    }
}

#[allow(unused)]
struct MonotonicBlockedIterator<'a, 'b: 'a, T: Iterator<Item = (f64, &'b mut f32)>> {
    block: std::iter::Enumerate<ArrayPairIter<'a>>,
    last_value: (usize, (f64, f64)),
    current_value: (usize, (f64, f64)),
    next_value: Option<(usize, (f64, f64))>,
    block_n: usize,
    it: T,
}

impl<'b, T: Iterator<Item = (f64, &'b mut f32)>> MZInterpolator
    for MonotonicBlockedIterator<'_, 'b, T>
{
}

type BlockIteratorPoint = (usize, (f64, f64));

impl<'a, 'b: 'a, T: Iterator<Item = (f64, &'b mut f32)>> MonotonicBlockedIterator<'a, 'b, T> {
    fn new(block: &'a ArrayPair<'a>, it: T) -> Self {
        let mut source = block.iter().enumerate();
        let current_value = source.next().map(|(i, (x, y))| (i, (x, y as f64))).unwrap();
        let next_value = source.next().map(|(i, (x, y))| (i, (x, y as f64)));
        let block_n = block.len();
        Self {
            block: source,
            last_value: current_value,
            current_value,
            next_value,
            block_n,
            it,
        }
    }

    fn next_value_from_source(&mut self) -> Option<BlockIteratorPoint> {
        self.block.next().map(|(i, (x, y))| (i, (x, y as f64)))
    }

    fn step(&mut self) -> Option<(f64, &'b mut f32, BlockIteratorPoint)> {
        if let Some((x, o)) = self.it.next() {
            if let Some((vi, (vmz, vint))) = self.next_value.as_ref() {
                if x >= *vmz {
                    self.last_value = self.current_value;
                    self.current_value = (*vi, (*vmz, *vint));
                    self.next_value = self.next_value_from_source();
                }
                Some((x, o, self.current_value))
            } else {
                Some((x, o, self.current_value))
            }
        } else {
            None
        }
    }

    fn interpolant_step(&mut self) -> Option<(f64, &'b mut f32)> {
        if let Some((mz, o, (_, (mz_j, inten_j)))) = self.step() {
            if mz_j <= mz {
                if let Some((_, (mz_j1, inten_j1))) = self.next_value {
                    let inten = self.interpolate_point(mz_j, mz, mz_j1, inten_j, inten_j1);
                    *o += inten as f32;
                    Some((mz, o))
                } else {
                    let (mz_j1, inten_j1) = (mz_j, inten_j);
                    let (_, (mz_j, inten_j)) = self.last_value;

                    let inten = self.interpolate_point(mz_j, mz, mz_j1, inten_j, inten_j1);
                    *o += inten as f32;
                    Some((mz, o))
                }
            } else {
                let (mz_j1, inten_j1) = (mz_j, inten_j);
                let (_, (mz_j, inten_j)) = self.last_value;

                let inten = self.interpolate_point(mz_j, mz, mz_j1, inten_j, inten_j1);
                *o += inten as f32;
                Some((mz, o))
            }
        } else {
            None
        }
    }
}

impl<'a, 'b: 'a, T: Iterator<Item = (f64, &'b mut f32)>> Iterator
    for MonotonicBlockedIterator<'a, 'b, T>
{
    type Item = (f64, &'b mut f32);

    fn next(&mut self) -> Option<Self::Item> {
        self.interpolant_step()
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

    /// A linear interpolation across all spectra between `start_mz` and `end_mz`, with
    /// their intensities written into `out`.
    pub(crate) fn interpolate_into_iter(
        &self,
        out: &mut [f32],
        start_mz: f64,
        end_mz: f64,
    ) -> usize {
        let offset = self.find_offset(start_mz);
        let stop_index = self.find_offset(end_mz);

        let grid_size = self.mz_grid.len();
        assert!(offset < grid_size || grid_size == 0);
        assert!(stop_index <= grid_size);
        assert!((stop_index - offset) == out.len());

        let grid_slice = &self.mz_grid[offset..stop_index];
        for block in self.array_pairs.iter() {
            if block.is_empty() {
                continue;
            }

            let start_idx = block.find(start_mz).saturating_sub(1);
            let block_slice = block.slice(start_idx, block.len());

            let it = MonotonicBlockedIterator::new(
                &block_slice,
                grid_slice.iter().copied().zip(out.iter_mut()),
            );
            let _traveled = it.count();
        }
        if self.array_pairs.len() > 1 {
            let normalizer = self.array_pairs.len() as f32;
            out.iter_mut().for_each(|y| *y /= normalizer);
        }
        stop_index - offset
    }

    #[inline(always)]
    /// Get the first and second control points' m/z and intensity values,
    /// (mz, inten, mz1, inten1), in ascendng m/z order around `x`
    fn get_interpolation_values(
        &self,
        x: f64,
        j: usize,
        mz_j: f64,
        block_n: usize,
        block_mz_array: &[f64],
        block_intensity_array: &[f32],
    ) -> Option<(f64, f64, f64, f64)> {
        let js1 = j + 1;
        if (mz_j <= x) && (js1 < block_n) {
            Some((
                mz_j,
                block_intensity_array[j] as f64,
                block_mz_array[js1],
                block_intensity_array[js1] as f64,
            ))
        } else if mz_j > x && j > 0 {
            let js1 = j - 1;
            Some((
                block_mz_array[js1],
                block_intensity_array[js1] as f64,
                block_mz_array[j],
                block_intensity_array[j] as f64,
            ))
        } else {
            None
        }
    }

    #[inline(always)]
    fn interpolate_into_idx_seq(
        &self,
        grid_mzs: &[f64],
        out: &mut [f32],
        block_mz_array: &[f64],
        block_intensity_array: &[f32],
        block_n: usize,
        block_searcher: &mut MonotonicBlockSearcher,
    ) {
        for (x, o) in grid_mzs.iter().copied().zip(out.iter_mut()) {
            let j = block_searcher.find(x);
            let mz_j = block_mz_array[j];

            if let Some((mz_j, inten_j, mz_j1, inten_j1)) = self.get_interpolation_values(
                x,
                j,
                mz_j,
                block_n,
                block_mz_array,
                block_intensity_array,
            ) {
                let interp = self.interpolate_point(mz_j, x, mz_j1, inten_j, inten_j1);
                *o += interp as f32;
            }
        }
    }

    #[inline(always)]
    fn interpolate_into_idx_lanes_fallback<const LANES: usize>(
        &self,
        grid_mz_block: &[f64],
        output_intensity_block: &mut [f32],
        block_mz_array: &[f64],
        block_intensity_array: &[f32],
        block_n: usize,
        block_searcher: &mut MonotonicBlockSearcher,
    ) {
        assert_eq!(grid_mz_block.len(), LANES);
        assert_eq!(output_intensity_block.len(), LANES);
        for lane_i in 0..LANES {
            let grid_mz = grid_mz_block[lane_i];
            let output_intensity = &mut output_intensity_block[lane_i];
            let mz_index_of_x = block_searcher.find(grid_mz);
            let mz_j = block_mz_array[mz_index_of_x];

            if let Some((mz_j, inten_j, mz_j1, inten_j1)) = self.get_interpolation_values(
                grid_mz,
                mz_index_of_x,
                mz_j,
                block_n,
                block_mz_array,
                block_intensity_array,
            ) {
                let interp = self.interpolate_point(mz_j, grid_mz, mz_j1, inten_j, inten_j1);
                *output_intensity += interp as f32;
            }
        }
    }

    #[cfg(feature = "avx")]
    fn normalize_intensity_by_scan_count_avx(&self, out: &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx") {
            // Use AVX SIMD instructions available on x86_64 CPUs to process up to eight steps at a time.
            unsafe {
                use std::arch::x86_64::*;
                const LANES: usize = 8;
                let normalizer = self.array_pairs.len() as f32;
                let normalizer_v8 = _mm256_broadcast_ss(&normalizer);
                let mut chunks_it = out.chunks_exact_mut(LANES);
                for chunk in chunks_it.by_ref() {
                    let o_v8: __m256 = _mm256_loadu_ps(chunk.as_ptr());
                    let o_normalized_v8 = _mm256_div_ps(o_v8, normalizer_v8);
                    _mm256_storeu_ps(chunk.as_mut_ptr(), o_normalized_v8);
                }
                for o in chunks_it.into_remainder() {
                    *o /= normalizer;
                }
            }
        } else {
            self.normalize_intensity_by_scan_count_fallback(out)
        }
        #[cfg(not(target_arch = "x86_64"))]
        self.normalize_intensity_by_scan_count_fallback(out);
    }

    fn normalize_intensity_by_scan_count_fallback(&self, out: &mut [f32]) {
        let normalizer = self.array_pairs.len() as f32;

        const LANES: usize = 8;
        let mut it = out.chunks_exact_mut(LANES);

        for chunk in it.by_ref() {
            // Make it obvious to the compiler to vectorize
            #[allow(clippy::needless_range_loop)]
            for i in 0..LANES {
                chunk[i] /= normalizer;
            }
        }
        it.into_remainder()
            .iter_mut()
            .for_each(|y| *y /= normalizer);
    }

    fn normalize_intensity_by_scan_count(&self, out: &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx") {
            #[cfg(feature = "avx")]
            self.normalize_intensity_by_scan_count_avx(out);
            #[cfg(not(feature = "avx"))]
            self.normalize_intensity_by_scan_count_fallback(out);
        } else {
            self.normalize_intensity_by_scan_count_fallback(out);
        }
        #[cfg(not(target_arch = "x86_64"))]
        self.normalize_intensity_by_scan_count_fallback(out);
    }

    pub(crate) fn interpolate_into_idx(
        &self,
        out: &mut [f32],
        start_mz: f64,
        end_mz: f64,
    ) -> usize {
        let offset = self.find_offset(start_mz);
        let stop_index = self.find_offset(end_mz);

        let grid_size = self.mz_grid.len();
        {
            assert!(offset < grid_size || grid_size == 0);
            assert!(stop_index <= grid_size);
            assert!((stop_index - offset) == out.len());
        }

        let grid_slice = &self.mz_grid[offset..stop_index];

        for block in self.array_pairs.iter() {
            if block.is_empty() {
                continue;
            }
            let mut block_searcher = MonotonicBlockSearcher::new(block);
            let block_n = block.len();
            let block_mz_array = block.mz_array.as_ref();
            let block_intensity_array = block.intensity_array.as_ref();
            assert_eq!(block_mz_array.len(), block_n);
            assert_eq!(block_intensity_array.len(), block_n);

            const LANES: usize = 4;
            let mut grid_chunks = grid_slice.chunks_exact(LANES);
            let mut out_chunks = out.chunks_exact_mut(LANES);

            while let (Some(grid_mz_block), Some(output_intensity_block)) =
                (grid_chunks.next(), out_chunks.next())
            {
                #[cfg(not(target_arch = "x86_64"))]
                let did_vector = false;
                #[cfg(target_arch = "x86_64")]
                let did_vector = if std::arch::is_x86_feature_detected!("avx") {
                    #[cfg(not(feature = "avx"))]
                    {
                        false
                    }
                    #[cfg(feature = "avx")]
                    // Use AVX SIMD instructions available on x86_64 CPUs to process up to four steps at a time.
                    unsafe {
                        use std::arch::x86_64::*;
                        let grid_mz_first = *grid_mz_block.get_unchecked(0);
                        let grid_mz_last = *grid_mz_block.get_unchecked(3);
                        let j_first = block_searcher.find(grid_mz_first);
                        let j_last = block_searcher.peek(grid_mz_last);
                        let mz_j_first = *block_mz_array.get_unchecked(j_first);

                        // If the solution uses the same two control points for every comparison, as given by both
                        // using the same first point in the block, then we can take this fast path that performs
                        // the interpolation operation using AVX and 256-bit vector instructions.
                        //
                        // This could also be done with the AVX 512-bit vectors but they are not available on most
                        // machines yet.
                        if j_first == j_last {
                            if let Some((mz_j, inten_j, mz_j1, inten_j1)) = self
                                .get_interpolation_values(
                                    grid_mz_first,
                                    j_first,
                                    mz_j_first,
                                    block_n,
                                    block_mz_array,
                                    block_intensity_array,
                                )
                            {
                                // Populate the vectors going into `interpolate_avx`
                                let mz_x_v4: __m256d = _mm256_loadu_pd(grid_mz_block.as_ptr());
                                let mz_j_v4: __m256d = _mm256_broadcast_sd(&mz_j);
                                let mz_j1_v4: __m256d = _mm256_broadcast_sd(&mz_j1);
                                let inten_j_v4: __m256d = _mm256_broadcast_sd(&inten_j);
                                let inten_j1_v4: __m256d = _mm256_broadcast_sd(&inten_j1);

                                // Perform the interpolation on the vector registers
                                let result_v4 = self.interpolate_avx(
                                    mz_j_v4,
                                    mz_x_v4,
                                    mz_j1_v4,
                                    inten_j_v4,
                                    inten_j1_v4,
                                );

                                // Cast down from f64 to f32 registers
                                let result_v4_f32 = _mm256_cvtpd_ps(result_v4);
                                // Load the accumulator from the output array of f32
                                let acc_v4 = _mm_loadu_ps(output_intensity_block.as_ptr());
                                // Add the result to the accumulator
                                let total_v4 = _mm_add_ps(result_v4_f32, acc_v4);
                                // Store the accumulator back to the array of f32
                                _mm_storeu_ps(output_intensity_block.as_mut_ptr(), total_v4);
                            }
                            true
                        } else {
                            false
                        }
                    }
                } else {
                    false
                };
                if !did_vector {
                    #[allow(clippy::if_same_then_else)] // hint to the compiler that a hardware feature will be available
                    #[cfg(target_arch = "x86_64")]
                    if std::arch::is_x86_feature_detected!("avx") {
                        self.interpolate_into_idx_lanes_fallback::<LANES>(
                            grid_mz_block,
                            output_intensity_block,
                            block_mz_array,
                            block_intensity_array,
                            block_n,
                            &mut block_searcher,
                        );
                    } else {
                        self.interpolate_into_idx_lanes_fallback::<LANES>(
                            grid_mz_block,
                            output_intensity_block,
                            block_mz_array,
                            block_intensity_array,
                            block_n,
                            &mut block_searcher,
                        );
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    self.interpolate_into_idx_lanes_fallback::<LANES>(
                        grid_mz_block,
                        output_intensity_block,
                        block_mz_array,
                        block_intensity_array,
                        block_n,
                        &mut block_searcher,
                    );
                }
            }

            // Clean up remainder
            self.interpolate_into_idx_seq(
                grid_chunks.remainder(),
                out_chunks.into_remainder(),
                block_mz_array,
                block_intensity_array,
                block_n,
                &mut block_searcher,
            );
        }
        if self.array_pairs.len() > 1 {
            self.normalize_intensity_by_scan_count(out);
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
            self.interpolate_into_iter(&mut sub, start_mz, end_mz);
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
            self.interpolate_into_iter(&mut sub, start_mz, end_mz);

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
                self.interpolate_into_iter(intensity_chunk, *start_mz, end_mz);
            });
        result
    }

    pub fn interpolate_between(&'a self, mz_start: f64, mz_end: f64) -> (Vec<f32>, (usize, usize)) {
        let (n_points, (start, end)) = self.points_between_with_indices(mz_start, mz_end);
        let mut result = self.create_intensity_array_of_size(n_points);
        self.interpolate_into_iter(&mut result, mz_start, mz_end);
        (result, (start, end))
    }

    /// Allocate a new intensity array and interpolate the averaged representation of the collected spectra
    /// and return it.
    ///
    /// ```math
    /// y_z = \frac{y_{j} \times (x_{j} - x_{i}) + y_{i} \times (x_z - x_j)}{x_j - x_i}
    /// ```
    pub fn interpolate(&'a self) -> Vec<f32> {
        let mut result = self.create_intensity_array();
        self.interpolate_into_idx(&mut result, self.mz_start, self.mz_end);
        result
    }

    #[allow(unused)]
    pub fn interpolate_iter(&'a self) -> Vec<f32> {
        let mut result = self.create_intensity_array();
        self.interpolate_into_iter(&mut result, self.mz_start, self.mz_end);
        result
    }
}

impl MZGrid for SignalAverager<'_> {
    fn mz_grid(&self) -> &[f64] {
        &self.mz_grid
    }
}
impl MZInterpolator for SignalAverager<'_> {}

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

#[inline(never)]
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

impl PartialEq for ArrayPairWithSegments<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.time == other.time && self.intensity_array == other.intensity_array
    }
}

impl PartialOrd for ArrayPairWithSegments<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for ArrayPairWithSegments<'_> {}

impl Ord for ArrayPairWithSegments<'_> {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.time.total_cmp(&other.time)
    }
}

impl CoordinateLike<Time> for ArrayPairWithSegments<'_> {
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

    pub fn from_iter<I: Iterator<Item = (f64, ArrayPair<'lifespan>)>>(
        mz_start: f64,
        mz_end: f64,
        dx: f64,
        iter: I,
    ) -> Self {
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
                segments,
                intensity_array: Vec::new(),
                time,
            };
        }
        let mut segment = Segment::default();
        let mut opened = false;

        let n = self.mz_grid.len();

        let mut intensity_axis_ = self.create_intensity_array();
        let intensity_axis = &mut intensity_axis_[0..n];

        let mut block_searcher = MonotonicBlockSearcher::new(&block);
        for (i, x) in self.mz_grid.iter().copied().enumerate() {
            let j = block_searcher.find(x);
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
            } else if opened {
                segment.end = i;
                opened = false;
                segments.push(segment);
                segment = Segment::default();
            }
        }
        if opened {
            segment.end = n;
            segments.push(segment);
        }
        ArrayPairWithSegments {
            array_pair: block,
            segments,
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

impl MZGrid for SegmentGridSignalAverager<'_> {
    fn mz_grid(&self) -> &[f64] {
        &self.mz_grid
    }
}
impl MZInterpolator for SegmentGridSignalAverager<'_> {}

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
    #[allow(unused)]
    use crate::text;
    use crate::FittedPeak;

    #[test]
    fn test_rebin_one() {
        let mut averager = SignalAverager::new(X[0], X[X.len() - 1], 0.001);
        averager.push(ArrayPair::wrap(&X, &Y));
        let yhat = averager.interpolate();
        // text::arrays_to_file(ArrayPair::wrap(&averager.mz_grid, &yhat), "interpolate_avx.txt").unwrap();
        let picker = PeakPicker::default();
        let mut acc = Vec::new();
        picker
            .discover_peaks(&averager.mz_grid, &yhat, &mut acc)
            .expect("Signal can be picked");
        let mzs = [180.0633881, 181.06387399204235, 182.06404644991485];
        for (i, (peak, mz)) in acc.iter().zip(mzs.iter()).enumerate() {
            let diff = peak.mz - mz;
            assert!((peak.mz - mz).abs() < 1e-4, "Diff {} on peak {i}", diff);
            assert!(peak.intensity > 0.0);
        }
    }

    #[test]
    fn test_averaging() -> io::Result<()> {
        let scans = text::arrays_over_time_from_file("./test/data/profiles.txt")?;
        let scans: Vec<_> = scans
            .into_iter()
            .skip(3)
            .take(3)
            .map(|(_, arrays)| arrays)
            .collect();

        let low_mz = scans
            .iter()
            .map(|s| s.min_mz)
            .min_by(|a, b| a.total_cmp(b))
            .unwrap();
        let high_mz = scans
            .iter()
            .map(|s| s.max_mz)
            .max_by(|a, b| a.total_cmp(b))
            .unwrap();

        let mut averager = SignalAverager::new(low_mz, high_mz, 0.001);
        averager.extend(scans.clone());

        let _yhat = averager.interpolate();
        Ok(())
    }

    #[test]
    fn test_rebin_chunked() {
        let mut averager = SignalAverager::new(X[0], X[X.len() - 1], 0.00001);
        averager.push(ArrayPair::wrap(&X, &Y));
        let yhat = averager.interpolate_chunks(3);
        // text::arrays_to_file(ArrayPair::wrap(&averager.mz_grid, &yhat), "chunked_iter.txt").unwrap();
        let picker = PeakPicker::default();
        let mut acc = Vec::new();
        picker
            .discover_peaks(&averager.mz_grid, &yhat, &mut acc)
            .expect("Signal can be picked");
        let mzs = [180.0633881, 181.06387399204235, 182.06404644991485];
        for (i, (peak, mz)) in acc.iter().zip(mzs.iter()).enumerate() {
            let diff = peak.mz - mz;
            assert!((peak.mz - mz).abs() < 1e-4, "Diff {} on peak {i}", diff);
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
            assert!((peak.mz - mz).abs() < 1e-4, "Diff {} on peak {i}", diff);
            assert!(peak.intensity > 0.0);
        }
    }

    #[test]
    #[cfg(feature = "parallelism")]
    fn test_rebin_parallel() {
        let mut averager = SignalAverager::new(X[0], X[X.len() - 1], 0.001);
        averager.push(ArrayPair::wrap(&X, &Y));
        let yhat = averager.interpolate_chunks_parallel(6);
        let picker = PeakPicker::new(0.0, 0.0, 1.0, Default::default());
        let mut acc = Vec::new();
        picker
            .discover_peaks(&averager.mz_grid, &yhat, &mut acc)
            .expect("Signal can be picked");
        let mzs = [180.0633881, 181.06387399204235, 182.06404644991485];
        for (i, (peak, mz)) in acc.iter().zip(mzs.iter()).enumerate() {
            let diff = peak.mz - mz;
            assert!((peak.mz - mz).abs() < 1e-4, "Diff {} on peak {i}", diff);
            assert!(peak.intensity > 0.0);
        }
    }

    #[test]
    fn test_rebin() {
        let pair = rebin(&X, &Y, 0.001);
        let (acc, _, n) = pair.mz_array().iter().copied().fold((0.0, pair.min_mz, 0), |(acc, last, n), mz| {
            (acc + (mz - last), mz, n + 1)
        });
        let avg = acc / (n as f64);
        assert!((avg - 0.0009998319327731112).abs() < 1e-6);
    }

    #[test_log::test]
    fn test_segment_grid() -> io::Result<()> {
        use crate::text::arrays_over_time_from_file;
        let time_arrays = arrays_over_time_from_file("./test/data/peaks_over_time.txt")?;

        let reprofiler = crate::reprofile::PeakSetReprofiler::new(200.0, 2000.0, 0.001);

        let prepare_block = |t: f64, row: ArrayPair| {
            // log::info!("{i}: {t} with {} peaks", row.len());
            let peaks: MZPeakSetType<FittedPeak> = row
                .mz_array
                .iter()
                .zip(row.intensity_array.iter())
                .map(|(mz, i)| FittedPeak::new(*mz, *i, 0, *i, 0.005))
                .collect();

            // log::info!("Reprofiling");
            let peak_models = reprofiler
                .build_peak_shape_models(peaks.as_slice(), crate::reprofile::PeakShape::Gaussian);
            let block = reprofiler.reprofile_from_models(&peak_models);
            (t, block)
        };

        let mut t_blocks: Vec<(f64, ArrayPair<'_>)> = time_arrays
            .into_iter()
            .take(5)
            .map(|(t, row)| prepare_block(t, row))
            .collect();

        t_blocks.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut averager = SegmentGridSignalAverager::from_iter(200.0, 2000.0, 0.001, t_blocks.into_iter());
        averager.array_pairs.sort();

        // log::info!("Start averaging");
        let views: Vec<_> = averager.iter(1).collect();
        assert_eq!(views.len(), 5);

        views.iter().for_each(|(_, block)| {
            assert!(block.intensity_array.iter().all(|i| *i >= 0.0));
        });
        Ok(())
    }
}
