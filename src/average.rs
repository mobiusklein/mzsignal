use std::ops::{Add, Index};
use std::sync::Mutex;

#[cfg(feature = "parallelism")]
use rayon::prelude::*;

use num_traits::{Float, ToPrimitive};
use crate::search;

pub fn gridspace<T: Float + ToPrimitive>(start: T, end: T, step: T) -> Vec<T> {
    let distance = end - start;
    let steps = (distance / step).to_usize().unwrap();
    let mut result = Vec::with_capacity(steps);
    for i in 0..steps {
        result.push(start + T::from(i).unwrap() * step);
    }
    result
}

#[derive(Debug, Default)]
pub struct ArrayPair<'lifespan> {
    pub mz_array: &'lifespan [f64],
    pub intensity_array: &'lifespan [f32],
    pub min_mz: f64,
    pub max_mz: f64
}

impl<'lifespan> ArrayPair<'lifespan> {
    pub fn new(mz_array: &'lifespan [f64], intensity_array: &'lifespan [f32]) -> ArrayPair<'lifespan> {
        let min_mz = match mz_array.first() {
            Some(min_mz) => *min_mz,
            None => 0.0
        };
        let max_mz = match mz_array.last() {
            Some(max_mz) => *max_mz,
            None =>  min_mz
        };
        ArrayPair {
            mz_array, intensity_array, min_mz, max_mz
        }
    }

    pub fn find(&self, mz: f64) -> usize {
        match self.mz_array.binary_search_by(|x| x.partial_cmp(&mz).unwrap()) {
            Ok(i) => i,
            Err(i) => i
        }
    }

    pub fn len(&self) -> usize {
        self.mz_array.len()
    }

    pub fn get(&self, i: usize) -> Option<(f64, f32)> {
        if i >= self.len() {
            return None
        } else {
            return Some((self.mz_array[i], self.intensity_array[i]))
        }
    }
}


#[derive(Debug, Default)]
pub struct SignalAverager<'lifespan> {
    pub mz_grid: Vec<f64>,
    pub mz_start: f64,
    pub mz_end: f64,
    pub array_pairs: Vec<ArrayPair<'lifespan>>,
}


impl<'lifespan, 'transient: 'lifespan> SignalAverager<'lifespan> {
    pub fn new(mz_start: f64, mz_end: f64, dx: f64) -> SignalAverager<'lifespan> {
        SignalAverager{
            mz_grid: gridspace(mz_start, mz_end, dx),
            mz_start, mz_end,
            array_pairs: Vec::new()
        }
    }

    pub fn push(&mut self, pair: ArrayPair<'transient>) {
        self.array_pairs.push(pair)
    }

    pub fn pop(&mut self) -> Option<ArrayPair<'lifespan>> {
        self.array_pairs.pop()
    }

    fn create_intensity_array(&self) -> Vec<f32> {
        self.create_intensity_array_of_size(self.mz_grid.len())
    }

    fn create_intensity_array_of_size(&self, size: usize) -> Vec<f32> {
        vec![0.0; size]
    }

    fn find_offset(&self, mz: f64) -> usize {
        match self.mz_grid.binary_search_by(|x| x.partial_cmp(&mz).unwrap()) {
            Ok(i) => i,
            Err(i) => i
        }
    }

    #[inline]
    pub fn interpolate_point(&self, mz_j: f64, mz_x: f64, mz_j1: f64, inten_j: f64, inten_j1: f64) -> f64 {
        ((inten_j * (mz_j1 - mz_x)) + (inten_j1 * (mz_x - mz_j))) / (mz_j1 - mz_j)
    }

    pub fn points_between(&self, start_mz: f64, end_mz: f64) -> usize {
        let offset = self.find_offset(start_mz);
        let stop_index = self.find_offset(end_mz);
        stop_index - offset
    }

    pub fn interpolate_into(&self, out: &mut [f32], start_mz: f64, end_mz: f64) -> usize {
        let offset = self.find_offset(start_mz);
        let stop_index = self.find_offset(end_mz);
        assert!(stop_index - offset == out.len());

        for block in self.array_pairs.iter() {
            for i in offset..stop_index {
                let x = self.mz_grid[i];
                let j = block.find(x);
                let mz_j = block.mz_array[j];

                let (mz_j, inten_j, mz_j1, inten_j1) = if (mz_j <= x) && ((j + 1) < block.len()) {
                    (mz_j, block.intensity_array[j], block.mz_array[j + 1], block.intensity_array[j + 1])
                } else if mz_j > x && j > 0 {
                    (block.mz_array[j - 1], block.intensity_array[j - 1],
                     block.mz_array[j], block.intensity_array[j])
                } else {
                    continue
                };
                let interp = self.interpolate_point(mz_j, x, mz_j1, inten_j as f64, inten_j1 as f64);
                out[i - offset] = interp as f32;
            }
        }
        stop_index - offset
    }

    pub fn interpolate_chunks(&'lifespan self, n_chunks: usize) -> Vec<f32> {
        let mut result = self.create_intensity_array();
        if self.array_pairs.is_empty() {
            return result;
        }
        let n_points = self.points_between(self.mz_start, self.mz_end);

        let points_per_chunk = n_points / n_chunks;
        for i in 0..n_chunks {
            let offset =  i * points_per_chunk;
            let (size, start_mz, end_mz) = if i == n_chunks - 1{
                (n_points - offset, self.mz_grid[offset], self.mz_end)
            } else {
                (points_per_chunk, self.mz_grid[offset], self.mz_grid[offset + points_per_chunk])
            };
            let mut sub = self.create_intensity_array_of_size(size);
            self.interpolate_into(&mut sub, start_mz, end_mz);
            (result[offset..offset + size]).copy_from_slice(&sub);
        }
        result
    }

    #[cfg(feature = "parallelism")]
    pub fn interpolate_chunks_parallel(&'lifespan self, n_chunks: usize) -> Vec<f32> {
        let result = self.create_intensity_array();
        if self.array_pairs.is_empty() {
            return result;
        }
        let n_points = self.points_between(self.mz_start, self.mz_end);
        let locked_result = Mutex::new(result);
        let points_per_chunk = n_points / n_chunks;
        (0..n_chunks).into_par_iter().for_each(|i| {
            let offset =  i * points_per_chunk;
            let (size, start_mz, end_mz) = if i == n_chunks - 1{
                (n_points - offset, self.mz_grid[offset], self.mz_end)
            } else {
                (points_per_chunk, self.mz_grid[offset], self.mz_grid[offset + points_per_chunk])
            };
            let mut sub = self.create_intensity_array_of_size(size);
            self.interpolate_into(&mut sub, start_mz, end_mz);
            let mut out = locked_result.lock().unwrap();
            (out[offset..offset + size]).copy_from_slice(&sub);
        });
        locked_result.into_inner().unwrap()
    }

    pub fn interpolate(&'lifespan self) -> Vec<f32> {
        let mut result = self.create_intensity_array();
        self.interpolate_into(&mut result, self.mz_start, self.mz_end);
        result
    }

    pub fn copy_mz_array(&self) -> Vec<f64> {
        self.mz_grid.clone()
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use crate::test_data::{X, Y};
    use crate::peak_picker::PeakPicker;


    #[test]
    fn test_rebin_one() {
        let mut averager = SignalAverager::new(X[0], X[X.len() - 1], 0.00001);
        averager.push(ArrayPair::new(&X, &Y));
        let yhat = averager.interpolate();
        let picker = PeakPicker::default();
        let mut acc = Vec::new();
        picker.discover_peaks(&averager.mz_grid, &yhat, &mut acc).expect("Signal can be picked");
        let mzs = [180.0633881, 181.06338858024316, 182.06338874740308];
        for (peak, mz) in acc.iter().zip(mzs.iter()) {
            assert!((peak.mz - mz).abs() < 1e-6);
        }
    }

    #[test]
    fn test_rebin_chunked() {
        let mut averager = SignalAverager::new(X[0], X[X.len() - 1], 0.00001);
        averager.push(ArrayPair::new(&X, &Y));
        let yhat = averager.interpolate_chunks(3);
        let picker = PeakPicker::default();
        let mut acc = Vec::new();
        picker.discover_peaks(&averager.mz_grid, &yhat, &mut acc).expect("Signal can be picked");
        let mzs = [180.0633881, 181.06338858024316, 182.06338874740308];
        for (peak, mz) in acc.iter().zip(mzs.iter()) {
            assert!((peak.mz - mz).abs() < 1e-6);
        }
    }

    #[test]
    #[cfg(feature = "parallelism")]
    fn test_rebin_parallel() {
        let mut averager = SignalAverager::new(X[0], X[X.len() - 1], 0.00001);
        averager.push(ArrayPair::new(&X, &Y));
        let yhat = averager.interpolate_chunks_parallel(6);
        let picker = PeakPicker::default();
        let mut acc = Vec::new();
        picker.discover_peaks(&averager.mz_grid, &yhat, &mut acc).expect("Signal can be picked");
        let mzs = [180.0633881, 181.06338858024316, 182.06338874740308];
        for (peak, mz) in acc.iter().zip(mzs.iter()) {
            assert!((peak.mz - mz).abs() < 1e-6);
        }
    }
}