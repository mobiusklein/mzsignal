//! Helper functions and data structures for manipulating array data

use std::borrow::Cow;
use std::iter::Sum;
use std::ops::{Add, Index, Range};

use std::convert;

use num_traits::{AsPrimitive, Float, ToPrimitive, Zero};


/// Create an evenly spaced grid from `start` to `end`, with `step` between points
pub fn gridspace<T: Float + ToPrimitive>(start: T, end: T, step: T) -> Vec<T> {
    let distance = end - start;
    let steps = (distance / step).to_usize().unwrap();
    let mut result = Vec::with_capacity(steps);
    for i in 0..steps {
        result.push(start + T::from(i).unwrap() * step);
    }
    result
}


/// Given an unsorted slice, find its minimum and maximum values
pub fn minmax<T: Float>(values: &[T]) -> (T, T) {
    let mut max = -T::infinity();
    let mut min = T::infinity();

    for v in values.iter() {
        if *v > max {
            max = *v;
        }
        if *v < min {
            min = *v
        }
    }
    (min, max)
}


/// Trapezoid integration
pub fn trapz<
    A: Float + Clone + AsPrimitive<B> + 'static,
    B: Float + Clone + AsPrimitive<A> + 'static + Sum,
>(
    x: &[A],
    y: &[B],
) -> B {
    let half = B::from(0.5).unwrap();
    let n = x.len();
    (0..n - 2)
        .map(|i| {
            let delta = x[i + 1] - x[i];
            delta.as_() * half * (y[i + 1] + y[i])
        })
        .sum()
}

pub trait MZGrid {
    fn mz_grid(&self) -> &[f64];

    fn create_intensity_array(&self) -> Vec<f32> {
        self.create_intensity_array_of_size(self.mz_grid().len())
    }

    fn create_intensity_array_of_size(&self, size: usize) -> Vec<f32> {
        vec![0.0; size]
    }

    fn find_offset(&self, mz: f64) -> usize {
        match self
            .mz_grid()
            .binary_search_by(|x| x.partial_cmp(&mz).unwrap())
        {
            Ok(i) => i,
            Err(i) => i,
        }
    }

    fn points_between(&self, start_mz: f64, end_mz: f64) -> usize {
        let offset = self.find_offset(start_mz);
        let stop_index = self.find_offset(end_mz);
        stop_index - offset
    }

    fn copy_mz_array(&self) -> Vec<f64> {
        let grid = self.mz_grid();
        grid.into()
    }
}

#[derive(Debug, Default, Clone)]
/// Represent an m/z array and an intensity array with independent
/// "borrowing" statuses using [`std::borrow::Cow`]. Adds a few helper
/// methods.
pub struct ArrayPair<'lifespan> {
    pub mz_array: Cow<'lifespan, [f64]>,
    pub intensity_array: Cow<'lifespan, [f32]>,
    pub min_mz: f64,
    pub max_mz: f64,
}

impl<'lifespan> ArrayPair<'lifespan> {
    pub fn wrap(
        mz_array: &'lifespan [f64],
        intensity_array: &'lifespan [f32],
    ) -> ArrayPair<'lifespan> {
        Self::new(Cow::Borrowed(mz_array), Cow::Borrowed(intensity_array))
    }

    pub fn new(
        mz_array: Cow<'lifespan, [f64]>,
        intensity_array: Cow<'lifespan, [f32]>,
    ) -> ArrayPair<'lifespan> {
        let min_mz = match mz_array.first() {
            Some(min_mz) => *min_mz,
            None => 0.0,
        };
        let max_mz = match mz_array.last() {
            Some(max_mz) => *max_mz,
            None => min_mz,
        };
        ArrayPair {
            mz_array,
            intensity_array,
            min_mz,
            max_mz,
        }
    }

    /// Find the index nearest to `mz`
    pub fn find(&self, mz: f64) -> usize {
        match self
            .mz_array
            .binary_search_by(|x| x.partial_cmp(&mz).unwrap())
        {
            Ok(i) => i.min(self.mz_array.len().saturating_sub(1)),
            Err(i) => i.min(self.mz_array.len().saturating_sub(1)),
        }
    }

    /// Select the slice of the m/z and intensity arrays between `low` m/z and `high` m/z
    /// returning a non-owning [`ArrayPair`]
    pub fn find_between(&self, low: f64, high: f64) -> ArrayPair<'_> {
        let i_low = self.find(low);
        let i_high = self.find(high);
        let mz_array = &self.mz_array[i_low..i_high];
        let intensity_array = &self.intensity_array[i_low..i_high];
        (mz_array, intensity_array).into()
    }

    pub fn len(&self) -> usize {
        self.mz_array.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, i: usize) -> Option<(f64, f32)> {
        if i >= self.len() {
            None
        } else {
            Some((self.mz_array[i], self.intensity_array[i]))
        }
    }

    pub fn borrow(&'_ self) -> ArrayPair<'_> {
        ArrayPair::new(
            Cow::Borrowed(&self.mz_array),
            Cow::Borrowed(&self.intensity_array),
        )
    }

    pub fn to_owned(self) -> ArrayPair<'static> {
        let mz_array = match self.mz_array {
            Cow::Borrowed(b) => Cow::Owned(b.to_vec()),
            Cow::Owned(b) => Cow::Owned(b),
        };
        let intensity_array = match self.intensity_array {
            Cow::Borrowed(b) => Cow::Owned(b.to_vec()),
            Cow::Owned(b) => Cow::Owned(b),
        };
        ArrayPair::new(mz_array, intensity_array)
    }
}

impl<'lifespan> From<(Cow<'lifespan, [f64]>, Cow<'lifespan, [f32]>)>
    for ArrayPair<'lifespan>
{
    fn from(pair: (Cow<'lifespan, [f64]>, Cow<'lifespan, [f32]>)) -> ArrayPair<'lifespan> {
        ArrayPair::new(pair.0, pair.1)
    }
}

impl<'lifespan> From<(&'lifespan [f64], &'lifespan [f32])> for ArrayPair<'lifespan> {
    fn from(pair: (&'lifespan [f64], &'lifespan [f32])) -> ArrayPair<'lifespan> {
        ArrayPair::wrap(pair.0, pair.1)
    }
}

impl<'lifespan> From<(Vec<f64>, Vec<f32>)> for ArrayPair<'lifespan> {
    fn from(pair: (Vec<f64>, Vec<f32>)) -> ArrayPair<'lifespan> {
        let mz_array = Cow::Owned(pair.0);
        let intensity_array = Cow::Owned(pair.1);
        ArrayPair::new(mz_array, intensity_array)
    }
}
