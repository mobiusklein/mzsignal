//! An implementation detail, [`mzpeaks::feature_map`] doesn't always have an
//! index-yielding search method, and [`FeatureLike`] types
//! do not carry their sort index around with them, so this defines a wrapper
//! type that provides the same API, but also carries around an index. Should
//! not really be needed outside this module.
use std::{marker::PhantomData, ops::Deref};

use mzpeaks::prelude::*;

/// Wrap a [`FeatureLike`] type `F` with a known index for
/// ease of reference
#[derive(Debug)]
pub struct IndexedFeature<'a, D, T, F: FeatureLike<D, T>> {
    /// Some [`FeatureLike`] type we want to build a graph over
    pub feature: &'a F,
    /// The index of `feature` to use as a key
    pub index: usize,
    _d: PhantomData<D>,
    _t: PhantomData<T>,
}

impl<'a, D, T, F: FeatureLike<D, T>> IndexedFeature<'a, D, T, F> {
    pub fn new(feature: &'a F, index: usize) -> Self {
        Self {
            feature,
            index,
            _d: PhantomData,
            _t: PhantomData,
        }
    }
}

impl<'a, D: Deref, T: Deref, F: FeatureLike<D, T> + Deref> Deref for IndexedFeature<'a, D, T, F> {
    type Target = &'a F;

    fn deref(&self) -> &Self::Target {
        &self.feature
    }
}

impl<'a, D, T, F: FeatureLike<D, T>> TimeInterval<T> for IndexedFeature<'a, D, T, F> {
    fn start_time(&self) -> Option<f64> {
        self.feature.start_time()
    }

    fn end_time(&self) -> Option<f64> {
        self.feature.end_time()
    }

    fn apex_time(&self) -> Option<f64> {
        self.feature.apex_time()
    }

    fn area(&self) -> f32 {
        self.feature.area()
    }

    fn iter_time(&self) -> impl Iterator<Item = f64> {
        self.feature.iter_time()
    }
}

impl<'a, D, T, F: FeatureLike<D, T>> PartialEq for IndexedFeature<'a, D, T, F> {
    fn eq(&self, other: &Self) -> bool {
        self.feature == other.feature && self.index == other.index
    }
}

impl<'a, D, T, F: FeatureLike<D, T>> PartialOrd for IndexedFeature<'a, D, T, F> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.feature.partial_cmp(&other.feature)
    }
}

impl<'a, D, T, F: FeatureLike<D, T>> CoordinateLike<D> for IndexedFeature<'a, D, T, F> {
    fn coordinate(&self) -> f64 {
        self.feature.coordinate()
    }
}

impl<'a, D, T, F: FeatureLike<D, T>> IntensityMeasurement for IndexedFeature<'a, D, T, F> {
    fn intensity(&self) -> f32 {
        self.feature.intensity()
    }
}

impl<'a, D, T, F: FeatureLike<D, T>> FeatureLike<D, T> for IndexedFeature<'a, D, T, F> {
    fn len(&self) -> usize {
        self.feature.len()
    }

    fn iter(&self) -> impl Iterator<Item = (f64, f64, f32)> {
        self.feature.iter()
    }
}
