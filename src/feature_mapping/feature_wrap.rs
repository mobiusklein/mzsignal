//! An implementation detail, [`mzpeaks::feature_map`] doesn't always have an
//! index-yielding search method, and [`FeatureLike`] types
//! do not carry their sort index around with them, so this defines a wrapper
//! type that provides the same API, but also carries around an index. Should
//! not really be needed outside this module.
use std::{iter::Peekable, marker::PhantomData};

use mzpeaks::{feature_map::FeatureMap, prelude::*};

struct FeatureSort<X, Y, F: FeatureLike<X, Y>> {
    feature: F,
    coordinate: f64,
    start_time: Option<f64>,
    _d: PhantomData<(X, Y)>,
}

impl<X, Y, F: FeatureLike<X, Y>> FeatureSort<X, Y, F> {
    fn new(feature: F) -> Self {
        let coordinate = feature.coordinate();
        let start_time = feature.start_time();
        Self { feature, coordinate, start_time, _d: PhantomData }
    }
}

impl<X, Y, F: FeatureLike<X, Y>> Eq for FeatureSort<X, Y, F> {}

impl<X, Y, F: FeatureLike<X, Y>> Ord for FeatureSort<X, Y, F> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.coordinate.total_cmp(&other.coordinate) {
            core::cmp::Ordering::Equal => {}
            ord => return ord,
        }
        match (self.start_time, other.start_time) {
            (Some(a), Some(b)) => {
                match a.total_cmp(&b) {
                    std::cmp::Ordering::Equal => self.feature.partial_cmp(&other.feature).unwrap_or(std::cmp::Ordering::Equal),
                    x => x,
                }

            },
            (None, Some(_)) => std::cmp::Ordering::Less,
            (Some(_), None) => std::cmp::Ordering::Greater,
            (None, None) => {
                self.feature.partial_cmp(&other.feature).unwrap_or(std::cmp::Ordering::Equal)
            }
        }
    }
}

impl<X, Y, F: FeatureLike<X, Y>> PartialOrd for FeatureSort<X, Y, F> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<X, Y, F: FeatureLike<X, Y>> PartialEq for FeatureSort<X, Y, F> {
    fn eq(&self, other: &Self) -> bool {
        self.coordinate == other.coordinate && self.start_time == other.start_time && self.feature == other.feature
    }
}


struct MergeSortIter<X, Y, F: FeatureLike<X, Y>, I: Iterator<Item = FeatureSort<X, Y, F>>, J: Iterator<Item = FeatureSort<X, Y, F>>,> {
    a: Peekable<I>,
    b: Peekable<J>,
    _d: PhantomData<(X, Y, F)>
}

impl<X, Y, F: FeatureLike<X, Y>, I: Iterator<Item = FeatureSort<X, Y, F>>, J: Iterator<Item = FeatureSort<X, Y, F>>> Iterator for MergeSortIter<X, Y, F, I, J> {
    type Item = FeatureSort<X, Y, F>;

    fn next(&mut self) -> Option<Self::Item> {
        match (self.a.peek(), self.b.peek()) {
            (None, None) => None,
            (None, Some(_)) => {
                self.b.next()
            },
            (Some(_), None) => self.a.next(),
            (Some(a), Some(b)) => {
                match a.cmp(b) {
                    std::cmp::Ordering::Less => self.a.next(),
                    std::cmp::Ordering::Equal => self.a.next(),
                    std::cmp::Ordering::Greater => self.b.next(),
                }
            },
        }
    }
}

impl<X, Y, F: FeatureLike<X, Y>, I: Iterator<Item = FeatureSort<X, Y, F>>, J: Iterator<Item = FeatureSort<X, Y, F>>> MergeSortIter<X, Y, F, I, J> {
    fn new(a: Peekable<I>, b: Peekable<J>) -> Self {
        Self { a, b, _d: PhantomData }
    }
}


pub(crate) fn merge_feature_maps<X, Y, F: FeatureLike<X, Y>>(map_a: FeatureMap<X, Y, F>, map_b: FeatureMap<X, Y, F>, min_length: usize) -> FeatureMap<X, Y, F> {
    let total = map_a.len() + map_b.len();

    let it_a= map_a.into_iter().filter(|f| f.len() >= min_length).map(FeatureSort::new).peekable();
    let it_b = map_b.into_iter().filter(|f| f.len() >= min_length).map(FeatureSort::new).peekable();
    let it_merge = MergeSortIter::new(it_a, it_b);

    let mut container = Vec::with_capacity(total);
    container.extend(it_merge.map(|f| f.feature));

    FeatureMap::wrap(container)
}
