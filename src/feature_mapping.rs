//! Construct feature maps from peak lists over time.
//!
//! The top-level type is [`FeatureExtracter`], or [`DeconvolvedFeatureExtracter`], it's
//! charge state-aware alternative. These types are parameterized over the time dimension
//! to work with [`mzpeaks`]'s coordinate system.
//!
//! ```rust
//! # use std::io;
//! # use mzpeaks::{CentroidPeak, MZPeakSetType, feature::Feature, feature_map::FeatureMap, Time, MZ, Tolerance};
//! # use mzsignal::text::arrays_over_time_from_file;
//! use mzsignal::feature_mapping::FeatureExtracter;
//!
//! # fn main() -> io::Result<()> {
//! let time_arrays = arrays_over_time_from_file("test/data/peaks_over_time.txt")?;
//! let mut time_axis: Vec<f64> = Vec::new();
//! let mut peak_table: Vec<MZPeakSetType<CentroidPeak>> = Vec::new();
//!
//! // Build up the parallel arrays of scan time and peak list
//! for (time, peaks_of_row) in time_arrays {
//!     time_axis.push(time);
//!     let peaks: MZPeakSetType<CentroidPeak> = peaks_of_row
//!         .mz_array
//!         .into_iter()
//!         .zip(peaks_of_row.intensity_array.into_iter())
//!         .map(|(mz, i)| CentroidPeak::new(*mz, *i, 0))
//!         .collect();
//!     peak_table.push(peaks);
//! }
//! let mut peak_map_builder = FeatureExtracter::<_, Time>::from_iter(
//!     time_axis.into_iter().zip(peak_table)
//! );
//! let features: FeatureMap<MZ, Time, Feature<MZ, Time>> = peak_map_builder.extract_features(
//!     Tolerance::PPM(10.0),
//!     3,
//!     0.25
//! );
//! #   Ok(())
//! }
//! ```
//!
use std::collections::{
    hash_map::{Entry, HashMap},
    HashSet, VecDeque,
};
use std::hash::Hash;
use std::marker::PhantomData;
use std::mem::{self, swap};
use std::ops::Index;

use mzpeaks::{
    feature::{ChargedFeature, Feature},
    feature_map::FeatureMap,
    peak_set::PeakSetVec,
    prelude::*,
    CentroidPeak, IonMobility, Mass, Time, Tolerance, MZ,
};

use crate::peak_statistics::isclose;
use crate::search::nearest;

/// Build graphs of features to bridge gaps in a feature map
pub mod graph {
    use super::*;

    /// Merge [`FeatureLike`] entities which are within the same mass dimension
    /// error tolerance and within a certain time of one-another by constructing a graph, extracting
    /// connected components, and stitch them together.
    pub trait FeatureGraphBuilder<D, T, F: FeatureLike<D, T> + FeatureLikeMut<D, T> + Clone> {
        /// Build the feature graph, where `features` are nodes and edges connect features which
        /// are close as given by `mass_error_tolerance`, with gap between start/end or end/start
        /// <= `maximum_gap_size` time units (given by `T`).
        ///
        /// The default implementation only filters on the [`mzpeaks::CoordinateLike::coordinate`]
        /// w.r.t. `D`. If additional constraints are needed, provide a specific implementation.
        fn build_graph(
            &self,
            features: &FeatureMap<D, T, F>,
            mass_error_tolerance: Tolerance,
            maximum_gap_size: f64,
        ) -> Vec<FeatureNode> {
            let features: FeatureMap<D, T, _> = features
                .iter()
                .enumerate()
                .map(|(i, f)| IndexedFeature::new(f, i))
                .collect();

            let mut nodes = Vec::with_capacity(features.len());

            for f in features.iter() {
                if f.is_empty() {
                    continue;
                }
                let candidates = features.all_features_for(f.coordinate(), mass_error_tolerance);
                let mut edges = Vec::new();
                let start_time = f.start_time().unwrap();
                let end_time = f.end_time().unwrap();
                for c in candidates {
                    if f.index == c.index || c.is_empty() {
                        continue;
                    } else {
                        let c_start = c.start_time().unwrap();
                        let c_end = c.end_time().unwrap();
                        if (start_time - c_end).abs() < maximum_gap_size
                            || (end_time - c_start).abs() < maximum_gap_size
                            || f.as_range().overlaps(&c.as_range())
                        {
                            edges.push(FeatureLink::new(f.index, c.index));
                        }
                    }
                }
                let node = FeatureNode::new(f.index, edges);
                nodes.push(node);
            }

            nodes
        }

        fn find_connected_components(&self, graph: Vec<FeatureNode>) -> Vec<Vec<usize>> {
            let mut tarjan = TarjanStronglyConnectedComponents::new(graph);
            tarjan.solve();
            tarjan.connected_components
        }

        fn merge_components(
            &self,
            features: &FeatureMap<D, T, F>,
            connected_components: Vec<Vec<usize>>,
        ) -> FeatureMap<D, T, F> {
            let mut merged_nodes = Vec::new();
            for component_indices in connected_components {
                if component_indices.is_empty() {
                    continue;
                }
                let mut features_of: Vec<_> = component_indices
                    .into_iter()
                    .map(|i| features.get_item(i))
                    .collect();

                let n_features_of = features_of.len();
                features_of
                    .sort_by(|a, b| a.start_time().unwrap().total_cmp(&b.start_time().unwrap()));

                let mut iter = features_of.iter().enumerate();
                let mut acc = (*iter.next().unwrap().1).clone();
                let mut prev = acc.last().unwrap();
                for (feature_index, f) in iter.skip(1) {
                    for (node_index, (x, y, z)) in f.iter().enumerate() {
                        if let Some(last) = acc.end_time() {
                            if isclose(*y, last) {
                                let e = y - last;
                                log::warn!(
                                    "Attempted to add data point ({x}, {y}, {z}) from feature {feature_index} at \
                                    {node_index}, prev {prev:?} to feature ending at {:?} ({e}) with {n_features_of} features to merge",
                                    f.at_time(last).unwrap()
                                )
                            }
                        }
                        acc.push_raw(*x, *y, *z);
                        prev = (*x, *y, *z);
                    }
                }
                merged_nodes.push(acc);
            }

            FeatureMap::new(merged_nodes)
        }

        fn bridge_feature_gaps(
            &self,
            features: &FeatureMap<D, T, F>,
            mass_error_tolerance: Tolerance,
            maximum_gap_size: f64,
        ) -> FeatureMap<D, T, F> {
            let graph = self.build_graph(features, mass_error_tolerance, maximum_gap_size);
            let components = self.find_connected_components(graph);
            self.merge_components(features, components)
        }
    }

    /// A trivial implementation of [`FeatureGraphBuilder`] using its default implementation
    #[derive(Debug, Default, Clone)]
    pub struct FeatureMerger<D, T, F: FeatureLike<D, T> + FeatureLikeMut<D, T> + Clone> {
        _d: PhantomData<D>,
        _t: PhantomData<T>,
        _f: PhantomData<F>,
    }

    impl<D, T, F: FeatureLike<D, T> + FeatureLikeMut<D, T> + Clone> FeatureGraphBuilder<D, T, F>
        for FeatureMerger<D, T, F>
    {
    }

    /// An implementation of [`FeatureGraphBuilder`] that only permits edges between
    /// nodes which have equal charge states
    #[derive(Debug, Default, Clone)]
    pub struct ChargeAwareFeatureMerger<
        D,
        T,
        F: FeatureLike<D, T> + FeatureLikeMut<D, T> + Clone + KnownCharge,
    > {
        _d: PhantomData<D>,
        _t: PhantomData<T>,
        _f: PhantomData<F>,
    }

    /// The core functionality of [`ChargeAwareFeatureMerger`] is in its non-default
    /// [`FeatureGraphBuilder`] implementation.
    impl<D, T, F: FeatureLike<D, T> + FeatureLikeMut<D, T> + Clone + KnownCharge>
        FeatureGraphBuilder<D, T, F> for ChargeAwareFeatureMerger<D, T, F>
    {
        /// This is identical to the default implementation save that
        /// it enforces the limitation on charge state matches.
        fn build_graph(
            &self,
            features: &FeatureMap<D, T, F>,
            mass_error_tolerance: Tolerance,
            maximum_gap_size: f64,
        ) -> Vec<FeatureNode> {
            let features: FeatureMap<D, T, _> = features
                .iter()
                .enumerate()
                .map(|(i, f)| IndexedFeature::new(f, i))
                .collect();

            let mut nodes = Vec::with_capacity(features.len());

            for f in features.iter() {
                if f.is_empty() {
                    continue;
                }
                let z = f.feature.charge();
                let candidates = features.all_features_for(f.coordinate(), mass_error_tolerance);
                let mut edges = Vec::new();
                let start_time = f.start_time().unwrap();
                let end_time = f.end_time().unwrap();
                for c in candidates {
                    if f.index == c.index || c.is_empty() || c.feature.charge() != z {
                        continue;
                    } else {
                        let c_start = c.start_time().unwrap();
                        let c_end = c.end_time().unwrap();
                        if (start_time - c_end).abs() < maximum_gap_size
                            || (end_time - c_start).abs() < maximum_gap_size
                            || f.as_range().overlaps(&c.as_range())
                        {
                            edges.push(FeatureLink::new(f.index, c.index));
                        }
                    }
                }
                let node = FeatureNode::new(f.index, edges);
                nodes.push(node);
            }

            nodes
        }
    }

    /// A node in a graph representing a feature stored in some other collection
    /// that holds extra state for [`TarjanStronglyConnectedComponents`].
    #[derive(Debug, Default, Clone)]
    pub struct FeatureNode {
        /// The location in the external collection where the feature this node represents
        /// resides.
        pub feature_index: usize,
        /// The indices of the related features in the external collection which this feature
        /// connects to.
        pub edges: Vec<FeatureLink>,
        /// The stack index produced by Tarjan's strongly connected component.
        index: Option<usize>,
        /// The putative root index for the current connected component while this
        /// node was on the stack.
        low_link: Option<usize>,
        /// Whether or not this node is still on the stack.
        on_stack: bool,
    }

    impl FeatureNode {
        pub fn new(feature_index: usize, edges: Vec<FeatureLink>) -> Self {
            Self {
                feature_index,
                edges,
                ..Default::default()
            }
        }
    }

    /// Describes a relationship between two features in some collection
    /// with static indices.
    #[derive(Debug, Default, Clone, Copy)]
    pub struct FeatureLink {
        pub from_index: usize,
        pub to_index: usize,
    }

    impl FeatureLink {
        pub fn new(from_index: usize, to_index: usize) -> Self {
            Self {
                from_index,
                to_index,
            }
        }
    }

    /// An implementation of Tarjan's strongly connected components algorithm.
    ///
    /// The main goal is to identify all connected components of the graph which
    /// represent segments of what could be the same feature. The manner of the
    /// graph's construction may influence how truly related those segments are.
    ///
    /// Once connected components are identified, some other algorithm may take
    /// care of merging them applying whatever criteria they consider necessary.
    ///
    /// Graph nodes are represented by `usize` values, corresponding to [`FeatureNode::feature_index`]
    /// and as such components are just [`Vec<usize>`].
    ///
    /// # See also:
    /// This is a near direct translation of the pseudocode from [Wikipedia](https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm)
    #[derive(Debug, Default)]
    pub struct TarjanStronglyConnectedComponents {
        nodes: Vec<FeatureNode>,
        current_component: Vec<usize>,
        connected_components: Vec<Vec<usize>>,
        index: usize,
    }

    impl TarjanStronglyConnectedComponents {
        pub fn new(nodes: Vec<FeatureNode>) -> Self {
            Self {
                nodes,
                ..Default::default()
            }
        }

        fn visit_node_set_index(&mut self, node_index: usize) {
            let node = self.nodes.get_mut(node_index).unwrap();
            node.index = Some(self.index);
            node.low_link = Some(self.index);
            node.on_stack = true;
            self.index += 1;
        }

        fn index_is_visited(&self, node_index: usize) -> bool {
            self.nodes.get(node_index).unwrap().index.is_some()
        }

        fn edges_of(&self, node_index: usize) -> Vec<usize> {
            self.nodes
                .get(node_index)
                .unwrap()
                .edges
                .iter()
                .map(|e| e.to_index)
                .collect()
        }

        /// Identify all connected components iteratively.
        pub fn solve(&mut self) {
            let mut stack = Vec::new();

            for i in 0..self.nodes.len() {
                if !self.index_is_visited(i) {
                    self.strong_connect(i, &mut stack)
                }
            }
        }

        fn node_mut(&'_ mut self, node_index: usize) -> &'_ mut FeatureNode {
            self.nodes.get_mut(node_index).unwrap()
        }

        fn node(&self, node_index: usize) -> &FeatureNode {
            self.nodes.get(node_index).unwrap()
        }

        /// Iterate over the connected components of the graph. This will
        /// be an empty iterator if [`TarjanStronglyConnectedComponents::solve`]
        /// has not been called yet.
        pub fn iter(&self) -> std::slice::Iter<Vec<usize>> {
            self.connected_components.iter()
        }

        fn consume_stack_until(&mut self, node_index: usize, stack: &mut Vec<usize>) {
            while let Some(w) = stack.pop() {
                if w != node_index {
                    self.node_mut(w).on_stack = false;
                    self.current_component.push(w);
                } else {
                    // stack.push(w);
                    self.node_mut(w).on_stack = false;
                    self.current_component.push(w);
                    break;
                }
            }
            self.connected_components
                .push(mem::take(&mut self.current_component));
        }

        fn strong_connect(&mut self, node_index: usize, stack: &mut Vec<usize>) {
            self.visit_node_set_index(node_index);
            stack.push(node_index);

            for w in self.edges_of(node_index) {
                if !self.index_is_visited(w) {
                    self.strong_connect(w, stack);
                    self.node_mut(node_index).low_link = Some(
                        self.node(w)
                            .low_link
                            .unwrap()
                            .min(self.node(node_index).low_link.unwrap()),
                    );
                } else if self.node(w).on_stack {
                    self.node_mut(node_index).low_link = Some(
                        self.node(w)
                            .index
                            .unwrap()
                            .min(self.node(node_index).low_link.unwrap()),
                    );
                }
            }

            if self.node(node_index).low_link == self.node(node_index).index {
                self.consume_stack_until(node_index, stack);
            }
        }
    }

    impl IntoIterator for TarjanStronglyConnectedComponents {
        type Item = Vec<usize>;

        type IntoIter = std::vec::IntoIter<Vec<usize>>;

        fn into_iter(self) -> Self::IntoIter {
            self.connected_components.into_iter()
        }
    }

    // An implementation detail, [`mzpeaks::feature_map`] doesn't always have an
    // index-yielding search method, and [`FeatureLike`] types
    // do not carry their sort index around with them, so this defines a wrapper
    // type that provides the same API, but also carries around an index. Should
    // not really be needed outside this module.
    mod index_impl {
        use super::*;

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

        impl<
                'a,
                D: std::ops::Deref,
                T: std::ops::Deref,
                F: FeatureLike<D, T> + std::ops::Deref,
            > std::ops::Deref for IndexedFeature<'a, D, T, F>
        {
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

            fn iter(&self) -> impl Iterator<Item = (&f64, &f64, &f32)> {
                self.feature.iter()
            }
        }
    }

    use index_impl::IndexedFeature;
}

/// Helper types for representing coordinates over a [`MapState`]
pub mod map {
    use super::*;
    /// Represents a coordinate in a [`PeakMapState`], referencing a specific peak
    /// at a specific time.
    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct MapIndex {
        /// The time index effectively refers to the row in [`PeakMapState::peak_table`]
        pub time_index: usize,
        /// The peak index effectively refers to the column in the `peak_index`-th row in [`PeakMapState::peak_table`]
        pub peak_index: usize,
    }

    impl MapIndex {
        pub fn new(time_index: usize, peak_index: usize) -> Self {
            Self {
                time_index,
                peak_index,
            }
        }
    }

    /// Connects two coordinates in a [`PeakMapState`], carrying some quality
    /// information about the linkage.
    #[derive(Debug, Default, Clone, Copy, PartialEq, PartialOrd)]
    pub struct MapLink {
        pub from_index: MapIndex,
        pub to_index: MapIndex,
        pub mass_error: f64,
        pub intensity_weight: f32,
    }

    impl Hash for MapLink {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            self.from_index.hash(state);
        }
    }

    impl MapLink {
        pub fn new(
            from_index: MapIndex,
            to_index: MapIndex,
            mass_error: f64,
            intensity_weight: f32,
        ) -> Self {
            Self {
                from_index,
                to_index,
                mass_error,
                intensity_weight,
            }
        }

        /// Compute a summary metric of how good of a link this is.
        ///
        /// This is used to solve the dynamic programming problem when
        /// extracting paths.
        pub fn score(&self) -> f32 {
            self.intensity_weight * (1.0 - self.mass_error.powi(4) as f32)
        }
    }
}

use graph::*;
use map::*;

/// A sparse, immutable matrix of peaks over time, the default [`MapState`] implementation.
#[derive(Debug, Clone)]
pub struct PeakMapState<C: IndexedCoordinate<D> + IntensityMeasurement, D> {
    /// A mapping from row index to time coordinate given by some external dimension `T`
    pub time_axis: Vec<f64>,
    /// The peaks of type `C` on coordinate system `D` at each time point
    pub peak_table: Vec<PeakSetVec<C, D>>,
}

impl<C: IndexedCoordinate<D> + IntensityMeasurement, D> Default for PeakMapState<C, D> {
    fn default() -> Self {
        Self {
            time_axis: Vec::new(),
            peak_table: Vec::new(),
        }
    }
}

/// A [`PeakMapState`] can be built from an iterator, but it expects the iterator to be sorted over the time dimension, and
/// will panic if it encounters a non-monotonic sequence.
impl<C: IndexedCoordinate<D> + IntensityMeasurement, D> FromIterator<(f64, PeakSetVec<C, D>)>
    for PeakMapState<C, D>
{
    fn from_iter<T: IntoIterator<Item = (f64, PeakSetVec<C, D>)>>(iter: T) -> Self {
        let mut this = Self::default();
        let mut last_time = f64::NEG_INFINITY;
        for (time, peaks) in iter {
            assert!(time > last_time);
            last_time = time;
            this.time_axis.push(time);
            this.peak_table.push(peaks);
        }
        this
    }
}

impl<D, C: IndexedCoordinate<D> + IntensityMeasurement> PeakMapState<C, D> {
    pub fn new(time_axis: Vec<f64>, peak_table: Vec<PeakSetVec<C, D>>) -> Self {
        assert_eq!(time_axis.len(), peak_table.len());
        Self {
            time_axis,
            peak_table,
        }
    }
}

/// Defines operations on a peak-over-time map
pub trait MapState<C: IndexedCoordinate<D> + IntensityMeasurement + 'static, D: 'static, T> {
    /// The type of feature this map is eventually made of. Must implement [`FeatureLike`]
    /// and [`FeatureLikeMut`].
    type FeatureType: FeatureLike<D, T> + Default + FeatureLikeMut<D, T> + Clone;

    /// The implementation of [`FeatureGraphBuilder`] to use with this map's [`MapState::FeatureType`]
    type FeatureMergerType: FeatureGraphBuilder<D, T, Self::FeatureType> + Default;

    /// Get a reference to the time axis as a slice of floats
    fn time_axis(&self) -> &[f64];

    /// Get a reference to the sparse peak table
    fn peak_table(&self) -> &[PeakSetVec<C, D>];

    /// Fill the peak table from an iterator over (time, peak list)s
    fn populate_from_iterator(&mut self, it: impl Iterator<Item = (f64, PeakSetVec<C, D>)>);

    /// The length of the time dimension of the map
    fn len(&self) -> usize {
        self.time_axis().len()
    }

    /// Whether there are any peak slices in the map
    fn is_empty(&self) -> bool {
        self.time_axis().is_empty()
    }

    /// Iterate over the peaks in row `time_index`
    ///
    /// This function panics if `time_index` is out of bounds
    fn iter_at_index(&self, time_index: usize) -> impl Iterator<Item = &C> {
        self.peak_table()[time_index].iter()
    }

    /// The change in absolute time between time indices `i` and `j`
    fn time_delta(&self, i: usize, j: usize) -> Option<f64> {
        let ti = self.time_axis().get(i);
        let tj = self.time_axis().get(j);

        if let (Some(ti), Some(tj)) = (ti.copied(), tj.copied()) {
            Some(ti - tj)
        } else {
            None
        }
    }

    /// Find the time index closest to `time`
    fn nearest_time_point(&self, time: f64) -> usize {
        nearest(&self.time_axis(), time, 0)
    }

    /// Iterate over [`MapIndex`] coordinates for the `query` peak in the row at `time_index`
    /// within `error_tolerance` mass error.
    ///
    /// This function panics if `time_index` is out of bounds
    fn query_with_index<'a>(
        &'a self,
        query: &'a C,
        time_index: usize,
        error_tolerance: Tolerance,
    ) -> impl Iterator<Item = MapIndex> + 'a {
        let hits = self.peak_table()[time_index].all_peaks_for(query.coordinate(), error_tolerance);
        hits.iter()
            .map(move |p| MapIndex::new(time_index, p.get_index() as usize))
    }

    /// Iterate over [`MapIndex`] coordinates for the `query` peak in the row closest
    /// to `time` within `error_tolerance` mass error.
    ///
    /// Relies on [`MapState::nearest_time_point`] and [`MapState::query_with_index`]
    fn query<'a>(
        &'a self,
        query: &'a C,
        time: f64,
        error_tolerance: Tolerance,
    ) -> impl Iterator<Item = MapIndex> + 'a {
        let time_index = self.nearest_time_point(time);
        self.query_with_index(query, time_index, error_tolerance)
    }

    /// Retrieve the peak at [`MapIndex`]
    fn peak_at(&self, index: MapIndex) -> &C {
        &self.peak_table()[index.time_index][index.peak_index]
    }

    /// Convert a [`MapPath`] into an instance of [`Self::FeatureType`]
    fn path_to_feature(&self, path: &MapPath) -> Self::FeatureType {
        let mut feature = Self::FeatureType::default();
        for link in path.iter() {
            let peak = self.peak_at(link.from_index);
            let time = self.time_axis()[link.from_index.time_index];
            if let Some(last) = feature.end_time() {
                assert!(!isclose(time, last))
            }
            feature.push(peak, time);
        }

        if let Some(link) = path.last() {
            let peak = self.peak_at(link.to_index);
            let time = self.time_axis()[link.to_index.time_index];
            if let Some(last) = feature.end_time() {
                assert!(!isclose(time, last))
            }
            feature.push(peak, time);
        }
        feature
    }

    /// Merge features which are within `mass_error_tolerance` of each other in the `D` dimension
    /// and within `maximum_gap_size` time units of each other.
    ///
    /// Uses [`MapState::FeatureMergerType`] to carry out the actual merging.
    fn merge_features(
        &self,
        features: &FeatureMap<D, T, Self::FeatureType>,
        mass_error_tolerance: Tolerance,
        maximum_gap_size: f64,
    ) -> FeatureMap<D, T, Self::FeatureType> {
        let merger = Self::FeatureMergerType::default();
        merger.bridge_feature_gaps(features, mass_error_tolerance, maximum_gap_size)
    }
}

impl<
        C: IndexedCoordinate<D> + IntensityMeasurement + 'static,
        D: Default + 'static + Clone,
        T: Default + Clone,
    > MapState<C, D, T> for PeakMapState<C, D>
{
    type FeatureType = Feature<D, T>;
    type FeatureMergerType = FeatureMerger<D, T, Self::FeatureType>;

    fn time_axis(&self) -> &[f64] {
        &self.time_axis
    }

    fn peak_table(&self) -> &[PeakSetVec<C, D>] {
        &self.peak_table
    }

    fn populate_from_iterator(&mut self, it: impl Iterator<Item = (f64, PeakSetVec<C, D>)>) {
        for (time, peaks) in it {
            if let Some(t) = self.time_axis.last() {
                assert!(*t < time);
            }
            self.time_axis.push(time);
            self.peak_table.push(peaks);
        }
    }

    fn query_with_index<'a>(
        &'a self,
        query: &'a C,
        time_index: usize,
        error_tolerance: Tolerance,
    ) -> impl Iterator<Item = MapIndex> + 'a {
        let hits = self.peak_table[time_index].all_peaks_for(query.coordinate(), error_tolerance);
        hits.iter()
            .map(move |p| MapIndex::new(time_index, p.get_index() as usize))
    }
}

/// A [`MapState`] implementation that restricts peak matches to only those which have
/// the same charge state.
#[derive(Debug, Clone)]
pub struct ChargedPeakMapState<C: IndexedCoordinate<D> + IntensityMeasurement + KnownCharge, D> {
    pub time_axis: Vec<f64>,
    pub peak_table: Vec<PeakSetVec<C, D>>,
}

impl<C: IndexedCoordinate<D> + IntensityMeasurement + KnownCharge, D> Default
    for ChargedPeakMapState<C, D>
{
    fn default() -> Self {
        Self {
            time_axis: Vec::new(),
            peak_table: Vec::new(),
        }
    }
}

impl<
        C: IndexedCoordinate<D> + IntensityMeasurement + KnownCharge + 'static,
        D: Default + 'static + Clone,
        T: Default + Clone,
    > MapState<C, D, T> for ChargedPeakMapState<C, D>
{
    type FeatureType = ChargedFeature<D, T>;
    type FeatureMergerType = ChargeAwareFeatureMerger<D, T, Self::FeatureType>;

    fn time_axis(&self) -> &[f64] {
        &self.time_axis
    }

    fn peak_table(&self) -> &[PeakSetVec<C, D>] {
        &self.peak_table
    }

    fn populate_from_iterator(&mut self, it: impl Iterator<Item = (f64, PeakSetVec<C, D>)>) {
        for (time, peaks) in it {
            if let Some(t) = self.time_axis.last() {
                assert!(*t < time);
            }
            self.time_axis.push(time);
            self.peak_table.push(peaks);
        }
    }

    fn query_with_index<'a>(
        &'a self,
        query: &'a C,
        time_index: usize,
        error_tolerance: Tolerance,
    ) -> impl Iterator<Item = MapIndex> + 'a {
        let hits = self.peak_table[time_index].all_peaks_for(query.coordinate(), error_tolerance);
        hits.iter()
            .filter(|p| p.charge() == query.charge())
            .map(move |p| MapIndex::new(time_index, p.get_index() as usize))
    }

    fn path_to_feature(&self, path: &MapPath) -> Self::FeatureType {
        let mut feature = Self::FeatureType::default();
        for link in path.iter() {
            let peak: &C =
                <ChargedPeakMapState<C, D> as MapState<C, D, T>>::peak_at(self, link.from_index);
            feature.charge = peak.charge();
            let time = <ChargedPeakMapState<C, D> as MapState<C, D, T>>::time_axis(self)
                [link.from_index.time_index];
            feature.push(peak, time);
        }

        if let Some(link) = path.last() {
            let peak = <ChargedPeakMapState<C, D> as MapState<C, D, T>>::peak_at(self, link.to_index);
            let time = <ChargedPeakMapState<C, D> as MapState<C, D, T>>::time_axis(self)[link.to_index.time_index];
            if let Some(last) = feature.end_time() {
                assert!(!isclose(time, last))
            }
            feature.push(peak, time);
        }
        feature
    }
}

impl<C: IndexedCoordinate<D> + IntensityMeasurement + KnownCharge, D> ChargedPeakMapState<C, D> {
    pub fn new(time_axis: Vec<f64>, peak_table: Vec<PeakSetVec<C, D>>) -> Self {
        assert_eq!(time_axis.len(), peak_table.len());
        Self {
            time_axis,
            peak_table,
        }
    }
}

/// A [`ChargedPeakMapState`] can be built from an iterator, but it expects the iterator to be sorted over the time dimension, and
/// will panic if it encounters a non-monotonic sequence.
impl<C: IndexedCoordinate<D> + IntensityMeasurement + KnownCharge, D>
    FromIterator<(f64, PeakSetVec<C, D>)> for ChargedPeakMapState<C, D>
{
    fn from_iter<T: IntoIterator<Item = (f64, PeakSetVec<C, D>)>>(iter: T) -> Self {
        let mut this = Self::default();
        let mut last_time = f64::NEG_INFINITY;
        for (time, peaks) in iter {
            assert!(time > last_time);
            last_time = time;
            this.time_axis.push(time);
            this.peak_table.push(peaks);
        }
        this
    }
}

/// Represents a sequence of [`MapLink`] entries with a dynamic programming
/// score.
#[derive(Default, Debug, Clone, PartialEq)]
pub struct MapPath {
    indices: Vec<MapLink>,
    pub coordinate: f64,
    pub score: f32,
}

impl PartialOrd for MapPath {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(
            self.coordinate
                .total_cmp(&other.coordinate)
                .then_with(|| {
                    self.first()
                        .map(|i| i.from_index)
                        .cmp(&other.first().map(|i| i.from_index))
                })
                .then_with(|| {
                    self.last()
                        .map(|i| i.to_index)
                        .cmp(&other.last().map(|i| i.to_index))
                }),
        )
    }
}

impl MapPath {
    pub fn new(indices: Vec<MapLink>, coordinate: f64, score: f32) -> Self {
        Self {
            indices,
            coordinate,
            score,
        }
    }

    pub fn push(&mut self, value: MapLink) {
        self.indices.push(value)
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<MapLink> {
        self.indices.iter()
    }

    pub fn first(&self) -> Option<&MapLink> {
        self.indices.first()
    }

    pub fn last(&self) -> Option<&MapLink> {
        self.indices.last()
    }
}

type MapCell = Vec<MapLink>;
type MapCells = Vec<MapCell>;

/// Extracts features from a [`MapState`] type using dynamic programming.
#[derive(Debug, Clone)]
pub struct FeatureExtracterType<
    S: MapState<C, D, T>,
    C: IndexedCoordinate<D> + IntensityMeasurement + 'static,
    D: 'static,
    T: 'static,
> {
    state: S,
    paths: Vec<MapCells>,
    _c: PhantomData<C>,
    _d: PhantomData<D>,
    _t: PhantomData<T>,
}

impl<S: MapState<C, D, T>, C: IndexedCoordinate<D> + IntensityMeasurement, D, T>
    FeatureExtracterType<S, C, D, T>
{
    fn init(state: S, paths: Vec<MapCells>) -> Self {
        Self {
            state,
            paths,
            _c: PhantomData,
            _d: PhantomData,
            _t: PhantomData,
        }
    }

    pub fn new(state: S) -> Self {
        let mut paths = Vec::with_capacity(state.len());
        paths.resize_with(state.len(), MapCells::new);

        Self::init(state, paths)
    }

    /// Get a reference to the enclosed [`MapState`]
    pub fn inner(&self) -> &S {
        &self.state
    }

    /// Drop associated dynamic programming table and retrieve the enclosed [`MapState`]
    pub fn into_inner(self) -> S {
        self.state
    }

    fn build_index(&mut self, error_tolerance: Tolerance) {
        for i in 0..self.state.len().saturating_sub(1) {
            self.process_time_index(i, error_tolerance)
        }
    }

    fn process_time_index(&mut self, index: usize, error_tolerance: Tolerance) {
        let mut cells = MapCells::with_capacity(self.state.peak_table()[index].len());
        for peak in self.state.iter_at_index(index) {
            let from_index = MapIndex::new(index, peak.get_index() as usize);
            cells.push(
                self.state
                    .query_with_index(&peak, index + 1, error_tolerance)
                    .map(|to_index| {
                        let peak_at = self.state.peak_at(to_index);
                        let mass_error = error_tolerance
                            .call(peak_at.coordinate(), peak.coordinate())
                            .abs()
                            / error_tolerance.tol();
                        let intensity_weight =
                            (peak.intensity().log10() + peak_at.intensity().log10()) / 2.0;
                        MapLink::new(from_index, to_index, mass_error, intensity_weight)
                    })
                    .collect(),
            );
        }
        self.paths[index] = cells;
    }

    fn init_path_from_link(&self, link: &MapLink) -> MapPath {
        let from_peak = self.state.peak_at(link.from_index);
        let to_peak = self.state.peak_at(link.to_index);
        let coord = ((from_peak.coordinate() * from_peak.intensity() as f64)
            + (to_peak.coordinate() * to_peak.intensity() as f64))
            / ((from_peak.intensity() + to_peak.intensity()) as f64);
        MapPath::new(vec![*link], coord, link.score())
    }

    fn extend_path(&self, path: &mut MapPath, link: MapLink) {
        let score = link.score();
        path.push(link);
        path.score += score;
    }

    /// Extract features from the peak map, connecting peaks whose `D` mass coordinate
    /// is within `error_tolerance` units of each other, of at least `min_length` points
    /// long, and having gaps of no more than `maximum_gap_size` `T` time units wide.
    ///
    /// This uses two dynamic programming algorithms, first a single span greedy path
    /// building stage followed by a gap bridging stage to tie together disjoint features.
    pub fn extract_features(
        &mut self,
        error_tolerance: Tolerance,
        min_length: usize,
        maximum_gap_size: f64,
    ) -> FeatureMap<D, T, S::FeatureType> {
        self.build_index(error_tolerance);
        let paths = self.build_paths();
        let features = self.paths_to_features(&paths, min_length);
        let features = FeatureMap::new(features);
        let features = self
            .state
            .merge_features(&features, error_tolerance, maximum_gap_size);
        features
    }

    fn solve_layer_links(
        &self,
        segments: &mut Vec<MapPath>,
        index_link_weights: &mut Vec<(usize, &MapLink, f32)>,
        seen_nodes: &mut HashSet<MapIndex>,
    ) -> Vec<bool> {
        index_link_weights.sort_by(|a, b| a.2.total_cmp(&b.2).reverse());

        let mut seen_map_paths = vec![false; segments.len()];

        for (path_i, link, weight) in index_link_weights {
            if seen_map_paths[*path_i] {
                continue;
            } else if seen_nodes.contains(&link.to_index) {
                continue;
            } else {
                let path = &mut segments[*path_i];
                seen_map_paths[*path_i] = true;
                seen_nodes.insert(link.to_index);
                if log::log_enabled!(log::Level::Trace) {
                    let time_at = self.state.time_axis()[link.to_index.time_index];
                    let peak_at = self.state.peak_at(link.to_index);
                    log::trace!(
                        "Binding {link:?} with weight {weight} at coordinate {} {time_at}|{}|{}",
                        path.coordinate,
                        peak_at.coordinate(),
                        peak_at.get_index()
                    );
                }
                self.extend_path(path, **link);
            }
        }
        seen_map_paths
    }

    fn solve_layer_links_unbound(&self, link_weights: &mut Vec<(&MapLink, f32)>, seen_nodes: &mut HashSet<MapIndex>, working_paths: &mut Vec<MapPath>) {
        link_weights.sort_by(|a, b| {
            a.1.total_cmp(&b.1).reverse()
        });

        for (link, weight) in link_weights.drain(..) {
            if seen_nodes.contains(&link.to_index) {
                continue;
            }
            seen_nodes.insert(link.to_index);
            if log::log_enabled!(log::Level::Trace) {
                let time_at = self.state.time_axis()[link.to_index.time_index];
                let peak_at = self.state.peak_at(link.to_index);
                log::trace!(
                    "Creating path from {link:?} with weight {weight} at coordinate {time_at}|{}|{}",
                    peak_at.coordinate(),
                    peak_at.get_index()
                );
            }
            let path = self.init_path_from_link(link);
            working_paths.push(path);
        }
    }

    fn build_paths(&self) -> Vec<MapPath> {
        let mut active_paths: Vec<MapPath> = Vec::new();
        let mut seen_nodes: HashSet<MapIndex> = HashSet::new();
        let mut ended_paths: Vec<MapPath> = Vec::new();
        let mut working_paths: Vec<MapPath> = Vec::new();

        let mut segments: Vec<MapPath> = Vec::new();
        let mut unbound_link_weights: Vec<(&MapLink, f32)> = Vec::new();
        let mut index_link_weights: Vec<(usize, &MapLink, f32)> = Vec::new();
        let mut counter: usize = 0;

        for (row_idx, row) in self.paths.iter().enumerate() {
            for path in active_paths.drain(..) {
                let last_index = path.last().unwrap().to_index;
                // This path's terminal node steps on this row/time
                if last_index.time_index == row_idx {
                    if let Some(row_cell) = row.get(last_index.peak_index) {
                        // If the path has no links at this time point
                        if row_cell.is_empty() {
                            ended_paths.push(path)
                        } else {
                            for link in row_cell.iter() {
                                index_link_weights.push((counter, link, link.score()));
                            }
                            segments.push(path);
                            counter += 1;
                        }
                    } else {
                        ended_paths.push(path);
                    }
                // Otherwise this path has ended and we'll move to the finished paths
                } else {
                    ended_paths.push(path);
                }
            }

            let visited_paths =
                self.solve_layer_links(&mut segments, &mut index_link_weights, &mut seen_nodes);
            for (visited, path) in visited_paths.into_iter().zip(segments.drain(..)) {
                if visited {
                    working_paths.push(path)
                } else {
                    ended_paths.push(path)
                }
            }


            for links in row.iter() {
                for link in links {
                    unbound_link_weights.push((link, link.score()))
                }
            }

            self.solve_layer_links_unbound(&mut unbound_link_weights, &mut seen_nodes, &mut working_paths);

            swap(&mut active_paths, &mut working_paths);
            seen_nodes.clear();
            index_link_weights.clear();
            counter = 0;
        }

        ended_paths.extend(active_paths);
        ended_paths
    }

    fn paths_to_features(&self, paths: &[MapPath], min_length: usize) -> Vec<S::FeatureType> {
        // A `MapPath` of length n represents n + 1 links, and that's the length we're filtering on
        let min_length_sub_1 = min_length.saturating_sub(1);
        paths
            .iter()
            .filter(|p| p.len() > min_length_sub_1)
            .map(|path| self.state.path_to_feature(path))
            .collect()
    }
}

impl<S: MapState<C, D, T> + Default, C: IndexedCoordinate<D> + IntensityMeasurement, D, T>
    FromIterator<(f64, PeakSetVec<C, D>)> for FeatureExtracterType<S, C, D, T>
{
    fn from_iter<I: IntoIterator<Item = (f64, PeakSetVec<C, D>)>>(iter: I) -> Self {
        let mut state = S::default();
        state.populate_from_iterator(iter.into_iter());
        state.into()
    }
}

impl<S: MapState<C, D, T>, C: IndexedCoordinate<D> + IntensityMeasurement + 'static, D, T> From<S>
    for FeatureExtracterType<S, C, D, T>
{
    fn from(value: S) -> Self {
        Self::new(value)
    }
}

/// The basic form of [`FeatureExtracterType`]
pub type FeatureExtracter<C, T> = FeatureExtracterType<PeakMapState<C, MZ>, C, MZ, T>;
/// A [`FeatureExtracter`] over retention time
pub type LCMSMapExtracter<C> = FeatureExtracter<C, Time>;
/// A [`FeatureExtracter`] over drift time
pub type IMMSMapExtracter<C> = FeatureExtracter<C, IonMobility>;

/// A charge-aware form of [`FeatureExtracterType`]
pub type DeconvolvedFeatureExtracter<C, T> =
    FeatureExtracterType<ChargedPeakMapState<C, Mass>, C, Mass, T>;
/// A [`DeconvolvedFeatureExtracter`] over retention time
pub type DeconvolvedLCMSMapExtracter<C> = DeconvolvedFeatureExtracter<C, Time>;
/// A [`DeconvolvedFeatureExtracter`] over drift time
pub type DeconvolvedIMMSMapExtracter<C> = DeconvolvedFeatureExtracter<C, IonMobility>;

#[cfg(test)]
mod test {
    use mzpeaks::{CentroidPeak, MZPeakSetType, Time};

    use super::*;

    use crate::text::arrays_over_time_from_file;
    use std::{
        fs,
        io::{self, prelude::*},
    };

    #[test_log::test]
    fn test_construction2() -> io::Result<()> {
        let time_arrays = arrays_over_time_from_file("./test/data/peaks_over_time_tims.txt")?;
        let mut time_axis = Vec::new();
        let mut peak_table = Vec::new();

        for (t, row) in time_arrays {
            time_axis.push(t);
            let peaks: MZPeakSetType<CentroidPeak> = row
                .mz_array
                .into_iter()
                .zip(row.intensity_array.into_iter())
                .map(|(mz, i)| CentroidPeak::new(*mz, *i, 0))
                .collect();
            peak_table.push(peaks);
        }

        let mut peak_map_builder =
            FeatureExtracter::<_, IonMobility>::from_iter(time_axis.into_iter().zip(peak_table));
        let features = peak_map_builder.extract_features(Tolerance::PPM(15.0), 3, 0.1);
        if false {
            let mut writer = io::BufWriter::new(fs::File::create("features_graph_tims.txt")?);
            writer.write_all(b"feature_id\tmz\trt\tintensity\n")?;
            for (i, f) in features.iter().enumerate() {
                for (mz, rt, inten) in f.iter() {
                    writer.write_all(format!("{i}\t{mz}\t{rt}\t{inten}\n").as_bytes())?;
                }
            }
        }
        Ok(())
    }

    #[test_log::test]
    fn test_construction() -> io::Result<()> {
        let time_arrays = arrays_over_time_from_file("./test/data/peaks_over_time.txt")?;
        let mut time_axis = Vec::new();
        let mut peak_table = Vec::new();

        for (t, row) in time_arrays {
            time_axis.push(t);
            let peaks: MZPeakSetType<CentroidPeak> = row
                .mz_array
                .into_iter()
                .zip(row.intensity_array.into_iter())
                .map(|(mz, i)| CentroidPeak::new(*mz, *i, 0))
                .collect();
            peak_table.push(peaks);
        }

        let mut peak_map_builder =
            FeatureExtracter::<_, Time>::from_iter(time_axis.into_iter().zip(peak_table));
        let features = peak_map_builder.extract_features(Tolerance::PPM(10.0), 3, 0.25);

        if false {
            let mut writer = io::BufWriter::new(fs::File::create("features_graph.txt")?);
            writer.write_all(b"feature_id\tmz\trt\tintensity\n")?;
            for (i, f) in features.iter().enumerate() {
                for (mz, rt, inten) in f.iter() {
                    writer.write_all(format!("{i}\t{mz}\t{rt}\t{inten}\n").as_bytes())?;
                }
            }
        }

        eprintln!("Extracted {} features", features.len());
        assert_eq!(features.len(), 15427);
        Ok(())
    }
}
