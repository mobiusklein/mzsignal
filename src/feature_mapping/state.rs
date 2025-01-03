use std::marker::PhantomData;
use std::mem;

use mzpeaks::{
    feature::{ChargedFeature, Feature},
    feature_map::FeatureMap,
    peak_set::PeakSetVec,
    prelude::*,
    Tolerance,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::peak_statistics::isclose;
use crate::search::nearest;

use super::map::*;

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
        this._from_iter(iter);
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

    fn _from_iter<T: IntoIterator<Item = (f64, PeakSetVec<C, D>)>>(&mut self, iter: T) {
        let mut last_time = f64::NEG_INFINITY;
        for (time, peaks) in iter {
            assert!(time > last_time);
            last_time = time;
            self.time_axis.push(time);
            self.peak_table.push(peaks);
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

    fn iter_all_nodes(&self) -> impl Iterator<Item = MapIndex> + '_ {
        self.peak_table().iter().enumerate().map(|(i, ps)| {
            (0..ps.len()).into_iter().map(move |j| MapIndex::new(i, j))
        }).flatten()
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

    /// Convert a single [`MapIndex`] into an instance of [`Self::FeatureType`],
    /// usually representing an orphaned peak that had no links.
    fn node_to_feature(&self, index: &MapIndex) -> Self::FeatureType {
        let peak = self.peak_at(*index);
        let time = self.time_axis()[index.time_index];
        let mut feature = Self::FeatureType::default();
        feature.push(peak, time);
        feature
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
        self._from_iter(it);
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
        self._from_iter(it);
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

    fn node_to_feature(&self, index: &MapIndex) -> Self::FeatureType {
        let peak = <ChargedPeakMapState<C, D> as MapState<C, D, T>>::peak_at(self, *index);
        let time = <ChargedPeakMapState<C, D> as MapState<C, D, T>>::time_axis(self)[index.time_index];
        let mut feature = Self::FeatureType::default();
        feature.push(peak, time);
        feature.charge = peak.charge();
        feature
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
            let peak =
                <ChargedPeakMapState<C, D> as MapState<C, D, T>>::peak_at(self, link.to_index);
            let time = <ChargedPeakMapState<C, D> as MapState<C, D, T>>::time_axis(self)
                [link.to_index.time_index];
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

    fn _from_iter<T: IntoIterator<Item = (f64, PeakSetVec<C, D>)>>(&mut self, iter: T) {
        let mut last_time = f64::NEG_INFINITY;
        for (time, peaks) in iter {
            assert!(time > last_time);
            last_time = time;
            self.time_axis.push(time);
            self.peak_table.push(peaks);
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
        this._from_iter(iter);
        this
    }
}

struct FeatureMergeQueue<'a, D, T, F: FeatureLike<D, T> + FeatureLikeMut<D, T> + Clone> {
    accumulator: F,
    features: &'a [&'a F],
    indices: Vec<usize>,
    _d: PhantomData<D>,
    _t: PhantomData<T>,
}

impl<'a, D, T, F: FeatureLike<D, T> + FeatureLikeMut<D, T> + Clone> FeatureMergeQueue<'a, D, T, F> {
    fn from_vec(features: &'a [&'a F]) -> Self {
        let mut acc = features[0].clone();
        acc.clear();
        let indices = vec![0; features.len()];
        Self {
            accumulator: acc,
            features,
            indices,
            _d: PhantomData,
            _t: PhantomData,
        }
    }

    fn merge(mut self) -> F {
        while let Some((x, y, z)) = self.next_point() {
            self.accumulator.push_raw(x, y, z);
        }
        self.accumulator
    }

    fn next_point(&mut self) -> Option<(f64, f64, f32)> {
        let coords = self
            .indices
            .iter()
            .copied()
            .enumerate()
            .zip(self.features)
            .filter_map(|((f_idx, t_idx), f)| {
                let pt = f.at(t_idx);
                if let Some(pt) = pt {
                    Some((f_idx, t_idx, pt))
                } else {
                    None
                }
            })
            .reduce(
                |(earliest_f_idx, earliest_t_idx, earliest_pt), (f_idx, t_idx, pt)| {
                    if pt.1 < earliest_pt.1 {
                        (f_idx, t_idx, pt)
                    } else {
                        (earliest_f_idx, earliest_t_idx, earliest_pt)
                    }
                },
            );

        if let Some((f_idx, _, pt)) = coords {
            self.indices[f_idx] += 1;
            Some(pt)
        } else {
            None
        }
    }
}

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
            let features_of: Vec<_> = component_indices
                .into_iter()
                .map(|i| features.get_item(i))
                .collect();

            let acc = if features_of.len() == 1 {
                features_of[0].clone()
            } else {
                FeatureMergeQueue::from_vec(&features_of).merge()
            };

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
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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

    #[allow(unused)]
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

    impl<'a, D: std::ops::Deref, T: std::ops::Deref, F: FeatureLike<D, T> + std::ops::Deref>
        std::ops::Deref for IndexedFeature<'a, D, T, F>
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

        fn iter(&self) -> impl Iterator<Item = (f64, f64, f32)> {
            self.feature.iter()
        }
    }
}

use index_impl::IndexedFeature;
