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
use std::collections::HashSet;
use std::marker::PhantomData;
use std::mem::swap;

use mzpeaks::{
    feature_map::FeatureMap, peak_set::PeakSetVec, prelude::*, IonMobility, Mass, Time, Tolerance,
    MZ,
};

mod feature_wrap;
pub mod map;
mod state;

use map::*;
pub use state::{
    ChargeAwareFeatureMerger, ChargedPeakMapState, FeatureGraphBuilder, FeatureLink, FeatureMerger,
    FeatureNode, IonMobilityChargedPeakMapState, MapState, PeakMapState,
};

#[doc(hidden)]
pub mod graph {
    pub use super::*;
}

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



/// Get a mutable reference ot the enclosed [`MapState`].
///
/// Care must be taken not to invalidate the invariants.
impl<S: MapState<C, D, T>, C: IndexedCoordinate<D> + IntensityMeasurement, D, T> AsMut<S> for FeatureExtracterType<S, C, D, T> {
    fn as_mut(&mut self) -> &mut S {
        &mut self.state
    }
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
            self.paths[i] = self.process_time_at(i, error_tolerance);
        }
    }

    fn process_time_at(&self, index: usize, error_tolerance: Tolerance) -> Vec<Vec<MapLink>> {
        let mut cells = MapCells::with_capacity(self.state.peak_table()[index].len());
        for peak in self.state.iter_at_index(index) {
            let from_index = MapIndex::new(index, peak.get_index() as usize);
            let links: Vec<MapLink> = self
                .state
                .query_with_index(peak, index + 1, error_tolerance)
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
                .collect();
            cells.push(links);
        }
        cells
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

    fn extract_orphan_nodes(
        &self,
        all_used_nodes: MapSet,
    ) -> impl Iterator<Item = S::FeatureType> + '_ {
        self.state
            .iter_all_nodes()
            .filter(move |i| !all_used_nodes.contains(i))
            .map(|i| self.state.node_to_feature(&i))
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
        let (paths, all_used_nodes) = self.build_paths();
        let mut features = self.paths_to_features(&paths, min_length);
        features.extend(self.extract_orphan_nodes(all_used_nodes));
        let features = FeatureMap::new(features);
        let features = self
            .state
            .merge_features(&features, error_tolerance, maximum_gap_size);
        // TODO: When viable to introduce a breaking change, push minimum length filter into `merge_features`
        features
            .into_iter()
            .filter(|f| f.len() >= min_length)
            .collect()
    }

    fn solve_layer_links(
        &self,
        segments: &mut [MapPath],
        index_link_weights: &mut Vec<(usize, &MapLink, f32)>,
        seen_nodes: &mut HashSet<MapIndex>,
    ) -> Vec<bool> {
        index_link_weights.sort_by(|a, b| a.2.total_cmp(&b.2).reverse());

        let mut seen_map_paths = vec![false; segments.len()];

        for (path_i, link, weight) in index_link_weights {
            if seen_map_paths[*path_i] || seen_nodes.contains(&link.to_index) || seen_nodes.contains(&link.from_index) {
                continue;
            } else {
                let path = &mut segments[*path_i];
                seen_map_paths[*path_i] = true;
                seen_nodes.insert(link.to_index);
                seen_nodes.insert(link.from_index);
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

    fn create_paths_from_unbound_links(
        &self,
        link_weights: &mut Vec<(&MapLink, f32)>,
        seen_nodes: &mut HashSet<MapIndex>,
        working_paths: &mut Vec<MapPath>,
    ) {
        link_weights.sort_by(|a, b| a.1.total_cmp(&b.1).reverse());

        for (link, weight) in link_weights.drain(..) {
            // We've already visited this node, so we can't use it to build a new link
            if seen_nodes.contains(&link.to_index) || seen_nodes.contains(&link.from_index) {
                continue;
            }
            seen_nodes.insert(link.to_index);
            seen_nodes.insert(link.from_index);
            let path = self.init_path_from_link(link);
            if log::log_enabled!(log::Level::Trace) {
                let time_at = self.state.time_axis()[link.to_index.time_index];
                let peak_at = self.state.peak_at(link.to_index);
                log::trace!(
                    "Creating path from {link:?} with weight {weight} at coordinate {time_at}|{}|{}",
                    peak_at.coordinate(),
                    peak_at.get_index()
                );
            }
            working_paths.push(path);
        }
    }

    #[inline(always)]
    fn check_for_duplicate_nodes(&self, paths: &[MapPath]) {
        #[cfg(debug_assertions)]
        {
            let mut seen_nodes =
                std::collections::HashMap::<MapIndex, Vec<(MapLink, usize)>>::new();
            for (i, path) in paths.iter().enumerate() {
                let link = path.last().copied().unwrap();
                seen_nodes
                    .entry(link.from_index)
                    .or_default()
                    .push((link, i))
            }
            for (index, links) in seen_nodes {
                if links.len() > 1 {
                    log::warn!("{index:?} is used multiple times: {links:?}");
                }
            }
        }
    }

    fn build_paths(&self) -> (Vec<MapPath>, MapSet) {
        let mut active_paths: Vec<MapPath> = Vec::new();
        let mut seen_nodes: HashSet<MapIndex> = HashSet::new();
        let mut ended_paths: Vec<MapPath> = Vec::new();
        let mut working_paths: Vec<MapPath> = Vec::new();

        let mut all_used_nodes = MapSet::default();

        let mut segments: Vec<MapPath> = Vec::new();
        let mut unbound_link_weights: Vec<(&MapLink, f32)> = Vec::new();
        let mut index_link_weights: Vec<(usize, &MapLink, f32)> = Vec::new();
        let mut segment_index_counter: usize = 0;

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
                            // This path will (potentially) be extended at this time point, add all its potential
                            // extensions to the `index_link_weights` buffer at the next segment index, add the
                            // path to the segments buffer, and increment the index.
                            for link in row_cell.iter() {
                                index_link_weights.push((
                                    segment_index_counter,
                                    link,
                                    link.score(),
                                ));
                            }
                            segments.push(path);
                            segment_index_counter += 1;
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

            self.create_paths_from_unbound_links(
                &mut unbound_link_weights,
                &mut seen_nodes,
                &mut working_paths,
            );

            self.check_for_duplicate_nodes(&working_paths);

            for index in seen_nodes.drain() {
                all_used_nodes.insert(&index);
            }

            swap(&mut active_paths, &mut working_paths);
            seen_nodes.clear();
            index_link_weights.clear();
            segment_index_counter = 0;
        }

        ended_paths.extend(active_paths);
        ended_paths.sort_by(|a, b| a.coordinate.total_cmp(&b.coordinate));
        (ended_paths, all_used_nodes)
    }

    fn paths_to_features(&self, paths: &[MapPath], min_length: usize) -> Vec<S::FeatureType> {
        paths
            .iter()
            .map(|path| self.state.path_to_feature(path))
            .filter(|p| p.len() >= min_length)
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

pub type DeconvolutedLCIMMSMapExtracter<C> =
    FeatureExtracterType<IonMobilityChargedPeakMapState<C, Mass>, C, Mass, Time>;

#[cfg(test)]
mod test {
    use mzpeaks::{
        peak::IonMobilityAwareDeconvolutedPeak, CentroidPeak, DeconvolutedPeak,
        DeconvolutedPeakSet, MZPeakSetType, Time,
    };

    use super::*;

    use crate::text::arrays_over_time_from_file;
    use std::io::{self};

    #[test_log::test]
    #[test_log(default_log_filter = "debug")]
    // This test doesn't actually do anything at run time, but it proves that this type
    // collection works at build time.
    fn test_lc_im_ms() -> io::Result<()> {
        let mut extractor: DeconvolutedLCIMMSMapExtracter<IonMobilityAwareDeconvolutedPeak> =
            DeconvolutedLCIMMSMapExtracter::new(IonMobilityChargedPeakMapState::default());
        extractor.as_mut().ion_mobility_error_tolerance = 0.1;
        extractor.extract_features(Tolerance::PPM(15.0), 2, 0.5);
        Ok(())
    }

    #[test_log::test]
    #[test_log(default_log_filter = "debug")]
    fn test_construction2() -> io::Result<()> {
        let time_arrays = arrays_over_time_from_file("./test/data/peaks_over_time_tims.txt")?;
        let mut time_axis = Vec::new();
        let mut peak_table = Vec::new();

        for (t, row) in time_arrays {
            time_axis.push(t);
            let peaks: MZPeakSetType<CentroidPeak> = row
                .mz_array
                .iter()
                .zip(row.intensity_array.iter())
                .map(|(mz, i)| CentroidPeak::new(*mz, *i, 0))
                .collect();
            peak_table.push(peaks);
        }

        let mut peak_map_builder =
            FeatureExtracter::<_, IonMobility>::from_iter(time_axis.into_iter().zip(peak_table));
        let features = peak_map_builder.extract_features(Tolerance::PPM(15.0), 3, 0.005);
        if false {
            crate::text::write_feature_table("features_graph_tims.txt", features.iter())?;
        }
        assert_eq!(features.len(), 792);
        Ok(())
    }

    #[test_log::test]
    #[test_log(default_log_filter = "debug")]
    fn test_construction() -> io::Result<()> {
        let time_arrays = arrays_over_time_from_file("./test/data/peaks_over_time.txt")?;
        let mut time_axis = Vec::new();
        let mut peak_table = Vec::new();

        for (t, row) in time_arrays {
            time_axis.push(t);
            let peaks: MZPeakSetType<CentroidPeak> = row
                .mz_array
                .iter()
                .zip(row.intensity_array.iter())
                .map(|(mz, i)| CentroidPeak::new(*mz, *i, 0))
                .collect();
            peak_table.push(peaks);
        }

        let mut peak_map_builder =
            FeatureExtracter::<_, Time>::from_iter(time_axis.into_iter().zip(peak_table));
        let features = peak_map_builder.extract_features(Tolerance::PPM(10.0), 3, 0.25);

        if false {
            crate::text::write_feature_table("feature_graph_expanded.txt", features.iter())?;
        }

        eprintln!("Extracted {} features", features.len());
        assert_eq!(features.len(), 25784);
        Ok(())
    }

    #[test_log::test]
    #[test_log(default_log_filter = "debug")]
    fn test_construction_charged() -> io::Result<()> {
        let time_arrays = arrays_over_time_from_file("./test/data/peaks_over_time.txt")?;
        let mut time_axis = Vec::new();
        let mut peak_table = Vec::new();

        for (t, row) in time_arrays {
            time_axis.push(t);
            let peaks: DeconvolutedPeakSet = row
                .mz_array
                .iter()
                .zip(row.intensity_array.iter())
                .map(|(mz, i)| DeconvolutedPeak::new(*mz, *i, 1, 0))
                .collect();
            peak_table.push(peaks);
        }

        let mut peak_map_builder = DeconvolvedFeatureExtracter::<_, Time>::from_iter(
            time_axis.into_iter().zip(peak_table),
        );
        let features = peak_map_builder.extract_features(Tolerance::PPM(10.0), 3, 0.25);

        eprintln!("Extracted {} features", features.len());
        assert_eq!(features.len(), 25784);
        Ok(())
    }
}
