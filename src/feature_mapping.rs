use std::collections::HashSet;
use std::hash::Hash;
use std::marker::PhantomData;
use std::mem::swap;
use std::ops::Index;

use mzpeaks::{feature::{ChargedFeature, Feature}, peak_set::PeakSetVec, prelude::*, CentroidPeak, IonMobility, Mass, Time, Tolerance, MZ, feature_map::FeatureMap};

use crate::search::nearest;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MapIndex {
    time_index: usize,
    peak_index: usize,
}

impl MapIndex {
    pub fn new(time_index: usize, peak_index: usize) -> Self {
        Self {
            time_index,
            peak_index,
        }
    }

    pub fn time_index(&self) -> usize {
        self.time_index
    }

    pub fn time_index_mut(&mut self) -> &mut usize {
        &mut self.time_index
    }

    pub fn peak_index(&self) -> usize {
        self.peak_index
    }

    pub fn peak_index_mut(&mut self) -> &mut usize {
        &mut self.peak_index
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, PartialOrd)]
pub struct MapLink {
    from_index: MapIndex,
    to_index: MapIndex,
    mass_error: f64,
    intensity_weight: f32,
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

    pub fn score(&self) -> f32 {
        self.intensity_weight * (self.mass_error.powi(4) as f32)
    }
}

#[derive(Default, Debug, Clone)]
pub struct PeakMapState<C: IndexedCoordinate<D> + IntensityMeasurement, D> {
    pub time_axis: Vec<f64>,
    pub peak_table: Vec<PeakSetVec<C, D>>,
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

pub trait MapState<C: IndexedCoordinate<D> + IntensityMeasurement + 'static, D: 'static, T> {
    type FeatureType: FeatureLike<D, T> + Default + FeatureLikeMut<D, T>;

    fn time_axis(&self) -> &[f64];
    fn peak_table(&self) -> &[PeakSetVec<C, D>];
    fn populate_from_iterator(&mut self, it: impl Iterator<Item=(f64, PeakSetVec<C, D>)>);

    fn len(&self) -> usize {
        self.time_axis().len()
    }

    fn is_empty(&self) -> bool {
        self.time_axis().is_empty()
    }

    fn iter_at_index(&self, time_index: usize) -> impl Iterator<Item = &C> {
        self.peak_table()[time_index].iter()
    }

    fn time_delta(&self, i: usize, j: usize) -> Option<f64> {
        let ti = self.time_axis().get(i);
        let tj = self.time_axis().get(j);

        if let (Some(ti), Some(tj)) = (ti.copied(), tj.copied()) {
            Some(ti - tj)
        } else {
            None
        }
    }

    fn nearest_time_point(&self, time: f64) -> usize {
        nearest(&self.time_axis(), time, 0)
    }

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

    fn query<'a>(
        &'a self,
        query: &'a C,
        time: f64,
        error_tolerance: Tolerance,
    ) -> impl Iterator<Item = MapIndex> + 'a {
        let time_index = self.nearest_time_point(time);
        self.query_with_index(query, time_index, error_tolerance)
    }

    fn peak_at(&self, index: MapIndex) -> &C {
        &self.peak_table()[index.time_index][index.peak_index]
    }

    fn path_to_feature(&self, path: &MapPath) -> Self::FeatureType {
        let mut feature = Self::FeatureType::default();
        for link in path.iter() {
            let peak = self.peak_at(link.from_index);
            let time = self.time_axis()[link.from_index.time_index];
            feature.push(peak, time);
        }
        feature
    }
}

impl<
        C: IndexedCoordinate<D> + IntensityMeasurement + 'static,
        D: Default + 'static,
        T: Default,
    > MapState<C, D, T> for PeakMapState<C, D>
{
    type FeatureType = Feature<D, T>;

    fn time_axis(&self) -> &[f64] {
        &self.time_axis
    }

    fn peak_table(&self) -> &[PeakSetVec<C, D>] {
        &self.peak_table
    }

    fn populate_from_iterator(&mut self, it: impl Iterator<Item=(f64, PeakSetVec<C, D>)>) {
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


#[derive(Default, Debug, Clone)]
pub struct ChargedPeakMapState<C: IndexedCoordinate<D> + IntensityMeasurement + KnownCharge, D> {
    pub time_axis: Vec<f64>,
    pub peak_table: Vec<PeakSetVec<C, D>>,
}

impl<C: IndexedCoordinate<D> + IntensityMeasurement + KnownCharge + 'static, D: Default + 'static, T: Default> MapState<C, D, T> for ChargedPeakMapState<C, D> {
    type FeatureType = ChargedFeature<D, T>;

    fn time_axis(&self) -> &[f64] {
        &self.time_axis
    }

    fn peak_table(&self) -> &[PeakSetVec<C, D>] {
        &self.peak_table
    }

    fn populate_from_iterator(&mut self, it: impl Iterator<Item=(f64, PeakSetVec<C, D>)>) {
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
            let peak: &C = <ChargedPeakMapState<C, D> as MapState<C, D, T>>::peak_at(self, link.from_index);
            feature.charge = peak.charge();
            let time = <ChargedPeakMapState<C, D> as MapState<C, D, T>>::time_axis(self)[link.from_index.time_index];
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


#[derive(Default, Debug, Clone, PartialEq)]
pub struct MapPath {
    indices: Vec<MapLink>,
    coordinate: f64,
    score: f32,
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

pub type MapCell = Vec<MapLink>;
pub type MapCells = Vec<MapCell>;

#[derive(Debug, Clone)]
pub struct PeakMapBuilderType<
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
    PeakMapBuilderType<S, C, D, T>
{
    pub fn new(state: S, paths: Vec<MapCells>) -> Self {
        Self {
            state,
            paths,
            _c: PhantomData,
            _d: PhantomData,
            _t: PhantomData,
        }
    }

    pub fn from_state(state: S) -> Self {
        let mut this = Self::new(state, Vec::new());
        this.paths.resize_with(this.state.len(), MapCells::new);
        this
    }

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
                            / error_tolerance.bounds(peak_at.coordinate()).1;
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
        let coord = (from_peak.coordinate() * from_peak.intensity() as f64)
            + (to_peak.coordinate() * to_peak.intensity() as f64)
                / ((from_peak.intensity() + to_peak.intensity()) as f64);
        MapPath::new(vec![*link], coord, link.score())
    }

    fn extend_path(&self, path: &mut MapPath, link: MapLink) {
        let score = link.score();
        path.push(link);
        path.score += score;
    }

    pub fn extract_features(&mut self, error_tolerance: Tolerance, min_length: usize) -> FeatureMap<D, T, S::FeatureType> {
        self.build_index(error_tolerance);
        let paths = self.build_paths();
        let features = self.paths_to_features(&paths, min_length);
        FeatureMap::new(features)
    }

    fn build_paths(&self) -> Vec<MapPath> {
        let mut active_paths: Vec<MapPath> = Vec::new();
        let mut seen_nodes: HashSet<MapIndex> = HashSet::new();
        let mut ended_paths: Vec<MapPath> = Vec::new();
        let mut working_paths: Vec<MapPath> = Vec::new();

        for (row_idx, row) in self.paths.iter().enumerate() {
            for mut path in active_paths.drain(..) {
                let last_index = path.last().unwrap().to_index;
                if last_index.time_index == row_idx {
                    if let Some(row_cell) = row.get(last_index.peak_index) {
                        if let Some(link) = row_cell
                            .iter()
                            .max_by(|a, b| a.score().total_cmp(&b.score()))
                        {
                            seen_nodes.insert(link.to_index);
                            self.extend_path(&mut path, *link);
                            working_paths.push(path);
                        } else {
                            ended_paths.push(path);
                        };
                    } else {
                        ended_paths.push(path);
                    }
                } else {
                    ended_paths.push(path);
                }
            }

            for links in row {
                for link in links {
                    if seen_nodes.contains(&link.to_index) {
                        continue;
                    } else {
                        let path = self.init_path_from_link(link);
                        working_paths.push(path);
                    }
                }
            }
            swap(&mut active_paths, &mut working_paths);
            seen_nodes.clear();
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
    FromIterator<(f64, PeakSetVec<C, D>)> for
    PeakMapBuilderType<S, C, D, T> {

    fn from_iter<I: IntoIterator<Item = (f64, PeakSetVec<C, D>)>>(iter: I) -> Self {
        let mut state = S::default();
        state.populate_from_iterator(iter.into_iter());
        state.into()
    }
}

impl<S: MapState<C, D, T>, C: IndexedCoordinate<D> + IntensityMeasurement + 'static, D, T> From<S>
    for PeakMapBuilderType<S, C, D, T>
{
    fn from(value: S) -> Self {
        Self::from_state(value)
    }
}

pub type PeakMapBuilder<C, T> = PeakMapBuilderType<PeakMapState<C, MZ>, C, MZ, T>;
pub type LCMSMapBuilder<C> = PeakMapBuilder<C, Time>;
pub type IMMSMapBuilder<C> = PeakMapBuilder<C, IonMobility>;

pub type DeconvolvedPeakMapBuilder<C, T> = PeakMapBuilderType<ChargedPeakMapState<C, Mass>, C, Mass, T>;
pub type DeconvolvedLCMSMapBuilder<C> = DeconvolvedPeakMapBuilder<C, Time>;
pub type DeconvolvedIMMSMapBuilder<C> = DeconvolvedPeakMapBuilder<C, IonMobility>;


#[cfg(test)]
mod test {
    use mzpeaks::{CentroidPeak, MZPeakSetType, Time};

    use super::*;

    use crate::text::arrays_over_time_from_file;
    use std::{
        fs,
        io::{self, prelude::*},
    };

    #[test]
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

        let peak_map_state = PeakMapState::new(time_axis, peak_table);
        let mut peak_map_builder = PeakMapBuilder::<_, Time>::from_state(peak_map_state);
        peak_map_builder.build_index(Tolerance::PPM(10.0));
        let paths = peak_map_builder.build_paths();
        let features = peak_map_builder.paths_to_features(&paths, 3);

        if false {
            let mut writer = io::BufWriter::new(fs::File::create("features.txt")?);
            writer.write_all(b"feature_id\tmz\trt\tintensity\n")?;
            for (i, f) in features.iter().enumerate() {
                for (mz, rt, inten) in f.iter() {
                    writer.write_all(format!("{i}\t{mz}\t{rt}\t{inten}\n").as_bytes())?;
                }
            }
        }

        eprintln!("Extracted {} features", features.len());
        assert_eq!(features.len(), 18362);
        Ok(())
    }
}
