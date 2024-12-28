//! Helper types for representing coordinates over a [`MapState`]

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Represents a coordinate in a [`PeakMapState`], referencing a specific peak
/// at a specific time.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MapLink {
    pub from_index: MapIndex,
    pub to_index: MapIndex,
    pub mass_error: f64,
    pub intensity_weight: f32,
}

impl std::hash::Hash for MapLink {
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



/// Represents a sequence of [`MapLink`] entries with a dynamic programming
/// score.
#[derive(Default, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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

pub(crate) type MapCell = Vec<MapLink>;
pub(crate) type MapCells = Vec<MapCell>;
