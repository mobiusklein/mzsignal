use std::{borrow::Cow, iter::FusedIterator, ops::{Deref, Range}};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use mzpeaks::prelude::TimeArray;

use crate::{trapz, ArrayPairSplit};



/// An iterator over [`PeakFitArgs`] which explicitly casts the signal magnitude
/// from `f32` to `f64`.
///
/// This conversion is done specifically for convenience during model fitting.
pub struct PeakFitArgsIter<'a> {
    inner: std::iter::Zip<
        std::iter::Copied<std::slice::Iter<'a, f64>>,
        std::iter::Copied<std::slice::Iter<'a, f32>>,
    >,
}

impl<'a> Iterator for PeakFitArgsIter<'a> {
    type Item = (f64, f64);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(x, y)| (x, y as f64))
    }
}

impl<'a> FusedIterator for PeakFitArgsIter<'a> {}

impl<'a> ExactSizeIterator for PeakFitArgsIter<'a> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<'a> PeakFitArgsIter<'a> {
    pub fn new(
        inner: std::iter::Zip<
            std::iter::Copied<std::slice::Iter<'a, f64>>,
            std::iter::Copied<std::slice::Iter<'a, f32>>,
        >,
    ) -> Self {
        Self { inner }
    }
}

/// A point along a [`PeakFitArgs`] which produces the greatest
/// valley between two peaks.
///
/// Produced by [`PeakFitArgs::locate_extrema`]
#[derive(Debug, Default, Clone, Copy, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SplittingPoint {
    /// The signal magnitude of the first maximum point
    pub first_maximum_height: f32,
    /// The signal magnitude at the nadir of the valley
    pub minimum_height: f32,
    /// The signal magnitude of the second maximum point
    pub second_maximum_height: f32,
    /// The time coordinate of the nadir of the valley
    pub minimum_time: f64,
}

impl SplittingPoint {
    pub fn new(first_maximum: f32, minimum: f32, second_maximum: f32, minimum_index: f64) -> Self {
        Self {
            first_maximum_height: first_maximum,
            minimum_height: minimum,
            second_maximum_height: second_maximum,
            minimum_time: minimum_index,
        }
    }

    pub fn total_distance(&self) -> f32 {
        (self.first_maximum_height - self.minimum_height)
            + (self.second_maximum_height - self.minimum_height)
    }
}

/// Represent an array pair for signal-over-time data
#[derive(Debug, Default, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PeakFitArgs<'a, 'b> {
    /// The time axis of the signal
    pub time: Cow<'a, [f64]>,
    /// The paired signal intensity to fit against
    pub intensity: Cow<'b, [f32]>,
}

impl<'c, 'd, 'a: 'c, 'b: 'd, 'e: 'c + 'd + 'a + 'b> PeakFitArgs<'a, 'b> {
    pub fn new(time: Cow<'a, [f64]>, intensity: Cow<'b, [f32]>) -> Self {
        assert_eq!(
            time.len(),
            intensity.len(),
            "time array length ({}) must equal intensity length ({})",
            time.len(),
            intensity.len()
        );
        Self { time, intensity }
    }

    pub fn duration(&self) -> f64 {
        if self.time.is_empty() {
            return 0.0
        }
        self.time.last().copied().unwrap() - self.time.first().copied().unwrap()
    }

    pub fn average_spacing(&self) -> f64 {
        let (acc, wt, _last) = self.iter().fold((0.0, 0.0, Option::<f64>::None), |(acc, wt, last), (x, _y)| {
            let diff = match last {
                Some(last) => {
                    x - last
                },
                None => {0.0}
            };
            (acc + diff, wt + 1.0, Some(x))
        });
        acc / wt
    }

    pub fn rebin(&self, dx: f64) -> PeakFitArgs {
        let pair = crate::average::rebin(&self.time, &self.intensity, dx);
        PeakFitArgs::new(pair.mz_array, pair.intensity_array)
    }

    /// Apply [`moving_average_dyn`](crate::smooth::moving_average_dyn) to the signal intensity
    /// returning a new [`PeakFitArgs`].
    pub fn smooth(&'e self, window_size: usize) -> PeakFitArgs<'a, 'd> {
        let mut store = self.borrow();
        let sink = store.intensity.to_mut();
        crate::smooth::moving_average_dyn(&self.intensity, sink.as_mut_slice(), window_size * 3);
        store
    }

    /// Find the indices of local maxima, optionally above some `min_height` threshold.
    pub fn peak_indices(&self, min_height: Option<f64>) -> Vec<usize> {
        let min_height = min_height.unwrap_or_default() as f32;
        let n = self.len();
        let n1 = n.saturating_sub(1);
        let mut indices = Vec::new();
        for (i, y) in self
            .intensity
            .iter()
            .copied()
            .enumerate()
            .take(n.saturating_sub(1))
            .skip(1)
        {
            if y > min_height
                && y >= self.intensity[i.saturating_sub(1)]
                && y >= self.intensity[(i + 1).min(n1)]
            {
                indices.push(i);
            }
        }
        indices
    }

    /// Find the indices of local minima, optionally below some `max_height` threshold.
    pub fn valley_indices(&self, max_height: Option<f64>) -> Vec<usize> {
        let max_height = max_height.unwrap_or(f64::INFINITY) as f32;
        let n = self.len();
        let n1 = n.saturating_sub(1);
        let mut indices = Vec::new();
        for (i, y) in self
            .intensity
            .iter()
            .copied()
            .enumerate()
            .take(n.saturating_sub(1))
            .skip(1)
        {
            if y < max_height
                && y <= self.intensity[i.saturating_sub(1)]
                && y <= self.intensity[(i + 1).min(n1)]
            {
                indices.push(i);
            }
        }
        indices
    }

    /// Find a point between two local maxima that would separate the signal between the two peaks
    pub fn locate_extrema(&self, min_height: Option<f64>) -> Option<SplittingPoint> {
        let maxima_indices = self.peak_indices(min_height);
        let minima_indices = self.valley_indices(None);

        let mut candidates = Vec::new();

        for (i, max_i) in maxima_indices.iter().copied().enumerate() {
            for j in (i + 1)..maxima_indices.len() {
                let max_j = maxima_indices[j];
                for min_k in minima_indices.iter().copied() {
                    if self.time[max_i] > self.time[min_k] || self.time[min_k] > self.time[max_j] {
                        continue;
                    }
                    let y_i = self.intensity[max_i];
                    let y_j = self.intensity[max_j];
                    let y_k = self.intensity[min_k];
                    if max_i < min_k
                        && min_k < max_j
                        && (y_i - y_k) > (y_i * 0.01)
                        && (y_j - y_k) > (y_j * 0.01)
                    {
                        candidates.push(SplittingPoint::new(y_i, y_k, y_j, self.time[min_k]));
                    }
                }
            }
        }
        let split_point = candidates
            .into_iter()
            .max_by(|a, b| a.total_distance().total_cmp(&b.total_distance()));
        split_point
    }

    /// Find the indices that separate the signal into discrete segments according
    /// to `split_points`.
    ///
    /// The returned [`Range`] can be extracted using [`PeakFitArgs::slice`]
    pub fn split_at(&self, split_points: &[SplittingPoint]) -> Vec<Range<usize>> {
        let n = self.len();
        let mut segments = Vec::new();
        let mut last_x = self.time.first().copied().unwrap_or_default() - 1.0;
        for point in split_points {
            let start_i = self
                .time
                .iter()
                .position(|t| *t > last_x && *t <= point.minimum_time)
                .unwrap_or_default();
            let end_i = self
                .time
                .iter()
                .rposition(|t| *t > last_x && *t <= point.minimum_time)
                .unwrap_or_default();
            if start_i != end_i {
                segments.push(start_i.saturating_sub(1)..(end_i + 1).min(n));
            }
            last_x = point.minimum_time;
        }

        let i = self.time.iter().position(|t| *t > last_x).unwrap_or(n);
        if i != n {
            segments.push(i..n);
        }
        segments
    }

    pub fn get(&self, index: usize) -> (f64, f32) {
        (self.time[index], self.intensity[index])
    }

    /// Select a sub-region of the signal given by `iv` like those returned by [`PeakFitArgs::split_at`]
    ///
    /// The returned instance will borrow the data from `self`
    pub fn slice(&'e self, iv: Range<usize>) -> PeakFitArgs<'c, 'd> {
        let x = &self.time[iv.clone()];
        let y = &self.intensity[iv.clone()];
        (x, y).into()
    }

    pub fn subtract(&mut self, intensities: &[f64]) {
        assert_eq!(self.intensity.len(), intensities.len());

        self.intensity
            .to_mut()
            .iter_mut()
            .zip(intensities.iter())
            .for_each(|(a, b)| {
                *a -= (*b) as f32;
            });
    }

    pub fn subtract_region(&mut self, time: Cow<'a, [f64]>, intensities: &[f64]) {
        let t_first = *time.first().unwrap();
        let i_first = self.find_time(t_first);
        self.intensity
            .to_mut()
            .iter_mut()
            .skip(i_first)
            .zip(intensities.iter())
            .for_each(|(a, b)| {
                *a -= (*b) as f32;
            });
    }

    /// Find the index nearest to `time`
    pub fn find_time(&self, time: f64) -> usize {
        let time_array = &self.time;
        let n = time_array.len().saturating_sub(1);
        let mut j = match time_array.binary_search_by(|x| x.partial_cmp(&time).unwrap()) {
            Ok(i) => i.min(n),
            Err(i) => i.min(n),
        };

        let i = j;
        let mut best = j;
        let err = (time_array[j] - time).abs();
        let mut best_err = err;
        let n = n + 1;
        // search backwards
        while j > 0 && j < n {
            let err = (time_array[j] - time).abs();
            if err < best_err {
                best_err = err;
                best = j;
            } else if err > best_err {
                break;
            }
            j -= 1;
        }
        j = i;
        // search forwards
        while j < n {
            let err = (time_array[j] - time).abs();
            if err < best_err {
                best_err = err;
                best = j;
            } else if err > best_err {
                break;
            }
            j += 1;
        }
        best
    }

    /// Integrate the area under the signal for this data using trapezoid integration
    pub fn integrate(&self) -> f32 {
        trapz(&self.time, &self.intensity)
    }

    /// Compute the mean over [`Self::time`] weighted by [`Self::intensity`]
    pub fn weighted_mean_time(&self) -> f64 {
        self.iter()
            .map(|(x, y)| ((x * y), y))
            .reduce(|(xa, ya), (x, y)| ((xa + x), (ya + y)))
            .map(|(x, y)| x / y)
            .unwrap_or_default()
    }

    /// Find the index where [`Self::intensity`] achieves its maximum value
    pub fn argmax(&self) -> usize {
        let mut ymax = 0.0;
        let mut ymax_i = 0;
        for (i, (_, y)) in self.iter().enumerate() {
            if y > ymax {
                ymax = y;
                ymax_i = i;
            }
        }
        ymax_i
    }

    /// The length of the arrays
    pub fn len(&self) -> usize {
        self.time.len()
    }

    /// Create a new [`PeakFitArgs`] from this one that borrows its data from this one
    pub fn borrow(&'e self) -> PeakFitArgs<'c, 'd> {
        let is_time_owned = matches!(self.time, Cow::Owned(_));
        let is_intensity_owned = matches!(self.intensity, Cow::Owned(_));
        let time = if is_time_owned {
            Cow::Borrowed(self.time.deref())
        } else {
            self.time.clone()
        };

        let intensity = if is_intensity_owned {
            Cow::Borrowed(self.intensity.deref())
        } else {
            self.intensity.clone()
        };
        Self::new(time, intensity)
    }

    /// Compute the "null model" residuals $`\sum_i(y_i - \bar{y})^2`$
    pub fn null_residuals(&self) -> f64 {
        let mean = self.intensity.iter().sum::<f32>() as f64 / self.len() as f64;
        self.intensity
            .iter()
            .map(|y| (*y as f64 - mean).powi(2))
            .sum()
    }

    /// Compute the simple linear model residuals $`\sum_i{(y_i - (\beta_{1}x_{i} + \beta_0))^2}`$
    pub fn linear_residuals(&self) -> f64 {
        let (xsum, ysum) = self
            .iter()
            .reduce(|(xa, ya), (x, y)| (xa + x, ya + y))
            .unwrap_or_default();

        let xmean = xsum / self.len() as f64;
        let ymean = ysum / self.len() as f64;

        let mut tss = 0.0;
        let mut hat = 0.0;

        for (x, y) in self.iter() {
            let delta_x = x - xmean;
            tss += delta_x.powi(2);
            hat += delta_x * (y - ymean);
        }

        let beta = hat / tss;
        let alpha = ymean - beta * xmean;

        self.iter()
            .map(|(x, y)| (y - ((x * beta) + alpha)).powi(2))
            .sum()
    }

    /// Create a [`PeakFitArgsIter`] which is a copying iterator that casts the intensity
    /// value to `f64` for convenience in model fitting.
    pub fn iter(&self) -> PeakFitArgsIter<'_> {
        PeakFitArgsIter::new(
            self.time
                .iter()
                .copied()
                .zip(self.intensity.iter().copied()),
        )
    }

    /// Borrow this data as an [`ArrayPairSplit`]
    pub fn as_array_pair(&'e self) -> ArrayPairSplit<'a, 'b> {
        let this = self.borrow();
        ArrayPairSplit::new(this.time, this.intensity)
    }
}

impl<'a, 'b> From<(Cow<'a, [f64]>, Cow<'b, [f32]>)> for PeakFitArgs<'a, 'b> {
    fn from(pair: (Cow<'a, [f64]>, Cow<'b, [f32]>)) -> PeakFitArgs<'a, 'b> {
        PeakFitArgs::new(pair.0, pair.1)
    }
}

impl<'a, 'b> From<(&'a [f64], &'b [f32])> for PeakFitArgs<'a, 'b> {
    fn from(pair: (&'a [f64], &'b [f32])) -> PeakFitArgs<'a, 'b> {
        PeakFitArgs::new(Cow::Borrowed(pair.0), Cow::Borrowed(pair.1))
    }
}

impl From<(Vec<f64>, Vec<f32>)> for PeakFitArgs<'static, 'static> {
    fn from(pair: (Vec<f64>, Vec<f32>)) -> PeakFitArgs<'static, 'static> {
        let mz_array = Cow::Owned(pair.0);
        let intensity_array = Cow::Owned(pair.1);
        PeakFitArgs::new(mz_array, intensity_array)
    }
}

impl<'a, 'b> From<PeakFitArgs<'a, 'b>> for ArrayPairSplit<'a, 'b> {
    fn from(value: PeakFitArgs<'a, 'b>) -> Self {
        (value.time, value.intensity).into()
    }
}

impl<'a, X, Y> From<&'a mzpeaks::feature::Feature<X, Y>> for PeakFitArgs<'a, 'a> {
    fn from(value: &'a mzpeaks::feature::Feature<X, Y>) -> Self {
        Self::from((value.time_view(), value.intensity_view()))
    }
}

impl<'a, X, Y> From<&'a mzpeaks::feature::FeatureView<'a, X, Y>> for PeakFitArgs<'a, 'a> {
    fn from(value: &'a mzpeaks::feature::FeatureView<X, Y>) -> Self {
        Self::from((value.time_view(), value.intensity_view()))
    }
}

impl<'a, X, Y> From<&'a mzpeaks::feature::SimpleFeature<X, Y>> for PeakFitArgs<'a, 'a> {
    fn from(value: &'a mzpeaks::feature::SimpleFeature<X, Y>) -> Self {
        Self::from((value.time_view(), value.intensity_view()))
    }
}

impl<'a, X, Y> From<&'a mzpeaks::feature::SimpleFeatureView<'a, X, Y>> for PeakFitArgs<'a, 'a> {
    fn from(value: &'a mzpeaks::feature::SimpleFeatureView<X, Y>) -> Self {
        Self::from((value.time_view(), value.intensity_view()))
    }
}

impl<'a, X, Y> From<&'a mzpeaks::feature::ChargedFeature<X, Y>> for PeakFitArgs<'a, 'a> {
    fn from(value: &'a mzpeaks::feature::ChargedFeature<X, Y>) -> Self {
        value.as_inner().0.into()
    }
}

impl<'a, X, Y> From<&'a mzpeaks::feature::ChargedFeatureView<'a, X, Y>> for PeakFitArgs<'a, 'a> {
    fn from(value: &'a mzpeaks::feature::ChargedFeatureView<X, Y>) -> Self {
        value.as_inner().0.into()
    }
}
