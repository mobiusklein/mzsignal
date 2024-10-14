//! Fitting methods for elution-over-time profile peak shape models.
//!
//! ![A skewed gaussian peak shape fit to an asymmetric profile][peak_fit]
//!
//! This covers multiple peak shape kinds and has some support
//! for multi-modal profiles.
//!
//! The supported peak shape types:
//! - [`GaussianPeakShape`]
//! - [`SkewedGaussianPeakShape`]
//! - [`BiGaussianPeakShape`]
//!
//! and the [`PeakShape`] type that can be used when dealing with
//!
//! Most of the fitting methods expect to work with [`PeakFitArgs`] which
//! can be created from borrowed signal-over-time data.
//!
//! # Example
//!
//! ```rust
//! # use mzsignal::text::load_feature_table;
//! use mzpeaks::feature::Feature;
//! use mzsignal::feature_statistics::{PeakFitArgs, SplittingPeakShapeFitter, FitConfig};
//!
//! # fn main() {
//! let features: Vec<Feature<_, _>> = load_feature_table("test/data/features_graph.txt").unwrap();
//! let feature = &features[10979];
//! let args = PeakFitArgs::from(feature);
//! let mut fitter = SplittingPeakShapeFitter::new(args);
//! fitter.fit_with(FitConfig::default().max_iter(10_000).smooth(1));
//! let z = fitter.score();
//! eprintln!("Score: {z}");
//! # }
//! ```
//!
//! # Model Fit Evaluation
//!
//! All peak shape models are optimized using a mean squared error (MSE) loss function, regularizing over the
//! position and shape parameters but not the amplitude parameter. Parameters are updated using a basic gradient
//! descent procedure.
//!
//! A peak shape fit isn't just about minimizing the residual error, it's about there actually being a peak, so
//! for downstream applications, we provide a [`PeakShapeModel::score`] method which compares the MSE of the model
//! to a straight line linear model of the form $`y = \alpha + \beta\times x`$. Prior work has shown this approach
//! can be more effective at distinguishing jagged noise regions where a peak shape can *fit*, but isn't meaningful.
//!
//!
//!
//! [peak_fit]: https://github.com/mobiusklein/mzsignal/blob/feature/argmin_shape_fit/doc/chromatogram.png?raw=true
use std::{
    borrow::Cow,
    f64::consts::{PI, SQRT_2},
    fmt::Debug,
    iter::FusedIterator,
    ops::{Deref, Range},
};

use libm::erf;

use mzpeaks::prelude::TimeArray;

use crate::arrayops::{trapz, ArrayPair, ArrayPairSplit};

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
                segments.push(start_i..(end_i + 1).min(n));
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

/// Fit peak shapes on implementing types
pub trait FitPeaksOn<'a>
where
    PeakFitArgs<'a, 'a>: From<&'a Self>,
    Self: 'a,
{
    fn as_peak_shape_args(&'a self) -> PeakFitArgs<'a, 'a> {
        let data: PeakFitArgs<'a, 'a> = PeakFitArgs::from(self);
        data
    }

    /// Fit multiple peak models on this signal.
    fn fit_peaks_with(&'a self, config: FitConfig) -> SplittingPeakShapeFitter<'a, 'a> {
        let data: PeakFitArgs<'a, 'a> = self.as_peak_shape_args();
        let mut model = SplittingPeakShapeFitter::new(data);
        model.fit_with(config);
        model
    }
}

impl<'a, X: 'a, Y: 'a> FitPeaksOn<'a> for mzpeaks::feature::Feature<X, Y> {}
impl<'a, X: 'a, Y: 'a> FitPeaksOn<'a> for mzpeaks::feature::FeatureView<'a, X, Y> {}
impl<'a, X: 'a, Y: 'a> FitPeaksOn<'a> for mzpeaks::feature::SimpleFeature<X, Y> {}
impl<'a, X: 'a, Y: 'a> FitPeaksOn<'a> for mzpeaks::feature::SimpleFeatureView<'a, X, Y> {}
impl<'a, X: 'a, Y: 'a> FitPeaksOn<'a> for mzpeaks::feature::ChargedFeature<X, Y> {}
impl<'a, X: 'a, Y: 'a> FitPeaksOn<'a> for mzpeaks::feature::ChargedFeatureView<'a, X, Y> {}

/// Hyperparameters for fitting a peak shape model
#[derive(Debug, Clone)]
pub struct FitConfig {
    /// The maximum number of iterations to attempt when fitting a peak model
    max_iter: usize,
    /// The rate at which model parameters are updated
    learning_rate: f64,
    /// The minimum distance between the current loss and the previous loss at which to decide the model
    /// has converged
    convergence: f64,
    /// How much smoothing to perform before fitting a peak model.
    ///
    /// See [`PeakFitArgs::smooth`]
    smooth: usize,
}

impl FitConfig {
    /// The maximum number of iterations to attempt when fitting a peak model
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// The rate at which model parameters are updated
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// The minimum distance between the current loss and the previous loss at which to decide the model
    /// has converged
    pub fn convergence(mut self, convergence: f64) -> Self {
        self.convergence = convergence;
        self
    }

    /// How much smoothing to perform before fitting a peak model.
    ///
    /// See [`PeakFitArgs::smooth`]
    pub fn smooth(mut self, smooth: usize) -> Self {
        self.smooth = smooth;
        self
    }
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            max_iter: 50_000,
            learning_rate: 1e-3,
            convergence: 1e-9,
            smooth: 0,
        }
    }
}

/// Describe a model fitting procedure's output
#[derive(Debug, Default, Clone, Copy)]
pub struct ModelFitResult {
    /// The loss at the end of the optimization run
    pub loss: f64,
    /// The number of iterations run
    pub iterations: usize,
    /// Whether or not the model converged within the specified number of iterations
    pub converged: bool,
    /// Whether or not the model was able to fit *at all*
    pub success: bool,
}

impl ModelFitResult {
    pub fn new(loss: f64, iterations: usize, converged: bool, success: bool) -> Self {
        Self {
            loss,
            iterations,
            converged,
            success,
        }
    }
}

/// A set of peak shape model fitting behaviors that interacts with the [`PeakShapeModel`]
/// trait associated with some peak signal data.
pub trait PeakShapeModelFitter<'a, 'b> {
    /// The [`PeakShapeModel`] that this type will fit.
    type ModelType: PeakShapeModel + Debug;

    /// Construct a new [`PeakShapeModelFitter`] from [`PeakFitArgs`]
    fn from_args(args: PeakFitArgs<'a, 'b>) -> Self;

    /// Compute the model gradient against the enclosed data
    ///
    /// # See also
    /// [`PeakShapeModel::gradient`]
    fn gradient(&self, params: &Self::ModelType) -> Self::ModelType;

    /// Compute the model loss function the enclosed data
    ///
    /// # See also
    /// [`PeakShapeModel::loss`]
    fn loss(&self, params: &Self::ModelType) -> f64;

    /// Borrow the enclosed data
    fn data(&self) -> &PeakFitArgs;

    /// Iterate over the enclosed data
    fn iter(&self) -> PeakFitArgsIter {
        self.data().iter()
    }

    /// Compute the model score against the enclosed data
    ///
    /// # See also
    /// [`PeakShapeModel::score`]
    fn score(&self, model_params: &Self::ModelType) -> f64 {
        model_params.score(self.data())
    }

    /// Do the actual model fitting on the enclosed data.
    fn fit_model(
        &mut self,
        model_params: &mut Self::ModelType,
        config: FitConfig,
    ) -> ModelFitResult;
}

/// A model of an elution profile peak shape that can be estimated using gradient descent
pub trait PeakShapeModel: Clone {
    type Fitter<'a, 'b>: PeakShapeModelFitter<'a, 'b, ModelType = Self>;

    /// Compute the theoretical intensity at a specified coordinate
    ///
    /// # See also
    /// [`PeakShapeModel::predict`]
    /// [`PeakShapeModel::predict_iter`]
    fn density(&self, x: f64) -> f64;

    /// Update the parameters of the model based upon the `gradient` and a
    /// given learning rate.
    fn gradient_update(&mut self, gradient: Self, learning_rate: f64);

    /// Given a coordinate sequence, produce the complementary sequence of theoretical intensities
    ///
    /// # See also
    /// [`PeakShapeModel::density`]
    /// [`PeakShapeModel::predict_iter`]
    fn predict(&self, times: &[f64]) -> Vec<f64> {
        times.iter().map(|t| self.density(*t)).collect()
    }

    /// Given a coordinate iterator, produce the complementary iterator of theoretical intensities
    ///
    /// # See also
    /// [`PeakShapeModel::density`]
    /// [`PeakShapeModel::predict`]
    fn predict_iter<I: IntoIterator<Item = f64>>(&self, times: I) -> impl Iterator<Item = f64> {
        times.into_iter().map(|t| self.density(t))
    }

    /// Compute the gradient of the loss function for parameter optimization.
    fn gradient(&self, data: &PeakFitArgs) -> Self;

    /// Compute the loss function for optimization, mean-squared error
    fn loss(&self, data: &PeakFitArgs) -> f64 {
        data.iter()
            .map(|(t, i)| (i - self.density(t)).powi(2))
            .sum::<f64>()
            / data.len() as f64
    }

    /// Compute the difference between the observed signal and the theoretical signal,
    /// clamping the value to be non-negative
    fn residuals<'a, 'b, 'e: 'a + 'b>(&self, data: &'e PeakFitArgs<'a, 'b>) -> PeakFitArgs<'a, 'b> {
        let mut data = data.borrow();
        for (yhat, y) in self
            .predict_iter(data.time.iter().copied())
            .zip(data.intensity.to_mut().iter_mut())
        {
            *y -= yhat as f32;
            if *y < 0.0 {
                *y = 0.0;
            }
        }
        data
    }

    /// Compute the 1 - ratio of the peak shape model squared error to
    /// a straight line linear model squared error.
    ///
    /// This value is 0 when the ordinary linear model is much better than the peak
    /// shape model, and approaches 1.0 when the peak shape model is a much better fit
    /// of the data than straight line model.
    ///
    /// *NOTE*: The function output is clamped to the $`[0, 1]`$ range for consistency
    fn score(&self, data: &PeakFitArgs) -> f64 {
        let linear_resid = data.linear_residuals();
        let mut shape_resid = 0.0;

        for (x, y) in data.iter() {
            shape_resid += (y - self.density(x)).powi(2);
        }

        let line_test = shape_resid
            / (if linear_resid > 0.0 {
                linear_resid
            } else {
                1.0
            });
        (1.0 - line_test.max(1e-5)).max(0.0).min(1.0)
    }

    /// Given observed data, compute some initial parameters.
    ///
    /// This is the preferred means of producing an initial model
    /// for some data, prior to fitting.
    fn guess(data: &PeakFitArgs) -> Self;

    /// Fit the peak shape model to some data using the default
    /// [`FitConfig`] settings.
    fn fit(&mut self, data: PeakFitArgs) -> ModelFitResult {
        self.fit_with(data, Default::default())
    }

    /// Fit the peak shape model to some data using `config` options
    fn fit_with(&mut self, args: PeakFitArgs, config: FitConfig) -> ModelFitResult {
        let mut fitter = Self::Fitter::from_args(args);
        fitter.fit_model(self, config)
    }
}

/// Gaussian peak shape model
///
/// ```math
/// y = a\exp\left({\frac{-(\mu - x)^2}{2\sigma^2}}\right)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct GaussianPeakShape {
    pub mu: f64,
    pub sigma: f64,
    pub amplitude: f64,
}

impl GaussianPeakShape {
    pub fn new(mu: f64, sigma: f64, amplitude: f64) -> Self {
        Self {
            mu,
            sigma,
            amplitude,
        }
    }

    /// Given observed data, compute some initial parameters
    pub fn guess(data: &PeakFitArgs) -> Self {
        if data.len() == 0 {
            return Self::new(1.0, 1.0, 1.0);
        }
        let idx = data.argmax();
        let mu = data.time[idx];
        let amplitude = data.intensity[idx] as f64;
        let sigma = 1.0;
        Self::new(mu, sigma, amplitude)
    }

    /// Compute the regularization term for the loss function
    pub fn regularization(&self) -> f64 {
        self.mu + self.sigma
    }

    /// Compute the loss function for optimization, mean-squared error
    pub fn loss(&self, data: &PeakFitArgs) -> f64 {
        data.iter()
            .map(|(t, i)| (i - self.density(t)).powi(2))
            .sum::<f64>()
            / data.len() as f64
            + self.regularization()
    }

    pub fn density(&self, x: f64) -> f64 {
        self.amplitude * (-0.5 * (x - self.mu).powi(2) / self.sigma.powi(2)).exp()
    }

    pub fn gradient(&self, data: &PeakFitArgs) -> Self {
        let amp = self.amplitude;
        let mu = self.mu;
        let sigma = self.sigma;

        let two_mu = 2.0 * mu;
        let sigma_squared = sigma.powi(2);
        let sigma_cubed = sigma.powi(3);
        let sigma_squared_inv = 1.0 / sigma_squared;

        let mut gradient_mu = 0.0;
        let mut gradient_sigma = 0.0;
        let mut gradient_amplitude = 0.0;

        for (x, y) in data.iter() {
            let mu_sub_x_squared = (-mu + x).powi(2);
            let half_mu_sub_x_squared_div_sigma_squared =
                -0.5 * mu_sub_x_squared * sigma_squared_inv;
            let half_mu_sub_x_squared_div_sigma_squared_exp =
                half_mu_sub_x_squared_div_sigma_squared.exp();

            let delta_y = -amp * half_mu_sub_x_squared_div_sigma_squared_exp + y;

            let delta_y_half_mu_sub_x_squared_div_sigma_squared_exp =
                delta_y * half_mu_sub_x_squared_div_sigma_squared_exp;

            gradient_mu += amp
                * (two_mu - 2.0 * x)
                * delta_y_half_mu_sub_x_squared_div_sigma_squared_exp
                * sigma_squared_inv
                + 1.0;

            gradient_sigma +=
                -2.0 * amp * mu_sub_x_squared * delta_y_half_mu_sub_x_squared_div_sigma_squared_exp
                    / sigma_cubed
                    + 1.0;

            gradient_amplitude += -2.0 * delta_y_half_mu_sub_x_squared_div_sigma_squared_exp;
        }

        let n = data.len() as f64;

        Self::new(gradient_mu / n, gradient_sigma / n, gradient_amplitude / n).gradient_norm()
    }

    fn gradient_norm(&self) -> Self {
        let mut g = [self.mu, self.sigma, self.amplitude];
        let gradnorm: f64 = g.iter().map(|f| f.abs()).sum::<f64>() / g.len() as f64;
        if gradnorm > 1.0 {
            g[0] /= gradnorm;
            g[1] /= gradnorm;
        }

        Self::new(g[0], g[1], g[2])
    }

    /// Compute the gradient w.r.t. $`\mu`$
    ///
    /// ```math
    /// -\frac{a \left(2 \mu - 2 x\right) \left(- a e^{- \frac{\left(- \mu + x\right)^{2}}{2 \sigma^{2}}} + y\right) e^{- \frac{\left(- \mu + x\right)^{2}}{2 \sigma^{2}}}}{\sigma^{2}} + 1
    /// ```
    fn mu_gradient(&self, data: &PeakFitArgs) -> f64 {
        let amp = self.amplitude;
        let mu = self.mu;
        let sigma = self.sigma;

        let two_mu = 2.0 * mu;
        let sigma_squared = sigma.powi(2);
        let sigma_squared_inv = 1.0 / sigma_squared;

        let grad: f64 = data
            .iter()
            .map(|(x, y)| {
                let mu_sub_x_squared = (-mu + x).powi(2);
                let half_mu_sub_x_squared_div_sigma_squared =
                    -0.5 * mu_sub_x_squared * sigma_squared_inv;
                let half_mu_sub_x_squared_div_sigma_squared_exp =
                    half_mu_sub_x_squared_div_sigma_squared.exp();

                amp * (two_mu - 2.0 * x)
                    * (-amp * half_mu_sub_x_squared_div_sigma_squared_exp + y)
                    * half_mu_sub_x_squared_div_sigma_squared_exp
                    * sigma_squared_inv
                    + 1.0
            })
            .sum();

        grad / data.len() as f64
    }

    /// Compute the gradient w.r.t. $`\sigma`$
    ///
    /// ```math
    /// - \frac{2 a \left(- \mu + x\right)^{2} \left(- a e^{- \frac{\left(- \mu + x\right)^{2}}{2 \sigma^{2}}} + y\right) e^{- \frac{\left(- \mu + x\right)^{2}}{2 \sigma^{2}}}}{\sigma^{3}} + 1
    /// ```
    fn sigma_gradient(&self, data: &PeakFitArgs) -> f64 {
        let amp = self.amplitude;
        let mu = self.mu;
        let sigma = self.sigma;

        let sigma_squared = sigma.powi(2);
        let sigma_cubed = sigma.powi(3);

        let grad: f64 = data
            .iter()
            .map(|(x, y)| {
                let mu_sub_x_squared = (-mu + x).powi(2);
                let half_mu_sub_x_squared_div_sigma_squared =
                    -0.5 * mu_sub_x_squared / sigma_squared;
                let half_mu_sub_x_squared_div_sigma_squared_exp =
                    half_mu_sub_x_squared_div_sigma_squared.exp();
                -2.0 * amp
                    * mu_sub_x_squared
                    * (-amp * half_mu_sub_x_squared_div_sigma_squared_exp + y)
                    * half_mu_sub_x_squared_div_sigma_squared_exp
                    / sigma_cubed
                    + 1.0
            })
            .sum();

        grad / data.len() as f64
    }

    /// Compute the gradient w.r.t. amplitude $`a`$
    ///
    /// ```math
    /// - 2 \left(- a e^{- \frac{\left(- \mu + x\right)^{2}}{2 \sigma^{2}}} + y\right) e^{- \frac{\left(- \mu + x\right)^{2}}{2 \sigma^{2}}}
    /// ```
    fn amplitude_gradient(&self, data: &PeakFitArgs) -> f64 {
        let amp = self.amplitude;
        let mu = self.mu;
        let sigma = self.sigma;

        let sigma_squared = sigma.powi(2);

        let grad: f64 = data
            .iter()
            .map(|(x, y)| {
                let mu_sub_x_squared = (-mu + x).powi(2);
                let half_mu_sub_x_squared_div_sigma_squared =
                    -0.5 * mu_sub_x_squared / sigma_squared;
                let half_mu_sub_x_squared_div_sigma_squared_exp =
                    half_mu_sub_x_squared_div_sigma_squared.exp();
                -2f64
                    * (-amp * half_mu_sub_x_squared_div_sigma_squared_exp + y)
                    * half_mu_sub_x_squared_div_sigma_squared_exp
            })
            .sum();

        grad / data.len() as f64
    }

    /// A non-optimized version of the gradient calculation used for testing
    /// correctness
    pub fn gradient_split(&self, data: &PeakFitArgs) -> Self {
        Self::new(
            self.mu_gradient(&data),
            self.sigma_gradient(&data),
            self.amplitude_gradient(&data),
        )
        .gradient_norm()
    }

    /// Update the parameters of the model based upon the `gradient` and a
    /// given learning rate.
    pub fn gradient_update(&mut self, gradient: Self, learning_rate: f64) {
        self.mu -= gradient.mu * learning_rate;
        self.sigma -= gradient.sigma * learning_rate;
        self.amplitude -= gradient.amplitude * learning_rate;
    }
}

impl PeakShapeModel for GaussianPeakShape {
    type Fitter<'a, 'b> = PeakShapeFitter<'a, 'b, Self>;

    fn density(&self, x: f64) -> f64 {
        self.density(x)
    }

    fn gradient_update(&mut self, gradient: Self, learning_rate: f64) {
        self.gradient_update(gradient, learning_rate);
    }

    fn guess(args: &PeakFitArgs) -> Self {
        Self::guess(args)
    }

    fn gradient(&self, data: &PeakFitArgs) -> Self {
        self.gradient(data)
    }

    fn loss(&self, data: &PeakFitArgs) -> f64 {
        data.iter()
            .map(|(t, i)| (i - self.density(t)).powi(2))
            .sum::<f64>()
            / data.len() as f64
            + self.regularization()
    }
}

/// Skewed Gaussian peak shape model
///
/// ```math
/// y = a\left(\text{erf}\left({\sqrt{2} \lambda\frac{\mu - x}{2\sigma}}\right) + 1\right)\exp\left(-\frac{(\mu-x)^2}{2\sigma^2}\right)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct SkewedGaussianPeakShape {
    pub mu: f64,
    pub sigma: f64,
    pub amplitude: f64,
    pub lambda: f64,
}

impl SkewedGaussianPeakShape {
    pub fn new(mu: f64, sigma: f64, amplitude: f64, lambda: f64) -> Self {
        Self {
            mu,
            sigma,
            amplitude,
            lambda,
        }
    }

    /// Given observed data, compute some initial parameters
    pub fn guess(data: &PeakFitArgs) -> Self {
        if data.len() == 0 {
            return Self::new(1.0, 1.0, 1.0, 1.0);
        }
        let idx = data.argmax();
        let mu = data.time[idx];
        let amplitude = data.intensity[idx] as f64;
        let sigma = 1.0;
        let lambda = 1.0;
        Self::new(mu, sigma, amplitude, lambda)
    }

    /// Compute the theoretical intensity at a specified coordinate
    pub fn density(&self, x: f64) -> f64 {
        self.amplitude
            * (erf(SQRT_2 * self.lambda * (-self.mu + x) / (2.0 * self.sigma)) + 1.0)
            * (-0.5 * (-self.mu + x).powi(2) / self.sigma.powi(2)).exp()
    }

    /// Compute the regularization term for the loss function
    pub fn regularization(&self) -> f64 {
        self.mu + self.sigma + self.lambda
    }

    /// A non-optimized version of the gradient calculation used for testing
    /// correctness
    pub fn gradient_split(&self, data: &PeakFitArgs) -> Self {
        Self::new(
            self.mu_gradient(&data),
            self.sigma_gradient(&data),
            self.amplitude_gradient(&data),
            self.lambda_gradient(&data),
        )
        .gradient_norm()
    }

    fn gradient_norm(&self) -> Self {
        let mut g = [self.mu, self.sigma, self.amplitude, self.lambda];
        let gradnorm: f64 = g.iter().map(|f| f.abs()).sum::<f64>() / g.len() as f64;
        if gradnorm > 1.0 {
            g[0] /= gradnorm;
            g[1] /= gradnorm;
            g[3] /= gradnorm;
        }

        SkewedGaussianPeakShape::new(g[0], g[1], g[2], g[3])
    }

    /// Compute the gradient of the loss function for parameter optimization.
    pub fn gradient(&self, data: &PeakFitArgs) -> Self {
        let amp = self.amplitude;
        let mu = self.mu;
        let sigma = self.sigma;
        let lam = self.lambda;

        let two_sigma = sigma * 2.0;
        let sigma_square = sigma.powi(2);
        let sigma_cubed = sigma.powi(3);
        let skew = 2.0 * 1.4142135623731 * amp * lam;
        let delta_skew = -2.0 * 1.4142135623731 * amp;
        let sqrt_2_lam = SQRT_2 * lam;
        let sqrt_pi_sigma = PI.sqrt() * sigma;
        let sqrt_pi_sigma_square = PI.sqrt() * sigma_square;
        let neg_half_lam_squared = -1_f64 / 2.0 * lam.powi(2);

        let mut gradient_mu = 0.0;
        let mut gradient_sigma = 0.0;
        let mut gradient_lambda = 0.0;
        let mut gradient_amplitude = 0.0;

        for (x, y) in data.iter() {
            let mu_sub_x = -mu + x;
            let mu_sub_x_squared = mu_sub_x.powi(2);
            let neg_half_mu_sub_x_squared_div_sigma_squared_exp =
                (-0.5 * mu_sub_x_squared / sigma_square).exp();
            let erf_sqrt_2_lam_mu_sub_x_div_two_sigma_plus_one =
                erf(sqrt_2_lam * mu_sub_x / two_sigma) + 1.0;
            let neg_half_lam_squared_mu_sub_x_squared_div_sigma_square_exp =
                (neg_half_lam_squared * mu_sub_x_squared / sigma_square).exp();

            let delta_y = -amp
                * erf_sqrt_2_lam_mu_sub_x_div_two_sigma_plus_one
                * neg_half_mu_sub_x_squared_div_sigma_squared_exp
                + y;

            gradient_mu += delta_y
                * (skew
                    * neg_half_mu_sub_x_squared_div_sigma_squared_exp
                    * neg_half_lam_squared_mu_sub_x_squared_div_sigma_square_exp
                    / sqrt_pi_sigma
                    + amp
                        * (2.0 * mu - 2.0 * x)
                        * erf_sqrt_2_lam_mu_sub_x_div_two_sigma_plus_one
                        * neg_half_mu_sub_x_squared_div_sigma_squared_exp
                        / sigma_square)
                + 1.0;

            gradient_sigma += delta_y
                * (skew
                    * mu_sub_x
                    * neg_half_mu_sub_x_squared_div_sigma_squared_exp
                    * neg_half_lam_squared_mu_sub_x_squared_div_sigma_square_exp
                    / (sqrt_pi_sigma_square)
                    - 2.0
                        * amp
                        * mu_sub_x_squared
                        * erf_sqrt_2_lam_mu_sub_x_div_two_sigma_plus_one
                        * neg_half_mu_sub_x_squared_div_sigma_squared_exp
                        / sigma_cubed)
                + 1.0;

            gradient_lambda += delta_skew
                * mu_sub_x
                * delta_y
                * neg_half_mu_sub_x_squared_div_sigma_squared_exp
                * neg_half_lam_squared_mu_sub_x_squared_div_sigma_square_exp
                / sqrt_pi_sigma
                + 1.0;

            gradient_amplitude += -2.0
                * delta_y
                * erf_sqrt_2_lam_mu_sub_x_div_two_sigma_plus_one
                * neg_half_mu_sub_x_squared_div_sigma_squared_exp
        }

        let n = data.len() as f64;

        Self::new(
            gradient_mu / n,
            gradient_sigma / n,
            gradient_amplitude / n,
            gradient_lambda / n,
        )
        .gradient_norm()
    }

    fn mu_gradient(&self, data: &PeakFitArgs) -> f64 {
        let amp = self.amplitude;
        let mu = self.mu;
        let sigma = self.sigma;
        let lam = self.lambda;

        let two_sigma = sigma * 2.0;
        let sigma_square = sigma.powi(2);
        let skew = 2.0 * 1.4142135623731 * amp * lam;
        let sqrt_2_lam = SQRT_2 * lam;
        let sqrt_pi_sigma = PI.sqrt() * sigma;
        let neg_half_lam_squared = -1_f64 / 2.0 * lam.powi(2);

        let mut grad = 0.0;
        for (x, y) in data.iter() {
            grad += (-amp
                * (erf(sqrt_2_lam * (-mu + x) / (two_sigma)) + 1.0)
                * (-0.5 * (-mu + x).powi(2) / sigma_square).exp()
                + y)
                * (skew
                    * (-0.5 * (-mu + x).powi(2) / sigma_square).exp()
                    * (neg_half_lam_squared * (-mu + x).powi(2) / sigma_square).exp()
                    / sqrt_pi_sigma
                    + amp
                        * (2.0 * mu - 2.0 * x)
                        * (erf(sqrt_2_lam * (-mu + x) / (two_sigma)) + 1.0)
                        * (-0.5 * (-mu + x).powi(2) / sigma_square).exp()
                        / sigma_square)
                + 1.0
        }
        grad / data.len() as f64
    }

    fn sigma_gradient(&self, data: &PeakFitArgs) -> f64 {
        let amp = self.amplitude;
        let mu = self.mu;
        let sigma = self.sigma;
        let lam = self.lambda;

        let two_sigma = sigma * 2.0;
        let sigma_square = sigma.powi(2);
        let sigma_cubed = sigma.powi(3);
        let skew = 2.0 * 1.4142135623731 * amp * lam;
        let sqrt_2_lam = SQRT_2 * lam;
        let sqrt_pi_sigma_square = PI.sqrt() * sigma_square;
        let neg_half_lam_squared = -1_f64 / 2.0 * lam.powi(2);

        let mut grad = 0.0;
        for (x, y) in data.iter() {
            grad += (-amp
                * (erf(sqrt_2_lam * (-mu + x) / two_sigma) + 1.0)
                * (-0.5 * (-mu + x).powi(2) / sigma_square).exp()
                + y)
                * (skew
                    * (-mu + x)
                    * (-0.5 * (-mu + x).powi(2) / sigma_square).exp()
                    * (neg_half_lam_squared * (-mu + x).powi(2) / sigma_square).exp()
                    / (sqrt_pi_sigma_square)
                    - 2.0
                        * amp
                        * (-mu + x).powi(2)
                        * (erf(sqrt_2_lam * (-mu + x) / two_sigma) + 1.0)
                        * (-0.5 * (-mu + x).powi(2) / sigma_square).exp()
                        / sigma_cubed)
                + 1.0
        }
        grad / data.len() as f64
    }

    fn amplitude_gradient(&self, data: &PeakFitArgs) -> f64 {
        let amp = self.amplitude;
        let mu = self.mu;
        let sigma = self.sigma;
        let lam = self.lambda;

        let two_sigma = sigma * 2.0;
        let sigma_square = sigma.powi(2);
        let sqrt_2_lam = SQRT_2 * lam;

        let mut grad: f64 = 0.0;
        for (x, y) in data.iter() {
            grad += -2.0
                * (-amp
                    * (erf(sqrt_2_lam * (-mu + x) / (two_sigma)) + 1.0)
                    * (-0.5 * (-mu + x).powi(2) / sigma_square).exp()
                    + y)
                * (erf(sqrt_2_lam * (-mu + x) / (two_sigma)) + 1.0)
                * (-0.5 * (-mu + x).powi(2) / sigma_square).exp()
        }
        grad / data.len() as f64
    }

    fn lambda_gradient(&self, data: &PeakFitArgs) -> f64 {
        let amp = self.amplitude;
        let mu = self.mu;
        let sigma = self.sigma;
        let lam = self.lambda;

        let two_sigma = sigma * 2.0;
        let sigma_square = sigma.powi(2);
        let delta_skew = -2.0 * 1.4142135623731 * amp;
        let sqrt_2_lam = SQRT_2 * lam;
        let sqrt_pi_sigma = PI.sqrt() * sigma;
        let neg_half_lam_squared = -1_f64 / 2.0 * lam.powi(2);

        let mut grad = 0.0;
        for (x, y) in data.iter() {
            let mu_sub_x = -mu + x;
            let mu_sub_x_squared = mu_sub_x.powi(2);
            let neg_half_mu_sub_x_squared_div_sigma_squared =
                (-0.5 * mu_sub_x_squared / sigma_square).exp();

            grad += delta_skew
                * mu_sub_x
                * (-amp
                    * (erf(sqrt_2_lam * mu_sub_x / (two_sigma)) + 1.0)
                    * neg_half_mu_sub_x_squared_div_sigma_squared
                    + y)
                * neg_half_mu_sub_x_squared_div_sigma_squared
                * (neg_half_lam_squared * mu_sub_x_squared / sigma_square).exp()
                / sqrt_pi_sigma
                + 1.0
        }
        grad / data.len() as f64
    }

    /// Update the parameters of the model based upon the `gradient` and a
    /// given learning rate.
    pub fn gradient_update(&mut self, gradient: Self, learning_rate: f64) {
        self.mu -= gradient.mu * learning_rate;
        self.sigma -= gradient.sigma * learning_rate;
        self.amplitude -= gradient.amplitude * learning_rate;
        if self.amplitude < 0.0 {
            self.amplitude = 0.0
        }
        self.lambda -= gradient.lambda * learning_rate;
    }
}

impl PeakShapeModel for SkewedGaussianPeakShape {
    type Fitter<'a, 'b> = PeakShapeFitter<'a, 'b, Self>;

    fn density(&self, x: f64) -> f64 {
        self.density(x)
    }

    fn gradient_update(&mut self, gradient: Self, learning_rate: f64) {
        self.gradient_update(gradient, learning_rate);
    }

    fn guess(args: &PeakFitArgs) -> Self {
        Self::guess(args)
    }

    fn gradient(&self, data: &PeakFitArgs) -> Self {
        self.gradient(data)
    }

    /// Compute the loss function for optimization, mean-squared error
    fn loss(&self, data: &PeakFitArgs) -> f64 {
        data.iter()
            .map(|(t, i)| (i - self.density(t)).powi(2))
            .sum::<f64>()
            / data.len() as f64
            + self.regularization()
    }
}

/// Bi-Gaussian peak shape model
///
/// ```math
/// y = \begin{cases}
///     a\exp\left({\frac{-(\mu - x)^2}{2\sigma_a^2}}\right) & x \le \mu \\
///     a\exp\left({\frac{-(\mu - x)^2}{2\sigma_b^2}}\right) & x \gt x
/// \end{cases}
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct BiGaussianPeakShape {
    pub mu: f64,
    pub sigma_falling: f64,
    pub sigma_rising: f64,
    pub amplitude: f64,
}

impl BiGaussianPeakShape {
    pub fn new(mu: f64, sigma_low: f64, sigma_high: f64, amplitude: f64) -> Self {
        Self {
            mu,
            sigma_falling: sigma_low,
            sigma_rising: sigma_high,
            amplitude,
        }
    }

    /// Given observed data, compute some initial parameters
    pub fn guess(data: &PeakFitArgs) -> Self {
        if data.len() == 0 {
            return Self::new(1.0, 1.0, 1.0, 1.0);
        }
        let idx = data.argmax();
        let mu = data.time[idx];
        let amplitude = data.intensity[idx] as f64;
        let sigma = 1.0;
        Self::new(mu, sigma, sigma, amplitude)
    }

    /// Compute the theoretical intensity at a specified coordinate
    pub fn density(&self, x: f64) -> f64 {
        if self.mu >= x {
            self.amplitude * (-0.5 * (-self.mu + x).powi(2) / self.sigma_falling.powi(2)).exp()
        } else {
            self.amplitude * (-0.5 * (-self.mu + x).powi(2) / self.sigma_rising.powi(2)).exp()
        }
    }

    /// Update the parameters of the model based upon the `gradient` and a
    /// given learning rate.
    pub fn gradient_update(&mut self, gradient: Self, learning_rate: f64) {
        self.mu -= gradient.mu * learning_rate;
        self.sigma_falling -= gradient.sigma_falling * learning_rate;
        self.sigma_rising -= gradient.sigma_rising * learning_rate;

        self.amplitude -= gradient.amplitude * learning_rate;
        if self.amplitude < 0.0 {
            self.amplitude = 0.0
        }
    }

    /// Compute the regularization term for the loss function
    pub fn regularization(&self) -> f64 {
        self.mu + self.sigma_falling + self.sigma_rising
    }

    /// Compute the gradient of the loss function for parameter optimization.
    pub fn gradient(&self, data: &PeakFitArgs) -> BiGaussianPeakShape {
        let mu = self.mu;
        let amp = self.amplitude;
        let sigma_low = self.sigma_falling;
        let sigma_high = self.sigma_rising;

        let sigma_low_squared = sigma_low.powi(2);
        let sigma_high_squared = sigma_high.powi(2);
        let sigma_low_cubed = sigma_low.powi(3);
        let sigma_high_cubed = sigma_high.powi(3);
        let two_mu = mu * 2.0;
        let neg_half_amp = -0.5 * amp;

        let mut gradient_mu = 0.0;
        let mut gradient_sigma_high = 0.0;
        let mut gradient_sigma_low = 0.0;
        let mut gradient_amplitude = 0.0;

        for (x, y) in data.iter() {
            let mu_sub_x_squared = (mu - x).powi(2);
            let neg_half_mu_sub_x_squared = -0.5 * mu_sub_x_squared;

            if mu >= x {
                let neg_half_mu_sub_x_squared_div_sigma_low_squared =
                    neg_half_mu_sub_x_squared / sigma_low_squared;

                let neg_half_mu_sub_x_squared_div_sigma_low_squared_exp =
                    neg_half_mu_sub_x_squared_div_sigma_low_squared.exp();
                let delta_y =
                    -2.0 * (y - amp * neg_half_mu_sub_x_squared_div_sigma_low_squared_exp);

                gradient_mu += delta_y
                    * (neg_half_amp
                        * (two_mu - 2.0 * x)
                        * neg_half_mu_sub_x_squared_div_sigma_low_squared_exp
                        / sigma_low_squared)
                    + 1.0;

                gradient_sigma_high += 1.0;

                gradient_sigma_low += delta_y
                    * (amp
                        * mu_sub_x_squared
                        * neg_half_mu_sub_x_squared_div_sigma_low_squared_exp
                        / sigma_low_cubed)
                    + 1.0;

                gradient_amplitude += delta_y * neg_half_mu_sub_x_squared_div_sigma_low_squared_exp
            } else {
                let neg_half_mu_sub_x_squared_div_sigma_high_squared =
                    neg_half_mu_sub_x_squared / sigma_high_squared;

                let neg_half_mu_sub_x_squared_div_sigma_high_squared_exp =
                    neg_half_mu_sub_x_squared_div_sigma_high_squared.exp();

                let delta_y =
                    -2.0 * (y - amp * neg_half_mu_sub_x_squared_div_sigma_high_squared_exp);

                gradient_mu += delta_y
                    * (neg_half_amp
                        * (two_mu - 2.0 * x)
                        * neg_half_mu_sub_x_squared_div_sigma_high_squared_exp
                        / sigma_high_squared)
                    + 1.0;

                gradient_sigma_high += delta_y
                    * (amp
                        * mu_sub_x_squared
                        * neg_half_mu_sub_x_squared_div_sigma_high_squared_exp
                        / sigma_high_cubed)
                    + 1.0;

                gradient_sigma_low += 1.0;

                gradient_amplitude += delta_y * neg_half_mu_sub_x_squared_div_sigma_high_squared_exp
            }
        }

        let n = data.len() as f64;

        BiGaussianPeakShape::new(
            gradient_mu / n,
            gradient_sigma_low / n,
            gradient_sigma_high / n,
            gradient_amplitude / n,
        )
        .gradient_norm()
    }

    /// A non-optimized version of the gradient calculation used for testing
    /// correctness
    pub fn gradient_split(&self, data: &PeakFitArgs) -> BiGaussianPeakShape {
        let g = Self::new(
            self.gradient_mu(&data),
            self.gradient_sigma_falling(&data),
            self.gradient_sigma_rising(&data),
            self.gradient_amplitude(&data),
        );
        g.gradient_norm()
    }

    fn gradient_norm(&self) -> Self {
        let mut g = [self.mu, self.sigma_falling, self.sigma_rising, self.amplitude];
        let gradnorm: f64 = g.iter().map(|f| f.abs()).sum::<f64>() / g.len() as f64;
        if gradnorm > 1.0 {
            g[0] /= gradnorm;
            g[1] /= gradnorm;
            g[2] /= gradnorm;
        }
        BiGaussianPeakShape::new(g[0], g[1], g[2], g[3])
    }

    fn gradient_mu(&self, data: &PeakFitArgs) -> f64 {
        let mu = self.mu;
        let amp = self.amplitude;
        let sigma_low = self.sigma_falling;
        let sigma_high = self.sigma_rising;

        let sigma_low_squared = sigma_low.powi(2);
        let sigma_high_squared = sigma_high.powi(2);
        let neg_half_amp = -0.5 * amp;

        data.iter()
            .map(|(x, y)| {
                let mu_sub_x_squared = (mu - x).powi(2);
                let neg_half_mu_sub_x_squared = -0.5 * mu_sub_x_squared;

                if mu >= x {
                    -2.0 * (y - amp * (neg_half_mu_sub_x_squared / sigma_low_squared).exp())
                        * (neg_half_amp
                            * (2.0 * mu - 2.0 * x)
                            * (neg_half_mu_sub_x_squared / sigma_low_squared).exp()
                            / sigma_low_squared)
                        + 1.0
                } else {
                    -2.0 * (y - amp * (neg_half_mu_sub_x_squared / sigma_high_squared).exp())
                        * (neg_half_amp
                            * (2.0 * mu - 2.0 * x)
                            * (neg_half_mu_sub_x_squared / sigma_high_squared).exp()
                            / sigma_high_squared)
                        + 1.0
                }
            })
            .sum::<f64>()
            / data.len() as f64
    }

    fn gradient_sigma_rising(&self, data: &PeakFitArgs) -> f64 {
        let mu = self.mu;
        let amp = self.amplitude;
        let sigma_low = self.sigma_falling;
        let sigma_high = self.sigma_rising;

        let sigma_low_squared = sigma_low.powi(2);
        let sigma_high_squared = sigma_high.powi(2);
        let sigma_high_cubed = sigma_high.powi(3);

        data.iter()
            .map(|(x, y)| {
                -2.0 * (y - if mu >= x {
                    amp * (-0.5 * (-mu + x).powi(2) / sigma_low_squared).exp()
                } else {
                    amp * (-0.5 * (-mu + x).powi(2) / sigma_high_squared).exp()
                }) * if mu >= x {
                    0.0
                } else {
                    amp * (-mu + x).powi(2) * (-0.5 * (-mu + x).powi(2) / sigma_high_squared).exp()
                        / sigma_high_cubed
                } + 1.0
            })
            .sum::<f64>()
            / data.len() as f64
    }

    fn gradient_sigma_falling(&self, data: &PeakFitArgs) -> f64 {
        let mu = self.mu;
        let amp = self.amplitude;
        let sigma_low = self.sigma_falling;
        let sigma_high = self.sigma_rising;

        let sigma_low_squared = sigma_low.powi(2);
        let sigma_high_squared = sigma_high.powi(2);
        let sigma_low_cubed = sigma_low.powi(3);

        data.iter()
            .map(|(x, y)| {
                -2.0 * (y - if mu >= x {
                    amp * (-0.5 * (-mu + x).powi(2) / sigma_low_squared).exp()
                } else {
                    amp * (-0.5 * (-mu + x).powi(2) / sigma_high_squared).exp()
                }) * if mu >= x {
                    amp * (-mu + x).powi(2) * (-0.5 * (-mu + x).powi(2) / sigma_low_squared).exp()
                        / sigma_low_cubed
                } else {
                    0.0
                } + 1.0
            })
            .sum::<f64>()
            / data.len() as f64
    }

    fn gradient_amplitude(&self, data: &PeakFitArgs) -> f64 {
        let mu = self.mu;
        let amp = self.amplitude;
        let sigma_low = self.sigma_falling;
        let sigma_high = self.sigma_rising;

        let sigma_low_squared = sigma_low.powi(2);
        let sigma_high_squared = sigma_high.powi(2);

        data.iter()
            .map(|(x, y)| {
                -2.0 * (y - if mu >= x {
                    amp * (-0.5 * (-mu + x).powi(2) / sigma_low_squared).exp()
                } else {
                    amp * (-0.5 * (-mu + x).powi(2) / sigma_high_squared).exp()
                }) * if mu >= x {
                    (-0.5 * (-mu + x).powi(2) / sigma_low_squared).exp()
                } else {
                    (-0.5 * (-mu + x).powi(2) / sigma_high_squared).exp()
                }
            })
            .sum::<f64>()
            / data.len() as f64
    }
}

impl PeakShapeModel for BiGaussianPeakShape {
    type Fitter<'a, 'b> = PeakShapeFitter<'a, 'b, Self>;

    fn density(&self, x: f64) -> f64 {
        self.density(x)
    }

    fn gradient_update(&mut self, gradient: Self, learning_rate: f64) {
        self.gradient_update(gradient, learning_rate);
    }

    fn guess(args: &PeakFitArgs) -> Self {
        Self::guess(args)
    }

    fn loss(&self, data: &PeakFitArgs) -> f64 {
        data.iter()
            .map(|(t, i)| (i - self.density(t)).powi(2))
            .sum::<f64>()
            / data.len() as f64
            + self.regularization()
    }

    fn gradient(&self, data: &PeakFitArgs) -> Self {
        self.gradient(data)
    }
}

/// Fit a single [`PeakShapeModel`] type
#[derive(Debug, Clone)]
pub struct PeakShapeFitter<'a, 'b, T: PeakShapeModel + Debug> {
    pub data: PeakFitArgs<'a, 'b>,
    pub model: Option<T>,
}

impl<'a, 'b, T: PeakShapeModel + Debug> PeakShapeModelFitter<'a, 'b>
    for PeakShapeFitter<'a, 'b, T>
{
    type ModelType = T;

    fn from_args(args: PeakFitArgs<'a, 'b>) -> Self {
        Self::new(args)
    }

    fn gradient(&self, params: &Self::ModelType) -> Self::ModelType {
        params.gradient(&self.data)
    }

    fn loss(&self, params: &Self::ModelType) -> f64 {
        params.loss(&self.data)
    }

    fn data(&self) -> &PeakFitArgs {
        &self.data
    }

    fn fit_model(
        &mut self,
        model_params: &mut Self::ModelType,
        config: FitConfig,
    ) -> ModelFitResult {
        let mut params = model_params.clone();

        let mut last_loss = f64::INFINITY;
        let mut best_loss = f64::INFINITY;
        let mut best_params = model_params.clone();
        let mut iters = 0;
        let mut converged = false;
        let mut success = true;

        let data = if config.smooth > 0 {
            self.data().smooth(config.smooth)
        } else {
            self.data().borrow()
        };

        for it in 0..config.max_iter {
            iters = it;
            let loss = params.loss(&data);
            let gradient = params.gradient(&data);

            log::trace!("{it}: Loss = {loss:0.3}: Gradient = {gradient:?}");
            params.gradient_update(gradient, config.learning_rate);
            if loss < best_loss {
                log::trace!("{it}: Updating best parameters {params:?}");
                best_loss = loss;
                best_params = params.clone();
            }

            if (last_loss - loss).abs() / (loss + 1e-6) < config.convergence {
                log::trace!("{it}: Convergence = {}", last_loss - loss);
                converged = true;
                break;
            }
            last_loss = loss;

            if loss.is_nan() || loss.is_infinite() {
                log::trace!("{it}: Aborting, loss invalid!");
                success = false;
                break;
            }
        }

        self.model = Some(best_params.clone());
        *model_params = best_params;
        ModelFitResult::new(best_loss, iters, converged, success)
    }
}

impl<'a, 'b, T: PeakShapeModel + Debug> PeakShapeFitter<'a, 'b, T> {
    pub fn new(data: PeakFitArgs<'a, 'b>) -> Self {
        Self { data, model: None }
    }

    /// Compute the model residuals over the observed time axis
    pub fn residuals(&self) -> PeakFitArgs<'_, '_> {
        self.model.as_ref().unwrap().residuals(&self.data)
    }

    /// Create a synthetic signal profile using the observed time axis but use the model predicted signal
    /// magnitude.
    pub fn predicted(&self) -> PeakFitArgs<'_, '_> {
        let predicted = self
            .model
            .as_ref()
            .unwrap()
            .predict(&self.data.time)
            .into_iter()
            .map(|x| x as f32)
            .collect();
        let mut dup = self.data.borrow();
        dup.intensity = Cow::Owned(predicted);
        dup
    }

    /// Compute the fitted model's score on the observed data
    ///
    /// # See also
    /// [`PeakShapeModel::score`]
    pub fn score(&self) -> f64 {
        self.model.as_ref().unwrap().score(&self.data)
    }
}

/// A dispatching peak shape model that can represent a variety of different
/// peak shapes.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum PeakShape {
    Gaussian(GaussianPeakShape),
    SkewedGaussian(SkewedGaussianPeakShape),
    BiGaussian(BiGaussianPeakShape),
}

macro_rules! dispatch_peak {
    ($d:ident, $r:ident, $e:expr) => {
        match $d {
            PeakShape::Gaussian($r) => $e,
            PeakShape::SkewedGaussian($r) => $e,
            PeakShape::BiGaussian($r) => $e,
        }
    };
}

impl From<GaussianPeakShape> for PeakShape {
    fn from(value: GaussianPeakShape) -> Self {
        Self::Gaussian(value)
    }
}

impl From<SkewedGaussianPeakShape> for PeakShape {
    fn from(value: SkewedGaussianPeakShape) -> Self {
        Self::SkewedGaussian(value)
    }
}

impl From<BiGaussianPeakShape> for PeakShape {
    fn from(value: BiGaussianPeakShape) -> Self {
        Self::BiGaussian(value)
    }
}

impl PeakShape {
    /// Guess an initial [`GaussianPeakShape`]
    pub fn gaussian(data: &PeakFitArgs) -> Self {
        Self::Gaussian(GaussianPeakShape::guess(data))
    }

    /// Guess an initial [`SkewedGaussianPeakShape`]
    pub fn skewed_gaussian(data: &PeakFitArgs) -> Self {
        Self::SkewedGaussian(SkewedGaussianPeakShape::guess(data))
    }

    /// Guess an initial [`BiGaussianPeakShape`]
    pub fn bigaussian(data: &PeakFitArgs) -> Self {
        Self::BiGaussian(BiGaussianPeakShape::guess(data))
    }

    /// Compute the theoretical intensity at a specified coordinate
    pub fn density(&self, x: f64) -> f64 {
        dispatch_peak!(self, p, p.density(x))
    }

    /// Given a coordinate sequence, produce the complementary sequence of theoretical intensities
    ///
    /// # See also
    /// [`PeakShape::density`]
    pub fn predict(&self, times: &[f64]) -> Vec<f64> {
        dispatch_peak!(self, p, p.predict(times))
    }

    /// Compute the difference between the observed signal and the theoretical signal,
    /// clamping the value to be non-negative
    ///
    /// # See also
    /// [`PeakShapeModel::residuals`]
    pub fn residuals<'a, 'b, 'e: 'a + 'b>(
        &self,
        data: &'e PeakFitArgs<'a, 'b>,
    ) -> PeakFitArgs<'a, 'b> {
        dispatch_peak!(self, p, p.residuals(data))
    }

    /// Compute the 1 - ratio of the peak shape model squared error to
    /// a straight line linear model squared error.
    ///
    /// # See also
    /// [`PeakShapeModel::score`]
    pub fn score(&self, data: &PeakFitArgs) -> f64 {
        dispatch_peak!(self, p, p.score(data))
    }

    /// Fit the peak shape model to some data using the default
    /// [`FitConfig`] settings.
    ///
    /// # See also
    /// [`PeakShapeModel::fit`]
    pub fn fit(&mut self, args: PeakFitArgs) -> ModelFitResult {
        dispatch_peak!(self, p, p.fit_with(args, Default::default()))
    }

    /// Fit the peak shape model to some data using `config` options
    ///
    /// # See also
    /// [`PeakShapeModel::fit_with`]
    pub fn fit_with(&mut self, args: PeakFitArgs, config: FitConfig) -> ModelFitResult {
        dispatch_peak!(self, p, p.fit_with(args, config))
    }
}

impl PeakShapeModel for PeakShape {
    type Fitter<'a, 'b> = PeakShapeFitter<'a, 'b, Self>;

    fn density(&self, x: f64) -> f64 {
        self.density(x)
    }

    fn gradient_update(&mut self, gradient: Self, learning_rate: f64) {
        match (self, gradient) {
            (Self::Gaussian(this), Self::Gaussian(gradient)) => {
                this.gradient_update(gradient, learning_rate);
            }
            (Self::BiGaussian(this), Self::BiGaussian(gradient)) => {
                this.gradient_update(gradient, learning_rate);
            }
            (Self::SkewedGaussian(this), Self::SkewedGaussian(gradient)) => {
                this.gradient_update(gradient, learning_rate);
            }
            (this, gradient) => panic!("Invalid gradient type {gradient:?} for model {this:?}"),
        };
    }

    fn gradient(&self, data: &PeakFitArgs) -> Self {
        match self {
            PeakShape::Gaussian(model) => Self::Gaussian(model.gradient(data)),
            PeakShape::SkewedGaussian(model) => Self::SkewedGaussian(model.gradient(data)),
            PeakShape::BiGaussian(model) => Self::BiGaussian(model.gradient(data)),
        }
    }

    fn guess(args: &PeakFitArgs) -> Self {
        Self::BiGaussian(BiGaussianPeakShape::guess(args))
    }
}

/// Represent a combination of multiple [`PeakShape`] models
#[derive(Debug, Default, Clone)]
pub struct MultiPeakShapeFit {
    fits: Vec<PeakShape>,
}

impl MultiPeakShapeFit {
    pub fn new(fits: Vec<PeakShape>) -> Self {
        Self { fits }
    }

    pub fn density(&self, x: f64) -> f64 {
        self.iter().map(|p| p.density(x)).sum()
    }

    pub fn predict(&self, times: &[f64]) -> Vec<f64> {
        times.iter().map(|t| self.density(*t)).collect()
    }

    pub fn residuals<'a, 'b, 'e: 'a + 'b>(
        &self,
        data: &'e PeakFitArgs<'a, 'b>,
    ) -> PeakFitArgs<'a, 'b> {
        let mut data = data.borrow();
        for (yhat, y) in data
            .time
            .iter()
            .copied()
            .map(|t| self.density(t))
            .zip(data.intensity.to_mut().iter_mut())
        {
            *y -= yhat as f32;
            // if *y < 0.0 {
            //     *y = 0.0;
            // }
        }
        data
    }

    pub fn iter(&self) -> std::slice::Iter<'_, PeakShape> {
        self.fits.iter()
    }

    pub fn score(&self, data: &PeakFitArgs<'_, '_>) -> f64 {
        let linear_resid = data.linear_residuals();
        let mut shape_resid = 0.0;

        for (x, y) in data.iter() {
            shape_resid += (y - self.density(x)).powi(2);
        }

        let line_test = shape_resid / linear_resid;
        1.0 - line_test.max(1e-5)
    }

    pub fn push(&mut self, model: PeakShape) {
        self.fits.push(model);
    }
}

/// Fitter for multiple peak shapes on the signal split across
/// multiple disjoint intervals.
///
/// This is preferred for "real world data" which may not be
/// well behaved signal.
#[derive(Debug, Clone)]
pub struct SplittingPeakShapeFitter<'a, 'b> {
    pub data: PeakFitArgs<'a, 'b>,
    pub peak_fits: MultiPeakShapeFit,
}

impl<'a> SplittingPeakShapeFitter<'a, 'a> {
    pub fn new(data: PeakFitArgs<'a, 'a>) -> Self {
        Self {
            data,
            peak_fits: Default::default(),
        }
    }

    fn fit_chunk_with(
        &self,
        chunk: PeakFitArgs<'_, '_>,
        config: FitConfig,
    ) -> (PeakShape, ModelFitResult) {
        let mut fits = Vec::new();

        let mut model = PeakShape::bigaussian(&chunk);
        let fit_result = model.fit_with(chunk.borrow(), config.clone());
        fits.push((model, fit_result));

        let (model, fit_result) = fits
            .into_iter()
            .min_by(|a, b| a.1.loss.total_cmp(&b.1.loss))
            .unwrap();
        (model, fit_result)
    }

    /// See [`PeakShapeFitter::fit_model`]
    pub fn fit_with(&mut self, config: FitConfig) {
        let partition_points = self.data.locate_extrema(None);
        let chunks = self.data.split_at(partition_points.as_slice());

        for chunk in chunks {
            let (model, fit_result) =
                self.fit_chunk_with(self.data.slice(chunk.clone()), config.clone());
            if fit_result.success {
                self.peak_fits.push(model);
            }
        }
    }

    /// See [`PeakShapeFitter::residuals`]
    pub fn residuals(&self) -> PeakFitArgs<'_, '_> {
        self.peak_fits.residuals(&self.data)
    }

    /// See [`PeakShapeFitter::predicted`]
    pub fn predicted(&self) -> PeakFitArgs<'_, '_> {
        let predicted = self
            .peak_fits
            .predict(&self.data.time)
            .into_iter()
            .map(|x| x as f32)
            .collect();
        let mut dup = self.data.borrow();
        dup.intensity = Cow::Owned(predicted);
        dup
    }

    /// See [`PeakShapeFitter::score`]
    pub fn score(&self) -> f64 {
        self.peak_fits.score(&self.data)
    }
}

#[cfg(test)]
mod test {
    use std::{
        fs,
        io::{self, prelude::*},
    };

    use log::debug;
    use mzpeaks::{feature::Feature, Time, MZ};

    use super::*;

    #[rstest::fixture]
    #[once]
    fn feature_table() -> Vec<Feature<MZ, Time>> {
        log::info!("Logging initialized");
        crate::text::load_feature_table("test/data/features_graph.txt").unwrap()
    }

    macro_rules! assert_is_close {
        ($t1:expr, $t2:expr, $tol:expr, $label:literal) => {
            assert!(
                ($t1 - $t2).abs() < $tol,
                "Observed {} {}, expected {}, difference {}",
                $label,
                $t1,
                $t2,
                $t1 - $t2,
            );
        };
        ($t1:expr, $t2:expr, $tol:expr, $label:literal, $obj:ident) => {
            assert!(
                ($t1 - $t2).abs() < $tol,
                "Observed {} {}, expected {}, difference {} from {:?}",
                $label,
                $t1,
                $t2,
                $t1 - $t2,
                $obj
            );
        };
    }

    #[rstest::rstest]
    #[test_log::test]
    fn test_fit_feature_14216(feature_table: &[Feature<MZ, Time>]) {
        let feature = &feature_table[14216];
        let args: PeakFitArgs = feature.into();

        let wmt = args.weighted_mean_time();
        assert_is_close!(wmt, 122.3535, 1e-3, "weighted mean time");

        let mut model = SkewedGaussianPeakShape::guess(&args);

        let expected_gradient = SkewedGaussianPeakShape {
            mu: 1.0877288990485208,
            sigma: 2.066092153296829,
            amplitude: 3421141.321363151,
            lambda: 0.846178224318954,
        };
        let gradient = model.gradient(&args);

        debug!("Initial:\n{model:?}");
        debug!("Gradient combo:\n{:?}", gradient);
        debug!("Gradient split:\n{:?}", model.gradient_split(&args));

        let _res = model.fit(args.borrow());
        let _score = model.score(&args);
        debug!("{model:?}\n{_res:?}\n{_score}\n");

        let expected = SkewedGaussianPeakShape {
            mu: 121.54820923262623,
            sigma: 0.14392304906433506,
            amplitude: 4768163.602247336,
            lambda: 0.055903399861434805,
        };

        assert_is_close!(expected.mu, model.mu, 1e-2, "mu");
        assert_is_close!(expected.sigma, model.sigma, 1e-3, "sigma");

        assert_is_close!(expected_gradient.mu, gradient.mu, 1e-2, "mu");
        assert_is_close!(expected_gradient.sigma, gradient.sigma, 1e-2, "sigma");
        // unstable
        // assert_is_close!(expected.lambda, model.lambda, 1e-3, "lambda");
        // assert_is_close!(expected.amplitude, model.amplitude, 100.0, "amplitude");
    }

    #[rstest::rstest]
    fn test_fit_feature_4490(feature_table: &[Feature<MZ, Time>]) {
        let feature = &feature_table[4490];
        let args = PeakFitArgs::from(feature);

        let expected_fits = MultiPeakShapeFit {
            fits: vec![
                PeakShape::BiGaussian(BiGaussianPeakShape {
                    mu: 125.41112342515179,
                    sigma_falling: 0.2130619068583823,
                    sigma_rising: 0.22703724439718917,
                    amplitude: 2535197.152987912,
                }),
                PeakShape::BiGaussian(BiGaussianPeakShape {
                    mu: 126.05807704271226,
                    sigma_falling: 0.9997523697939726,
                    sigma_rising: 1.1813146384558728,
                    amplitude: 267102.87981724285,
                }),
            ],
        };

        let mut fitter = SplittingPeakShapeFitter::new(args.borrow());
        fitter.fit_with(FitConfig::default().max_iter(10_000));
        debug!("Score: {}", fitter.score());
        debug!("Fits: {:?}", fitter.peak_fits);

        for (exp, obs) in expected_fits.iter().zip(fitter.peak_fits.iter()) {
            let expected_mu = dispatch_peak!(exp, model, model.mu);
            let observed_mu = dispatch_peak!(obs, model, model.mu);

            assert_is_close!(expected_mu, observed_mu, 1e-3, "mu");
        }
    }

    #[rstest::rstest]
    fn test_fit_feature_10979(feature_table: &[Feature<MZ, Time>]) {
        let feature = &feature_table[10979];
        let args: PeakFitArgs<'_, '_> = feature.into();

        let expected_split_point = SplittingPoint {
            first_maximum_height: 1562937.5,
            minimum_height: 130524.61,
            second_maximum_height: 524531.8,
            minimum_time: 127.1233653584,
        };

        let observed_split_point = args.locate_extrema(None).unwrap();

        assert_is_close!(
            expected_split_point.minimum_time,
            observed_split_point.minimum_time,
            1e-2,
            "minimum_time",
            observed_split_point
        );

        let mut fitter = SplittingPeakShapeFitter::new(args.borrow());
        fitter.fit_with(FitConfig::default().max_iter(10_000).smooth(1));
        debug!("Score: {}", fitter.score());
        debug!("Fits: {:?}", fitter.peak_fits);

        let expected_fits = MultiPeakShapeFit {
            fits: vec![
                PeakShape::BiGaussian(BiGaussianPeakShape {
                    mu: 125.23159261436803,
                    sigma_falling: 0.3419458944098784,
                    sigma_rising: 0.945137432308437,
                    amplitude: 1277701.0773096785,
                }),
                PeakShape::BiGaussian(BiGaussianPeakShape {
                    mu: 127.25253895052438,
                    sigma_falling: 0.09483862559250714,
                    sigma_rising: 0.32453016302695475,
                    amplitude: 487770.8422514298,
                }),
            ],
        };

        for (exp, obs) in expected_fits.iter().zip(fitter.peak_fits.iter()) {
            let expected_mu = dispatch_peak!(exp, model, model.mu);
            let observed_mu = dispatch_peak!(obs, model, model.mu);

            assert_is_close!(observed_mu, expected_mu, 1e-3, "mu");
        }
    }

    #[rstest::rstest]
    fn test_fit_args(feature_table: &[Feature<MZ, Time>]) {
        let features = feature_table;
        let feature = &features[160];
        let (_, y, z) = feature.as_view().into_inner();
        let args = PeakFitArgs::from((y, z));

        let wmt = args.weighted_mean_time();
        assert!(
            (wmt - 123.455).abs() < 1e-3,
            "Observed average weighted mean time {wmt}, expected 123.455"
        );

        let mut model = GaussianPeakShape::new(wmt, 1.0, 1.0);
        let _res = model.fit(args.borrow());
        let _score = model.score(&args);
        // eprint!("{model:?}\n{_res:?}\n{_score}\n");

        let mu = 123.44796615442881;
        let sigma = 0.1015352963957489;
        // let amplitude = 629639.6468112208;

        assert!(
            (model.mu - mu).abs() < 1e-3,
            "Model {0} found, expected {mu}, error = {1}",
            model.mu,
            model.mu - mu
        );
        assert!(
            (model.sigma - sigma).abs() < 1e-3,
            "Model {0} found, expected {sigma}, error = {1}",
            model.sigma,
            model.sigma - sigma
        );
        // Seems to be sensitive to the platform
        // assert!(
        //     (model.amplitude - amplitude).abs() < 1e-2,
        //     "Model {0} found, expected {amplitude}, error = {1}",
        //     model.amplitude,
        //     model.amplitude - amplitude
        // );
    }

    #[rstest::rstest]
    fn test_mixed_signal() {
        let time = vec![
            5., 5.05, 5.1, 5.15, 5.2, 5.25, 5.3, 5.35, 5.4, 5.45, 5.5, 5.55, 5.6, 5.65, 5.7, 5.75,
            5.8, 5.85, 5.9, 5.95, 6., 6.05, 6.1, 6.15, 6.2, 6.25, 6.3, 6.35, 6.4, 6.45, 6.5, 6.55,
            6.6, 6.65, 6.7, 6.75, 6.8, 6.85, 6.9, 6.95, 7., 7.05, 7.1, 7.15, 7.2, 7.25, 7.3, 7.35,
            7.4, 7.45, 7.5, 7.55, 7.6, 7.65, 7.7, 7.75, 7.8, 7.85, 7.9, 7.95, 8., 8.05, 8.1, 8.15,
            8.2, 8.25, 8.3, 8.35, 8.4, 8.45, 8.5, 8.55, 8.6, 8.65, 8.7, 8.75, 8.8, 8.85, 8.9, 8.95,
            9., 9.05, 9.1, 9.15, 9.2, 9.25, 9.3, 9.35, 9.4, 9.45, 9.5, 9.55, 9.6, 9.65, 9.7, 9.75,
            9.8, 9.85, 9.9, 9.95, 10., 10.05, 10.1, 10.15, 10.2, 10.25, 10.3, 10.35, 10.4, 10.45,
            10.5, 10.55, 10.6, 10.65, 10.7, 10.75, 10.8, 10.85, 10.9, 10.95, 11., 11.05, 11.1,
            11.15, 11.2, 11.25, 11.3, 11.35, 11.4, 11.45, 11.5, 11.55, 11.6, 11.65, 11.7, 11.75,
            11.8, 11.85, 11.9, 11.95,
        ];

        let intensity: Vec<f32> = vec![
            1.27420451e-10,
            6.17462536e-10,
            2.87663017e-09,
            1.28813560e-08,
            5.54347499e-08,
            2.29248641e-07,
            9.10983876e-07,
            3.47838473e-06,
            1.27613560e-05,
            4.49843188e-05,
            1.52358163e-04,
            4.95800813e-04,
            1.55018440e-03,
            4.65685268e-03,
            1.34410596e-02,
            3.72739646e-02,
            9.93134748e-02,
            2.54238248e-01,
            6.25321648e-01,
            1.47773227e+00,
            3.35519458e+00,
            7.31929673e+00,
            1.53408987e+01,
            3.08931557e+01,
            5.97728698e+01,
            1.11116052e+02,
            1.98463694e+02,
            3.40579046e+02,
            5.61550473e+02,
            8.89601967e+02,
            1.35407175e+03,
            1.98029962e+03,
            2.78272124e+03,
            3.75722702e+03,
            4.87459147e+03,
            6.07720155e+03,
            7.28110184e+03,
            8.38438277e+03,
            9.28130702e+03,
            9.87974971e+03,
            1.01181592e+04,
            9.97790016e+03,
            9.48776976e+03,
            8.71945410e+03,
            7.77507793e+03,
            6.76998890e+03,
            5.81486954e+03,
            5.00095885e+03,
            4.39083710e+03,
            4.01545364e+03,
            3.87648735e+03,
            3.95216934e+03,
            4.20449848e+03,
            4.58618917e+03,
            5.04639622e+03,
            5.53495197e+03,
            6.00532094e+03,
            6.41666569e+03,
            6.73537620e+03,
            6.93625265e+03,
            7.00335463e+03,
            6.93041211e+03,
            6.72066330e+03,
            6.38603273e+03,
            5.94566001e+03,
            5.42389927e+03,
            4.84799871e+03,
            4.24571927e+03,
            3.64315240e+03,
            3.06295366e+03,
            2.52313467e+03,
            2.03646669e+03,
            1.61046411e+03,
            1.24784786e+03,
            9.47346984e+02,
            7.04682299e+02,
            5.13587560e+02,
            3.66751988e+02,
            2.56606294e+02,
            1.75913423e+02,
            1.18159189e+02,
            7.77629758e+01,
            5.01435513e+01,
            3.16806551e+01,
            1.96114662e+01,
            1.18949556e+01,
            7.06890964e+00,
            4.11603334e+00,
            2.34823840e+00,
            1.31263002e+00,
            7.18917952e-01,
            3.85791959e-01,
            2.02844792e-01,
            1.04498823e-01,
            5.27467595e-02,
            2.60865722e-02,
            1.26408156e-02,
            6.00164112e-03,
            2.79191246e-03,
            1.27253700e-03,
            5.68297684e-04,
            2.48667027e-04,
            1.06609858e-04,
            4.47830199e-05,
            1.84317357e-05,
            7.43285977e-06,
            2.93685491e-06,
            1.13696185e-06,
            4.31266912e-07,
            1.60281439e-07,
            5.83656297e-08,
            2.08241826e-08,
            7.27973551e-09,
            2.49344667e-09,
            8.36799506e-10,
            2.75156384e-10,
            8.86491588e-11,
            2.79837874e-11,
            8.65516222e-12,
            2.62289424e-12,
            7.78795072e-13,
            2.26570027e-13,
            6.45830522e-14,
            1.80372998e-14,
            4.93584288e-15,
            1.32339040e-15,
            3.47657399e-16,
            8.94853257e-17,
            2.25677892e-17,
            5.57651734e-18,
            1.35012489e-18,
            3.20273997e-19,
            7.44399825e-20,
            1.69522634e-20,
            3.78256125e-21,
            8.26953508e-22,
            1.77138542e-22,
            3.71776457e-23,
            7.64517712e-24,
            1.54038778e-24,
        ];

        let args = PeakFitArgs::from((time, intensity));
        let split_point = args.locate_extrema(None).unwrap();
        assert_is_close!(split_point.minimum_time, 7.5, 1e-3, "minimum_height", split_point);

        let mut fitter = SplittingPeakShapeFitter::new(args);
        fitter.fit_with(FitConfig::default().max_iter(50_000));
        let score = fitter.score();
        assert!(score > 0.95, "Expected score {score} to be greater than 0.95");
    }
}
