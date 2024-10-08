use std::{
    borrow::{Borrow, Cow},
    f64::consts::{PI, SQRT_2},
    fmt::Debug,
    iter::FusedIterator,
    marker::PhantomData,
    ops::Range,
};

use libm::erf;

use mzpeaks::prelude::TimeArray;

use crate::arrayops::{trapz, ArrayPair, ArrayPairSplit};

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

#[derive(Debug, Default, Clone, Copy, PartialEq, PartialOrd)]
pub struct SplittingPoint {
    pub first_maximum_height: f32,
    pub minimum_height: f32,
    pub second_maximum_height: f32,
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

#[derive(Debug, Default, Clone)]
pub struct PeakFitArgs<'a, 'b> {
    pub time: Cow<'a, [f64]>,
    pub intensity: Cow<'b, [f32]>,
}

impl<'c, 'd, 'a: 'c, 'b: 'd, 'e: 'c + 'd> PeakFitArgs<'a, 'b> {
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

    pub fn smooth(&self, window_size: usize) -> PeakFitArgs<'a, 'd> {
        let mut store = self.borrow();
        let sink = store.intensity.to_mut();
        crate::smooth::moving_average_dyn(&self.intensity, sink.as_mut_slice(), window_size);
        store
    }

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

    pub fn split_at(&'e self, split_points: &[SplittingPoint]) -> Vec<Range<usize>> {
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

    pub fn integrate(&self) -> f32 {
        trapz(&self.time, &self.intensity)
    }

    pub fn weighted_mean_time(&self) -> f64 {
        self.iter()
            .map(|(x, y)| ((x * y), y))
            .reduce(|(xa, ya), (x, y)| ((xa + x), (ya + y)))
            .map(|(x, y)| x / y)
            .unwrap_or_default()
    }

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

    pub fn len(&self) -> usize {
        self.time.len()
    }

    pub fn borrow(&self) -> Self {
        Self::new(
            match &self.time {
                Cow::Borrowed(x) => Cow::Borrowed(*x),
                Cow::Owned(_) => self.time.clone(),
            },
            match &self.intensity {
                Cow::Borrowed(x) => Cow::Borrowed(*x),
                Cow::Owned(_) => self.intensity.clone(),
            },
        )
    }

    pub fn null_residuals(&self) -> f64 {
        let mean = self.intensity.iter().sum::<f32>() as f64 / self.len() as f64;
        self.intensity
            .iter()
            .map(|y| (*y as f64 - mean).powi(2))
            .sum()
    }

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

    pub fn iter(&self) -> PeakFitArgsIter<'_> {
        PeakFitArgsIter::new(
            self.time
                .iter()
                .copied()
                .zip(self.intensity.iter().copied()),
        )
    }

    pub fn as_array_pair(&self) -> ArrayPairSplit<'a, 'b> {
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

pub trait FitPeaksOn<'a>
where
    PeakFitArgs<'a, 'a>: From<&'a Self>,
    Self: 'a,
{
    fn fit_peaks_with(&'a self, config: FitConfig) -> SplittingPeakShapeFitter<'a, 'a> {
        let data: PeakFitArgs<'a, 'a> = PeakFitArgs::from(self);
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

#[derive(Debug, Clone)]
pub struct FitConfig {
    max_iter: usize,
    learning_rate: f64,
    convergence: f64,
    smooth: usize,
}

impl FitConfig {
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    pub fn convergence(mut self, convergence: f64) -> Self {
        self.convergence = convergence;
        self
    }

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
            convergence: 1e-6,
            smooth: 0,
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ModelFitResult {
    pub loss: f64,
    pub iterations: usize,
    pub converged: bool,
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

pub trait PeakShapeModelFitter<'a, 'b> {
    type ModelType: PeakShapeModel + Debug;

    fn from_args(args: PeakFitArgs<'a, 'b>) -> Self;

    fn gradient(&self, params: &Self::ModelType) -> Self::ModelType;

    fn loss(&self, params: &Self::ModelType) -> f64;

    fn data(&self) -> &PeakFitArgs;

    fn iter(&self) -> PeakFitArgsIter {
        self.data().iter()
    }

    fn score(&self, model_params: &Self::ModelType) -> f64 {
        model_params.score(self.data())
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

            if (last_loss - loss).abs() < config.convergence {
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
        *model_params = best_params;
        ModelFitResult::new(best_loss, iters, converged, success)
    }
}

pub trait PeakShapeModel: Clone {
    type Fitter<'a, 'b>: PeakShapeModelFitter<'a, 'b, ModelType = Self>;

    fn density(&self, x: f64) -> f64;

    fn gradient_update(&mut self, gradient: Self, learning_rate: f64);

    fn predict(&self, times: &[f64]) -> Vec<f64> {
        times.iter().map(|t| self.density(*t)).collect()
    }

    fn predict_iter<I: IntoIterator<Item = f64>>(&self, times: I) -> impl Iterator<Item = f64> {
        times.into_iter().map(|t| self.density(t))
    }

    fn gradient(&self, data: &PeakFitArgs) -> Self;

    fn loss(&self, data: &PeakFitArgs) -> f64 {
        data.iter()
            .map(|(t, i)| (i - self.density(t)).powi(2))
            .sum::<f64>()
            / data.len() as f64
    }

    fn residuals<'a, 'b>(&self, data: &PeakFitArgs<'a, 'b>) -> PeakFitArgs<'a, 'b> {
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

    fn score(&self, data: &PeakFitArgs) -> f64 {
        let linear_resid = data.linear_residuals();
        let mut shape_resid = 0.0;

        for (x, y) in data.iter() {
            shape_resid += (y - self.density(x)).powi(2);
        }

        let line_test = shape_resid / linear_resid;
        1.0 - line_test.max(1e-5)
    }

    fn guess(args: &PeakFitArgs) -> Self;

    fn fit(&mut self, args: PeakFitArgs) -> ModelFitResult {
        self.fit_with(args, Default::default())
    }

    fn fit_with(&mut self, args: PeakFitArgs, config: FitConfig) -> ModelFitResult {
        let mut fitter = Self::Fitter::from_args(args);
        fitter.fit_model(self, config)
    }
}

#[derive(Debug, Clone, Copy)]
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

    pub fn regularization(&self) -> f64 {
        self.mu + self.sigma
    }

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

    pub fn from_slice(data: &[f64]) -> Self {
        Self {
            mu: data[0],
            sigma: data[1],
            amplitude: data[2],
        }
    }

    pub fn to_vec(&self) -> Vec<f64> {
        vec![self.mu, self.sigma, self.amplitude]
    }

    pub fn mu_gradient(&self, data: &PeakFitArgs) -> f64 {
        let amp = self.amplitude;
        let mu = self.mu;
        let sigma = self.sigma;

        let two_mu = 2.0 * mu;
        let sigma_squared = sigma.powi(2);

        let grad: f64 = data
            .iter()
            .map(|(x, y)| {
                let mu_sub_x_squared = (-mu + x).powi(2);
                let half_mu_sub_x_squared_div_sigma_squared =
                    -0.5 * mu_sub_x_squared / sigma_squared;
                let half_mu_sub_x_squared_div_sigma_squared_exp =
                    half_mu_sub_x_squared_div_sigma_squared.exp();

                amp * (two_mu - 2.0 * x)
                    * (-amp * half_mu_sub_x_squared_div_sigma_squared_exp + y)
                    * half_mu_sub_x_squared_div_sigma_squared_exp
                    / sigma_squared
                    + 1.0
            })
            .sum();

        grad / data.len() as f64
    }

    pub fn sigma_gradient(&self, data: &PeakFitArgs) -> f64 {
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

    pub fn amplitude_gradient(&self, data: &PeakFitArgs) -> f64 {
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

    pub fn gradient(&self, data: &PeakFitArgs) -> Self {
        let mut g = [
            self.mu_gradient(&data),
            self.sigma_gradient(&data),
            self.amplitude_gradient(&data),
        ];
        let gradnorm: f64 = g.iter().map(|f| f.abs()).sum::<f64>() / g.len() as f64;
        if gradnorm > 1.0 {
            g[0] /= gradnorm;
            g[1] /= gradnorm;
        }

        Self::new(g[0], g[1], g[2])
    }

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

#[derive(Debug, Clone, Copy)]
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

    pub fn density(&self, x: f64) -> f64 {
        self.amplitude
            * (erf(SQRT_2 * self.lambda * (-self.mu + x) / (2.0 * self.sigma)) + 1.0)
            * (-0.5 * (-self.mu + x).powi(2) / self.sigma.powi(2)).exp()
    }

    pub fn from_slice(data: &[f64]) -> Self {
        Self {
            mu: data[0],
            sigma: data[1],
            amplitude: data[2],
            lambda: data[3],
        }
    }

    pub fn regularization(&self) -> f64 {
        self.mu + self.sigma + self.lambda
    }

    pub fn to_vec(&self) -> Vec<f64> {
        vec![self.mu, self.sigma, self.amplitude, self.lambda]
    }

    pub fn gradient(&self, data: &PeakFitArgs) -> SkewedGaussianPeakShape {
        let mut g = [
            self.mu_gradient(&data),
            self.sigma_gradient(&data),
            self.amplitude_gradient(&data),
            self.lambda_gradient(&data),
        ];
        let gradnorm: f64 = g.iter().map(|f| f.abs()).sum::<f64>() / g.len() as f64;
        if gradnorm > 1.0 {
            g[0] /= gradnorm;
            g[1] /= gradnorm;
            g[3] /= gradnorm;
        }

        SkewedGaussianPeakShape::new(g[0], g[1], g[2], g[3])
    }

    pub fn mu_gradient(&self, data: &PeakFitArgs) -> f64 {
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

    pub fn sigma_gradient(&self, data: &PeakFitArgs) -> f64 {
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

    pub fn amplitude_gradient(&self, data: &PeakFitArgs) -> f64 {
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

    pub fn lambda_gradient(&self, data: &PeakFitArgs) -> f64 {
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

    fn loss(&self, data: &PeakFitArgs) -> f64 {
        data.iter()
            .map(|(t, i)| (i - self.density(t)).powi(2))
            .sum::<f64>()
            / data.len() as f64
            + self.regularization()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BiGaussianPeakShape {
    pub mu: f64,
    pub sigma_low: f64,
    pub sigma_high: f64,
    pub amplitude: f64,
}

impl BiGaussianPeakShape {
    pub fn new(mu: f64, sigma_low: f64, sigma_high: f64, amplitude: f64) -> Self {
        Self {
            mu,
            sigma_low,
            sigma_high,
            amplitude,
        }
    }

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

    pub fn density(&self, x: f64) -> f64 {
        if self.mu >= x {
            self.amplitude * (-1_f64 / 2.0 * (-self.mu + x).powi(2) / self.sigma_low.powi(2)).exp()
        } else {
            self.amplitude * (-1_f64 / 2.0 * (-self.mu + x).powi(2) / self.sigma_high.powi(2)).exp()
        }
    }

    pub fn gradient_update(&mut self, gradient: Self, learning_rate: f64) {
        self.mu -= gradient.mu * learning_rate;
        self.sigma_low -= gradient.sigma_low * learning_rate;
        self.sigma_high -= gradient.sigma_high * learning_rate;

        self.amplitude -= gradient.amplitude * learning_rate;
        if self.amplitude < 0.0 {
            self.amplitude = 0.0
        }
    }

    pub fn regularization(&self) -> f64 {
        self.mu + self.sigma_low + self.sigma_high
    }

    pub fn gradient(&self, data: &PeakFitArgs) -> BiGaussianPeakShape {
        let mut g = [
            self.gradient_mu(&data),
            self.gradient_sigma_low(&data),
            self.gradient_sigma_high(&data),
            self.gradient_amplitude(&data),
        ];
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
        let sigma_low = self.sigma_low;
        let sigma_high = self.sigma_high;

        data.iter()
            .map(|(x, y)| {
                -2.0 * (y - if mu >= x {
                    amp * (-0.5 * (-mu + x).powi(2) / sigma_low.powi(2)).exp()
                } else {
                    amp * (-0.5 * (-mu + x).powi(2) / sigma_high.powi(2)).exp()
                }) * if mu >= x {
                    -0.5 * amp
                        * (2.0 * mu - 2.0 * x)
                        * (-0.5 * (-mu + x).powi(2) / sigma_low.powi(2)).exp()
                        / sigma_low.powi(2)
                } else {
                    -0.5 * amp
                        * (2.0 * mu - 2.0 * x)
                        * (-0.5 * (-mu + x).powi(2) / sigma_high.powi(2)).exp()
                        / sigma_high.powi(2)
                } + 1.0
            })
            .sum::<f64>()
            / data.len() as f64
    }

    fn gradient_sigma_high(&self, data: &PeakFitArgs) -> f64 {
        let mu = self.mu;
        let amp = self.amplitude;
        let sigma_low = self.sigma_low;
        let sigma_high = self.sigma_high;

        data.iter()
            .map(|(x, y)| {
                -2.0 * (y - if mu >= x {
                    amp * (-1_f64 / 2.0 * (-mu + x).powi(2) / sigma_low.powi(2)).exp()
                } else {
                    amp * (-1_f64 / 2.0 * (-mu + x).powi(2) / sigma_high.powi(2)).exp()
                }) * if mu >= x {
                    0.0
                } else {
                    amp * (-mu + x).powi(2)
                        * (-1_f64 / 2.0 * (-mu + x).powi(2) / sigma_high.powi(2)).exp()
                        / sigma_high.powi(3)
                } + 1.0
            })
            .sum::<f64>()
            / data.len() as f64
    }

    fn gradient_sigma_low(&self, data: &PeakFitArgs) -> f64 {
        let mu = self.mu;
        let amp = self.amplitude;
        let sigma_low = self.sigma_low;
        let sigma_high = self.sigma_high;

        data.iter()
            .map(|(x, y)| {
                -2.0 * (y - if mu >= x {
                    amp * (-1_f64 / 2.0 * (-mu + x).powi(2) / sigma_low.powi(2)).exp()
                } else {
                    amp * (-1_f64 / 2.0 * (-mu + x).powi(2) / sigma_high.powi(2)).exp()
                }) * if mu >= x {
                    amp * (-mu + x).powi(2)
                        * (-1_f64 / 2.0 * (-mu + x).powi(2) / sigma_low.powi(2)).exp()
                        / sigma_low.powi(3)
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
        let sigma_low = self.sigma_low;
        let sigma_high = self.sigma_high;

        data.iter()
            .map(|(x, y)| {
                -2.0 * (y - if mu >= x {
                    amp * (-1_f64 / 2.0 * (-mu + x).powi(2) / sigma_low.powi(2)).exp()
                } else {
                    amp * (-1_f64 / 2.0 * (-mu + x).powi(2) / sigma_high.powi(2)).exp()
                }) * if mu >= x {
                    (-1_f64 / 2.0 * (-mu + x).powi(2) / sigma_low.powi(2)).exp()
                } else {
                    (-1_f64 / 2.0 * (-mu + x).powi(2) / sigma_high.powi(2)).exp()
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

#[derive(Debug)]
pub struct PeakShapeFitter<'a, 'b, T: PeakShapeModel + Debug> {
    pub data: PeakFitArgs<'a, 'b>,
    pub model: Option<T>,
}

impl<'a, 'b, T: PeakShapeModel + Debug> PeakShapeModelFitter<'a, 'b> for PeakShapeFitter<'a, 'b, T> {
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

            if (last_loss - loss).abs() < config.convergence {
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

    pub fn residuals(&self) -> PeakFitArgs<'_, '_> {
        self.model.as_ref().unwrap().residuals(&self.data)
    }

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

    pub fn score(&self) -> f64 {
        self.model.as_ref().unwrap().score(&self.data)
    }
}

#[derive(Debug, Clone, Copy)]
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

impl PeakShape {
    pub fn gaussian(data: &PeakFitArgs) -> Self {
        Self::Gaussian(GaussianPeakShape::guess(data))
    }

    pub fn skewed_gaussian(data: &PeakFitArgs) -> Self {
        Self::SkewedGaussian(SkewedGaussianPeakShape::guess(data))
    }

    pub fn bigaussian(data: &PeakFitArgs) -> Self {
        Self::BiGaussian(BiGaussianPeakShape::guess(data))
    }

    pub fn density(&self, x: f64) -> f64 {
        dispatch_peak!(self, p, p.density(x))
    }

    pub fn predict(&self, times: &[f64]) -> Vec<f64> {
        dispatch_peak!(self, p, p.predict(times))
    }

    pub fn residuals<'a, 'b>(&self, data: &PeakFitArgs<'a, 'b>) -> PeakFitArgs<'a, 'b> {
        dispatch_peak!(self, p, p.residuals(data))
    }

    pub fn score(&self, data: &PeakFitArgs) -> f64 {
        dispatch_peak!(self, p, p.score(data))
    }

    pub fn fit(&mut self, args: PeakFitArgs) -> ModelFitResult {
        dispatch_peak!(self, p, p.fit_with(args, Default::default()))
    }

    pub fn fit_with(&mut self, args: PeakFitArgs, config: FitConfig) -> ModelFitResult {
        dispatch_peak!(self, p, p.fit_with(args, config))
    }
}

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

    pub fn residuals<'a, 'b>(&self, data: &PeakFitArgs<'a, 'b>) -> PeakFitArgs<'a, 'b> {
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

    pub fn fit_with(&mut self, config: FitConfig) {
        let partition_points = self.data.locate_extrema(None);
        let chunks = self.data.split_at(partition_points.as_slice());

        for chunk in chunks {
            let (model, _fit_result) =
                self.fit_chunk_with(self.data.slice(chunk.clone()), config.clone());
            self.peak_fits.push(model);
        }
    }

    pub fn residuals(&self) -> PeakFitArgs<'_, '_> {
        self.peak_fits.residuals(&self.data)
    }

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

        eprintln!("Initial:\n{model:?}");
        eprintln!("Gradient:\n{:?}", gradient);

        let res = model.fit(args.borrow());
        let score = model.score(&args);
        eprintln!("{model:?}\n{res:?}\n{score}\n");

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
                    sigma_low: 0.2130619068583823,
                    sigma_high: 0.22703724439718917,
                    amplitude: 2535197.152987912,
                }),
                PeakShape::BiGaussian(BiGaussianPeakShape {
                    mu: 126.05807704271226,
                    sigma_low: 0.9997523697939726,
                    sigma_high: 1.1813146384558728,
                    amplitude: 267102.87981724285,
                }),
            ],
        };

        let mut fitter = SplittingPeakShapeFitter::new(args);
        fitter.fit_with(FitConfig::default().max_iter(10_000));
        eprintln!("Score: {}", fitter.score());
        eprintln!("Fits: {:?}", fitter.peak_fits);

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

        eprintln!("Split Points: {:?}", observed_split_point,);

        assert_is_close!(
            expected_split_point.minimum_time,
            observed_split_point.minimum_time,
            1e-2,
            "minimum_time"
        );

        let mut fitter = SplittingPeakShapeFitter::new(args.borrow());
        fitter.fit_with(FitConfig::default().max_iter(10_000).smooth(3));
        eprintln!("Score: {}", fitter.score());
        eprintln!("Fits: {:?}", fitter.peak_fits);

        let expected_fits = MultiPeakShapeFit {
            fits: vec![
                PeakShape::BiGaussian(BiGaussianPeakShape {
                    mu: 125.28364636849112,
                    sigma_low: 0.3419458944098784,
                    sigma_high: 0.945137432308437,
                    amplitude: 1277701.0773096785,
                }),
                PeakShape::BiGaussian(BiGaussianPeakShape {
                    mu: 127.32401611160664,
                    sigma_low: 0.09483862559250714,
                    sigma_high: 0.32453016302695475,
                    amplitude: 487770.8422514298,
                }),
            ],
        };

        for (exp, obs) in expected_fits.iter().zip(fitter.peak_fits.iter()) {
            let expected_mu = dispatch_peak!(exp, model, model.mu);
            let observed_mu = dispatch_peak!(obs, model, model.mu);

            assert_is_close!(expected_mu, observed_mu, 1e-3, "mu");
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
        let res = model.fit(args.borrow());
        let score = model.score(&args);
        eprint!("{model:?}\n{res:?}\n{score}\n");

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
}
