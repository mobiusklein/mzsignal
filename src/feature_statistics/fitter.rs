use std::{borrow::Cow, fmt::Debug};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::{
    FitConfig, FitConstraints, ModelFitResult, MultiPeakShapeFit, PeakFitArgs, PeakShape,
    PeakShapeModel, PeakShapeModelFitter, SplittingPoint,
};

/// Fit a single [`PeakShapeModel`] type
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
        params.gradient(&self.data, None)
    }

    fn loss(&self, params: &Self::ModelType, constraints: Option<&FitConstraints>) -> f64 {
        params.loss(&self.data, constraints)
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

        let start_t = self.data.time.first().copied().unwrap_or_default();
        let end_t = self.data.time.last().copied().unwrap_or_default();

        let mut constraints = FitConstraints::default();
        let mut constraints_ref = None;

        if config.use_constraints {
            constraints = constraints.width_boundary(end_t - start_t)
            .center_lower_bound(start_t)
            .center_upper_bound(end_t)
            .weight(0.1);
            constraints_ref = Some(&constraints);
        };

        for it in 0..config.max_iter {
            iters = it;
            let loss = params.loss(&data, constraints_ref);
            let gradient = params.gradient(&data, constraints_ref);

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

        let score = best_params.score(&data);
        self.model = Some(best_params.clone());
        *model_params = best_params;
        ModelFitResult::new(best_loss, iters, converged, success, score)
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

/// Fitter for multiple peak shapes on the signal split across
/// multiple disjoint intervals.
///
/// This is preferred for "real world data" which may not be
/// well behaved signal.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
        config: &FitConfig,
    ) -> (PeakShape, ModelFitResult) {
        let mut fits = Vec::new();

        let mut model = PeakShape::bigaussian(&chunk);
        let fit_result = model.fit_with(chunk.borrow(), config.clone());
        fits.push((model, fit_result));

        let mut model = PeakShape::skewed_gaussian(&chunk);
        let fit_result = model.fit_with(chunk.borrow(), config.clone());
        fits.push((model, fit_result));

        let mut model = PeakShape::gaussian(&chunk);
        let fit_result = model.fit_with(chunk.borrow(), config.clone());
        fits.push((model, fit_result));

        let (model, fit_result) = fits
            .into_iter()
            .max_by(|a, b| a.1.score.total_cmp(&b.1.score))
            .unwrap();
        (model, fit_result)
    }

    fn should_split(
        &self,
        data: PeakFitArgs<'a, 'a>,
        config: &FitConfig,
    ) -> (bool, Option<SplittingPoint>) {
        let threshold = data
            .intensity
            .get(data.argmax())
            .copied()
            .unwrap_or_default()
            * config.splitting_threshold;
        let partition_points = data.locate_extrema(None).and_then(|sp| {
            let dist = sp.total_distance();
            if dist / 2.0 > threshold {
                Some(sp)
            } else {
                None
            }
        });
        let chunks = data.split_at(partition_points.as_slice());
        let should_split = chunks.iter().filter(|chunk| (chunk.len() >= 6)).count() > 1;
        (should_split, partition_points)
    }

    fn split_then_fit(
        &self,
        data: PeakFitArgs<'a, 'a>,
        config: &FitConfig,
        split_points: &[SplittingPoint],
    ) -> MultiPeakShapeFit {
        let mut peak_fits = Vec::new();
        let chunks = data.split_at(split_points);
        for chunk in chunks {
            if chunk.len() < 6 {
                continue;
            }
            let data_chunk = data.slice(chunk.clone());
            let (model, fit_result) = self.fit_chunk_with(data_chunk, config);
            if fit_result.success {
                peak_fits.push(model);
            }
        }
        MultiPeakShapeFit::new(peak_fits)
    }

    /// See [`PeakShapeFitter::fit_model`]
    pub fn fit_with(&mut self, mut config: FitConfig) {
        let data = if config.smooth > 0 {
            let smooth = config.smooth;
            config = config.smooth(0);
            self.data.smooth(smooth)
        } else {
            self.data.borrow()
        };

        let (total_fit, fit_result) = self.fit_chunk_with(data.borrow(), &config);

        let (should_split, partition_points) = self.should_split(data.borrow(), &config);
        if !should_split {
            self.peak_fits.push(total_fit);
            return;
        }

        let split_fits = self.split_then_fit(data.borrow(), &config, partition_points.as_slice());

        let total_score = if fit_result.success {
            total_fit.score(&data)
        } else {
            f64::NEG_INFINITY
        };

        let split_score = split_fits.score(&data);

        if total_score > split_score {
            self.peak_fits.push(total_fit);
        } else {
            self.peak_fits.extend(split_fits)
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
