use std::fmt::Debug;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::{PeakFitArgs, PeakFitArgsIter};

/// Hyperparameters for fitting a peak shape model
#[derive(Debug, Clone)]
pub struct FitConfig {
    /// The maximum number of iterations to attempt when fitting a peak model
    pub max_iter: usize,
    /// The rate at which model parameters are updated
    pub learning_rate: f64,
    /// The minimum distance between the current loss and the previous loss at which to decide the model
    /// has converged
    pub convergence: f64,
    /// How much smoothing to perform before fitting a peak model.
    ///
    /// See [`PeakFitArgs::smooth`]
    pub smooth: usize,

    pub splitting_threshold: f32,
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

    pub fn splitting_threshold(mut self, value: f32) -> Self {
        self.splitting_threshold = value;
        self
    }
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            max_iter: 50_000,
            learning_rate: 1e-4,
            convergence: 1e-9,
            smooth: 0,
            splitting_threshold: 0.1,
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
    pub score: f64,
}

impl ModelFitResult {
    pub fn new(loss: f64, iterations: usize, converged: bool, success: bool, score: f64) -> Self {
        Self {
            loss,
            iterations,
            converged,
            success,
            score,
        }
    }
}


#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FitConstraints {
    pub center_boundaries: (f64, f64),
    pub width_boundary: f64,
    pub weight: f64,
}

impl Default for FitConstraints {
    fn default() -> Self {
        Self {
            center_boundaries: (f64::NEG_INFINITY, f64::INFINITY),
            width_boundary: 0.05,
            weight: 0.1,
        }
    }
}

impl FitConstraints {
    pub fn new(center_boundaries: (f64, f64), width_boundary: f64, weight: f64) -> Self {
        Self {
            center_boundaries,
            width_boundary,
            weight,
        }
    }

    pub fn weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }

    pub fn center_lower_bound(mut self, lower_bound: f64) -> Self {
        self.center_boundaries.0 = lower_bound;
        self
    }

    pub fn center_upper_bound(mut self, upper_bound: f64) -> Self {
        self.center_boundaries.1 = upper_bound;
        self
    }

    pub fn width_boundary(mut self, width_boundary: f64) -> Self {
        self.width_boundary = width_boundary;
        self
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
    fn gradient(&self, data: &PeakFitArgs, constraints: Option<&FitConstraints>) -> Self;

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