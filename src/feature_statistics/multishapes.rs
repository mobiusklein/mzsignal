#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::{
    BiGaussianPeakShape, FitConfig, FitConstraints, GaussianPeakShape, ModelFitResult, PeakFitArgs,
    PeakShapeFitter, PeakShapeModel, SkewedGaussianPeakShape, PeakShapeModelFitter
};

/// A dispatching peak shape model that can represent a variety of different
/// peak shapes.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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

pub(crate) use dispatch_peak;

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

    fn gradient(&self, data: &PeakFitArgs, _constraints: Option<&FitConstraints>) -> Self {
        match self {
            PeakShape::Gaussian(model) => Self::Gaussian(model.gradient(data, _constraints)),
            PeakShape::SkewedGaussian(model) => Self::SkewedGaussian(model.gradient(data)),
            PeakShape::BiGaussian(model) => Self::BiGaussian(model.gradient(data, _constraints)),
        }
    }

    fn guess(args: &PeakFitArgs) -> Self {
        Self::BiGaussian(BiGaussianPeakShape::guess(args))
    }

    fn loss(&self, data: &PeakFitArgs, constraints: Option<&FitConstraints>) -> f64 {
        match self {
            PeakShape::Gaussian(model) => PeakShapeModel::loss(model, data, constraints),
            PeakShape::SkewedGaussian(model) => model.loss(data, constraints),
            PeakShape::BiGaussian(model) => model.loss(data, constraints),
        }
    }
}

/// Represent a combination of multiple [`PeakShape`] models
#[derive(Debug, Default, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MultiPeakShapeFit {
    pub fits: Vec<PeakShape>,
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


impl Extend<PeakShape> for MultiPeakShapeFit {
    fn extend<T: IntoIterator<Item = PeakShape>>(&mut self, iter: T) {
        self.fits.extend(iter)
    }
}

impl IntoIterator for MultiPeakShapeFit {
    type Item = PeakShape;

    type IntoIter = std::vec::IntoIter<PeakShape>;

    fn into_iter(self) -> Self::IntoIter {
        self.fits.into_iter()
    }
}