use std::f64::consts::{PI, SQRT_2};

use libm::erf;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use mzpeaks::prelude::Span1D;

use crate::peak_statistics::{
    fit_falling_side_width, fit_rising_side_width, full_width_at_half_max,
};

use super::{FitConstraints, PeakFitArgs, PeakShapeFitter, PeakShapeModel};

mod gaussian;
mod skewed_gaussian;
mod bigaussian;

pub use gaussian::GaussianPeakShape;
pub use skewed_gaussian::SkewedGaussianPeakShape;
pub use bigaussian::BiGaussianPeakShape;