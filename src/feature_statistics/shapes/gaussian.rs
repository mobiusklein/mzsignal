use std::f64::consts::{PI, SQRT_2};

use libm::erf;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use mzpeaks::prelude::Span1D;

use crate::peak_statistics::{
    fit_falling_side_width, fit_rising_side_width, full_width_at_half_max,
};

use super::{FitConstraints, PeakFitArgs, PeakShapeFitter, PeakShapeModel};

/// Gaussian peak shape model
///
/// ```math
/// y = a\exp\left({\frac{-(\mu - x)^2}{2\sigma^2}}\right)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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

        let fwhm = full_width_at_half_max(&data.time, &data.intensity, idx, 1.0);
        let mut sigma = fwhm.full_width_at_half_max / 2.355;
        if sigma.is_nan() || sigma == 0.0 {
            sigma = data.duration() / 2.0 / 2.355;
        }
        Self::new(mu, sigma, amplitude)
    }

    /// Compute the Lagrangian term for the loss function
    fn lagrangian_loss(&self, constraints: &FitConstraints) -> f64 {
        let sigma_bound = constraints.width_boundary;
        let mut loss = 0.0;
        if !constraints.center_boundaries.contains(&self.mu) {
            loss += self.amplitude.powi(2) * self.mu.powi(2);
        }
        if self.sigma > sigma_bound {
            loss += self.amplitude.powi(2) * self.sigma.powi(2);
        }
        constraints.weight * loss
    }

    fn gradient_lagrangian_iter<'a>(
        &'a self,
        data: &'a PeakFitArgs,
        constraints: &'a FitConstraints,
    ) -> impl Iterator<Item = (f64, f64, f64)> + 'a {
        let amp = self.amplitude;
        let mu = self.mu;
        let sigma = self.sigma;

        let sigma_bound = constraints.width_boundary;
        let mut d_amplitude = 0.0;
        let mut d_sigma = 0.0;
        let mut d_mu = 0.0;

        if !constraints.center_boundaries.contains(&self.mu) {
            d_amplitude += 2.0 * amp * mu.powi(2);
            d_mu += 2.0 * amp.powi(2) * mu;
        }
        if self.sigma > sigma_bound {
            d_sigma += 2.0 * amp.powi(2) * sigma;
            d_amplitude += 2.0 * amp * sigma.powi(2);
        }

        let d_lagrangian = (d_mu, d_sigma, d_amplitude);
        std::iter::repeat_n(d_lagrangian, data.len())
    }

    fn gradient_lagrangian(&self, data: &PeakFitArgs, constraints: &FitConstraints) -> Self {
        let (mu_penalty, sigma_penalty, amplitude_penalty) = self
            .gradient_lagrangian_iter(data, constraints)
            .reduce(|acc, point| (acc.0 + point.0, acc.1 + point.1, acc.2 + point.2))
            .unwrap_or_default();
        Self::new(mu_penalty, sigma_penalty, amplitude_penalty)
    }

    /// Compute the regularization term for the loss function
    fn regularization(&self) -> f64 {
        self.mu + self.sigma
    }

    /// Compute the loss function for optimization, mean-squared error
    pub fn loss(&self, data: &PeakFitArgs, constraints: Option<&FitConstraints>) -> f64 {
        let mut loss = data.iter()
            .map(|(t, i)| (i - self.density(t)).powi(2))
            .sum::<f64>()
            / data.len() as f64
            + self.regularization();
        if let Some(constraints) = constraints {
            loss += self.lagrangian_loss(constraints);
        }
        loss
    }

    pub fn density(&self, x: f64) -> f64 {
        self.amplitude * (-0.5 * (x - self.mu).powi(2) / self.sigma.powi(2)).exp()
    }

    #[inline]
    fn jacobian_iter<'a>(
        &'a self,
        data: &'a PeakFitArgs,
    ) -> impl Iterator<Item = (f64, f64, f64)> + 'a {
        let amp = self.amplitude;
        let mu = self.mu;
        let sigma = self.sigma;

        let two_mu = 2.0 * mu;
        let sigma_squared = sigma.powi(2);
        let sigma_cubed = sigma.powi(3);
        let sigma_squared_inv = 1.0 / sigma_squared;

        let jacobian = data.iter().map(move |(x, y)| {
            let mu_sub_x_squared = (-mu + x).powi(2);
            let half_mu_sub_x_squared_div_sigma_squared =
                -0.5 * mu_sub_x_squared * sigma_squared_inv;
            let half_mu_sub_x_squared_div_sigma_squared_exp =
                half_mu_sub_x_squared_div_sigma_squared.exp();

            let delta_y = -amp * half_mu_sub_x_squared_div_sigma_squared_exp + y;

            let delta_y_half_mu_sub_x_squared_div_sigma_squared_exp =
                delta_y * half_mu_sub_x_squared_div_sigma_squared_exp;

            let d_mu = amp
                * (two_mu - 2.0 * x)
                * delta_y_half_mu_sub_x_squared_div_sigma_squared_exp
                * sigma_squared_inv
                + 1.0;

            let d_sigma =
                -2.0 * amp * mu_sub_x_squared * delta_y_half_mu_sub_x_squared_div_sigma_squared_exp
                    / sigma_cubed
                    + 1.0;

            let d_amplitude = -2.0 * delta_y_half_mu_sub_x_squared_div_sigma_squared_exp;

            (d_mu, d_sigma, d_amplitude)
        });
        jacobian
    }

    pub fn gradient(&self, data: &PeakFitArgs, constraints: Option<&FitConstraints>) -> Self {
        let (mut gradient_mu, mut gradient_sigma, mut gradient_amplitude) = self
            .jacobian_iter(data)
            .reduce(|acc, point| (acc.0 + point.0, acc.1 + point.1, acc.2 + point.2))
            .unwrap_or_default();

        let n = data.len() as f64;

        if let Some(constraints) = constraints {
            let penalties = self.gradient_lagrangian(&data, constraints);
            gradient_mu += constraints.weight * penalties.mu;
            gradient_sigma += constraints.weight * penalties.sigma;
            gradient_amplitude += constraints.weight * penalties.amplitude;
        }

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

    fn gradient(&self, data: &PeakFitArgs, constraints: Option<&FitConstraints>) -> Self {
        self.gradient(data, constraints)
    }

    fn loss(&self, data: &PeakFitArgs, constraints: Option<&FitConstraints>) -> f64 {
        self.loss(data, constraints)
    }
}
