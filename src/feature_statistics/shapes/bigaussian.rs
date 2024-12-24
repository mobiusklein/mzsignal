#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use mzpeaks::prelude::Span1D;

use crate::peak_statistics::{
    fit_falling_side_width, fit_rising_side_width
};

use super::{FitConstraints, PeakFitArgs, PeakShapeFitter, PeakShapeModel};

/// Bi-Gaussian peak shape model
///
/// ```math
/// y = \begin{cases}
///     a\exp\left({\frac{-(\mu - x)^2}{2\sigma_a^2}}\right) & x \le \mu \\
///     a\exp\left({\frac{-(\mu - x)^2}{2\sigma_b^2}}\right) & x \gt x
/// \end{cases}
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BiGaussianPeakShape {
    pub mu: f64,
    pub sigma_falling: f64,
    pub sigma_rising: f64,
    pub amplitude: f64,
}

impl BiGaussianPeakShape {
    pub fn new(mu: f64, sigma_rising: f64, sigma_falling: f64, amplitude: f64) -> Self {
        Self {
            mu,
            sigma_falling,
            sigma_rising,
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
        let falling_width = fit_falling_side_width(&data.time, &data.intensity, idx, 1.0);
        let rising_width = fit_rising_side_width(&data.time, &data.intensity, idx, 1.0);
        let total_width = data.duration();

        let sigma_falling = if falling_width.is_nan() || falling_width <= 0.0 {
            total_width / 2.0 / 2.355
        } else {
            (falling_width - mu) / 2.355
        };
        let sigma_rising = if rising_width.is_nan() || rising_width <= 0.0 {
            total_width / 2.0 / 2.355
        } else {
            (mu - rising_width) / 2.355
        };
        Self::new(mu, sigma_rising, sigma_falling, amplitude)
    }

    /// Compute the theoretical intensity at a specified coordinate
    pub fn density(&self, x: f64) -> f64 {
        if self.mu >= x {
            self.amplitude * (-0.5 * (-self.mu + x).powi(2) / self.sigma_rising.powi(2)).exp()
        } else {
            self.amplitude * (-0.5 * (-self.mu + x).powi(2) / self.sigma_falling.powi(2)).exp()
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
            self.amplitude = 0.001
        }
        if self.sigma_rising < 0.0 {
            self.sigma_rising = 0.01
        }
        if self.sigma_falling < 0.0 {
            self.sigma_falling = 0.01
        }
    }

    /// Compute the Lagrangian term for the loss function
    fn lagrangian_loss(&self, constraints: &FitConstraints) -> f64 {
        let sigma_bound = constraints.width_boundary;
        let mut loss = 0.0;
        if !constraints.center_boundaries.contains(&self.mu) {
            loss += self.amplitude.powi(2) * self.mu.powi(2);
        }
        if self.sigma_rising > sigma_bound {
            loss += self.amplitude.powi(2) * self.sigma_rising.powi(2);
        }
        if self.sigma_falling > sigma_bound {
            loss += self.amplitude.powi(2) * self.sigma_falling.powi(2);
        }
        constraints.weight * loss
    }

    pub fn loss(&self, data: &PeakFitArgs, constraints: Option<&FitConstraints>) -> f64 {
        data.iter()
            .map(|(t, i)| (i - self.density(t)).powi(2))
            .sum::<f64>()
            / data.len() as f64
            + self.regularization()
            + constraints
                .map(|constraints| self.lagrangian_loss(constraints))
                .unwrap_or_default()
    }

    fn gradient_lagrangian_iter<'a>(
        &'a self,
        data: &'a PeakFitArgs,
        constraints: &'a FitConstraints,
    ) -> impl Iterator<Item = (f64, f64, f64, f64)> + 'a {
        let width_bound = constraints.width_boundary;
        let sigma_falling = self.sigma_falling;
        let sigma_rising = self.sigma_rising;
        let width_bound = width_bound;
        let mu_bounds = &constraints.center_boundaries;

        let amp = self.amplitude;
        let mu = self.mu;

        let amp_squared = amp.powi(2);
        let mu_squared = mu.powi(2);
        let sigma_rising_squared = sigma_rising.powi(2);
        let sigma_falling_squared = sigma_falling.powi(2);

        let d_sigma_rising =  if sigma_rising > width_bound || sigma_rising < 0.0 {
            2.0 * amp_squared * sigma_rising
        } else {
            0.0
        };

        let d_sigma_falling = if sigma_falling > width_bound || sigma_falling < 0.0 {
            2.0 * amp_squared * sigma_falling
        }  else {
            0.0
        };

        let d_amplitude_rising = if !mu_bounds.contains(&mu) {
                2.0 * amp * mu_squared
            } else {
                0.0
            } + if sigma_rising > width_bound || sigma_rising < 0.0 {
                2.0 * amp * sigma_rising_squared
            } else {
                0.0
            };

        let d_amplitude_falling = if !mu_bounds.contains(&mu) {
                2.0 * amp * mu_squared
            } else {
                0.0
            } + if sigma_falling > width_bound || sigma_falling < 0.0 {
                2.0 * amp * sigma_falling_squared
            } else {
                0.0
            };

        let d_mu = if !mu_bounds.contains(&mu) {
                2.0 * amp_squared * mu
            } else {
                0.0
            };


        data.iter().map(move |(x, _)| {
            // Rising
            if mu >= x {
                (d_mu, d_sigma_rising, 0.0, d_amplitude_rising)
            } else {
                // Falling
                (d_mu, 0.0, d_sigma_falling, d_amplitude_falling)
            }
        })
    }

    fn gradient_lagrangian(&self, data: &PeakFitArgs, constraints: &FitConstraints) -> Self {
        let (mu_penalty, sigma_rising_penalty, sigma_falling_penalty, amplitude_penalty) = self
            .gradient_lagrangian_iter(data, constraints)
            .reduce(|acc, point| {
                (
                    acc.0 + point.0,
                    acc.1 + point.1,
                    acc.2 + point.2,
                    acc.3 + point.3,
                )
            })
            .unwrap_or_default();

        Self::new(
            mu_penalty,
            sigma_rising_penalty,
            sigma_falling_penalty,
            amplitude_penalty,
        )
    }

    /// Compute the regularization term for the loss function
    fn regularization(&self) -> f64 {
        self.mu + self.sigma_falling + self.sigma_rising
    }

    #[inline]
    fn jacobian_iter<'a>(
        &'a self,
        data: &'a PeakFitArgs,
    ) -> impl Iterator<Item = (f64, f64, f64, f64)> + 'a {
        let mu = self.mu;
        let amp = self.amplitude;
        let sigma_rising = self.sigma_rising;
        let sigma_falling = self.sigma_falling;

        let sigma_rising_squared = sigma_rising.powi(2);
        let sigma_falling_squared = sigma_falling.powi(2);
        let sigma_rising_cubed = sigma_rising.powi(3);
        let sigma_falling_cubed = sigma_falling.powi(3);

        data.iter().map(move |(x, y)| {
            let diff_mu_x_squared = (mu - x).powi(2);
            let amp_mu_x_diff_squared = amp * diff_mu_x_squared;

            let mut d_mu = 1.0;
            let mut d_sigma_rising = 1.0;
            let mut d_sigma_falling = 1.0;
            let mut d_amplitude = 0.0;
            // The rising arm, mu > x, e.g. x is before the mean, so we are increasing from left to right
            if mu >= x {
                let g0_rising = (-0.5 * diff_mu_x_squared / sigma_rising_squared).exp();
                let g1_rising = amp * g0_rising;
                let neg_2_y_diff_g1_rising = -2.0 * (y - g1_rising);
                d_sigma_falling += neg_2_y_diff_g1_rising * 0.0;
                d_sigma_rising +=
                    neg_2_y_diff_g1_rising * amp_mu_x_diff_squared * g0_rising / sigma_rising_cubed;

                d_mu += neg_2_y_diff_g1_rising * -0.5 * amp * (2.0 * mu - 2.0 * x) * g0_rising
                    / sigma_rising_squared;
                d_amplitude += neg_2_y_diff_g1_rising * g0_rising;
            } else {
                let g0_falling = (-0.5 * diff_mu_x_squared / sigma_falling_squared).exp();
                let g1_falling = amp * g0_falling;
                let neg_2_y_diff_g1_falling = -2.0 * (y - g1_falling);

                d_sigma_falling += neg_2_y_diff_g1_falling * amp_mu_x_diff_squared * g0_falling
                    / sigma_falling_cubed;
                d_sigma_rising += neg_2_y_diff_g1_falling * 0.0;

                d_mu += neg_2_y_diff_g1_falling * -0.5 * amp * (2.0 * mu - 2.0 * x) * g0_falling
                    / sigma_falling_squared;
                d_amplitude += neg_2_y_diff_g1_falling * g0_falling;
            }
            (d_mu, d_sigma_rising, d_sigma_falling, d_amplitude)
        })
    }

    /// Compute the gradient of the loss function for parameter optimization.
    pub fn gradient(
        &self,
        data: &PeakFitArgs,
        constraints: Option<&FitConstraints>,
    ) -> BiGaussianPeakShape {
        let (
            mut gradient_mu,
            mut gradient_sigma_rising,
            mut gradient_sigma_falling,
            mut gradient_amplitude,
        ) = self
            .jacobian_iter(data)
            .reduce(|acc, point| {
                (
                    acc.0 + point.0,
                    acc.1 + point.1,
                    acc.2 + point.2,
                    acc.3 + point.3,
                )
            })
            .unwrap_or_default();

        // Regularization
        let n = data.len() as f64;
        if let Some(constraints) = constraints {
            let penalty = self.gradient_lagrangian(data, constraints);
            gradient_mu += constraints.weight * penalty.mu;
            gradient_amplitude += constraints.weight * penalty.amplitude;
            gradient_sigma_rising += constraints.weight * penalty.sigma_rising;
            gradient_sigma_falling += constraints.weight * penalty.sigma_falling;
        }

        BiGaussianPeakShape::new(
            gradient_mu / n,
            gradient_sigma_rising / n,
            gradient_sigma_falling / n,
            gradient_amplitude / n,
        )
        .gradient_norm()
    }

    /// A non-optimized version of the gradient calculation used for testing
    /// correctness
    pub fn gradient_split(&self, data: &PeakFitArgs) -> BiGaussianPeakShape {
        let g = Self::new(
            self.gradient_mu(&data),
            self.gradient_sigma_rising(&data),
            self.gradient_sigma_falling(&data),
            self.gradient_amplitude(&data),
        );
        g.gradient_norm()
    }

    fn gradient_norm(&self) -> Self {
        let mut g = [
            self.mu,
            self.sigma_falling,
            self.sigma_rising,
            self.amplitude,
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
        let sigma_falling = self.sigma_falling;
        let sigma_rising = self.sigma_rising;

        data.iter()
            .map(|(x, y)| {
                -2.0 * (y - if mu >= x {
                    amp * (-1_f64 / 2.0 * (-mu + x).powi(2) / sigma_rising.powi(2)).exp()
                } else {
                    amp * (-1_f64 / 2.0 * (-mu + x).powi(2) / sigma_falling.powi(2)).exp()
                }) * if mu >= x {
                    -1_f64 / 2.0
                        * amp
                        * (2.0 * mu - 2.0 * x)
                        * (-1_f64 / 2.0 * (-mu + x).powi(2) / sigma_rising.powi(2)).exp()
                        / sigma_rising.powi(2)
                } else {
                    -1_f64 / 2.0
                        * amp
                        * (2.0 * mu - 2.0 * x)
                        * (-1_f64 / 2.0 * (-mu + x).powi(2) / sigma_falling.powi(2)).exp()
                        / sigma_falling.powi(2)
                } + 1.0
            })
            .sum::<f64>()
            / data.len() as f64
    }

    fn gradient_sigma_rising(&self, data: &PeakFitArgs) -> f64 {
        let mu = self.mu;
        let amp = self.amplitude;
        let sigma_falling = self.sigma_falling;
        let sigma_rising = self.sigma_rising;

        data.iter()
            .map(|(x, y)| {
                -2.0 * (y - if mu >= x {
                    amp * (-1_f64 / 2.0 * (-mu + x).powi(2) / sigma_rising.powi(2)).exp()
                } else {
                    amp * (-1_f64 / 2.0 * (-mu + x).powi(2) / sigma_falling.powi(2)).exp()
                }) * if mu >= x {
                    amp * (-mu + x).powi(2)
                        * (-1_f64 / 2.0 * (-mu + x).powi(2) / sigma_rising.powi(2)).exp()
                        / sigma_rising.powi(3)
                } else {
                    0.0
                } + 1.0
            })
            .sum::<f64>()
            / data.len() as f64
    }

    fn gradient_sigma_falling(&self, data: &PeakFitArgs) -> f64 {
        let mu = self.mu;
        let amp = self.amplitude;
        let sigma_falling = self.sigma_falling;
        let sigma_rising = self.sigma_rising;

        data.iter()
            .map(|(x, y)| {
                -2.0 * (y - if mu >= x {
                    amp * (-1_f64 / 2.0 * (-mu + x).powi(2) / sigma_rising.powi(2)).exp()
                } else {
                    amp * (-1_f64 / 2.0 * (-mu + x).powi(2) / sigma_falling.powi(2)).exp()
                }) * if mu >= x {
                    0.0
                } else {
                    amp * (-mu + x).powi(2)
                        * (-1_f64 / 2.0 * (-mu + x).powi(2) / sigma_falling.powi(2)).exp()
                        / sigma_falling.powi(3)
                } + 1.0
            })
            .sum::<f64>()
            / data.len() as f64
    }

    fn gradient_amplitude(&self, data: &PeakFitArgs) -> f64 {
        let mu = self.mu;
        let amp = self.amplitude;
        let sigma_falling = self.sigma_falling;
        let sigma_rising = self.sigma_rising;

        data.iter()
            .map(|(x, y)| {
                -2.0 * (y - if mu >= x {
                    amp * (-1_f64 / 2.0 * (-mu + x).powi(2) / sigma_rising.powi(2)).exp()
                } else {
                    amp * (-1_f64 / 2.0 * (-mu + x).powi(2) / sigma_falling.powi(2)).exp()
                }) * if mu >= x {
                    (-1_f64 / 2.0 * (-mu + x).powi(2) / sigma_rising.powi(2)).exp()
                } else {
                    (-1_f64 / 2.0 * (-mu + x).powi(2) / sigma_falling.powi(2)).exp()
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

    fn loss(&self, data: &PeakFitArgs, constraints: Option<&FitConstraints>) -> f64 {
        self.loss(data, constraints)
    }

    fn gradient(&self, data: &PeakFitArgs, constraints: Option<&FitConstraints>) -> Self {
        self.gradient(data, constraints)
    }
}
