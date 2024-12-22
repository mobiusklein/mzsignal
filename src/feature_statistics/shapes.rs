use std::f64::consts::{PI, SQRT_2};

use libm::erf;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

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
        let sigma = fwhm.full_width_at_half_max / 2.355;
        Self::new(mu, sigma, amplitude)
    }

    pub fn lagrangian(&self, data: &PeakFitArgs, constraints: &FitConstraints) -> Self {
        let n = data.len() as f64;
        let sigma_bound = constraints.width_boundary;
        let amplitude_penalty =
            (2.0 * if constraints.center_boundaries.0 >= self.mu
                || self.mu >= constraints.center_boundaries.1
            {
                self.mu
            } else {
                0.0
            } + 2.0
                * if self.sigma > sigma_bound {
                    self.sigma
                } else {
                    0.0
                })
                * (if constraints.center_boundaries.0 >= self.mu
                    || self.mu >= constraints.center_boundaries.1
                {
                    self.amplitude * self.mu
                } else {
                    0.0
                } + if self.sigma > sigma_bound {
                    self.amplitude * self.sigma
                } else {
                    0.0
                });

        let sigma_penalty =
            2.0 * (if constraints.center_boundaries.0 >= self.mu
                || self.mu >= constraints.center_boundaries.1
            {
                self.amplitude * self.mu
            } else {
                0.0
            } + if self.sigma > sigma_bound {
                self.amplitude * self.sigma
            } else {
                0.0
            }) * if self.sigma > sigma_bound {
                self.amplitude
            } else {
                0.0
            };

        let mu_penalty =
            2.0 * (if constraints.center_boundaries.0 >= self.mu
                || self.mu >= constraints.center_boundaries.1
            {
                self.amplitude * self.mu
            } else {
                0.0
            } + if self.sigma > sigma_bound {
                self.amplitude * self.sigma
            } else {
                0.0
            }) * if constraints.center_boundaries.0 >= self.mu
                || self.mu >= constraints.center_boundaries.1
            {
                self.amplitude
            } else {
                0.0
            };
        Self::new(mu_penalty * n, sigma_penalty * n, amplitude_penalty * n)
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

    pub fn gradient(&self, data: &PeakFitArgs, constraints: Option<&FitConstraints>) -> Self {
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

        if let Some(constraints) = constraints {
            let penalties = self.lagrangian(&data, constraints);
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
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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

        let mut falling_width = fit_falling_side_width(&data.time, &data.intensity, idx, 1.0);
        let mut rising_width = fit_rising_side_width(&data.time, &data.intensity, idx, 1.0);
        let total_width = data.time.last().copied().unwrap() - data.time.first().copied().unwrap();
        if falling_width.is_nan() || falling_width <= 0.0 {
            falling_width = total_width / 2.0;
        } else {
            falling_width -= mu;
        }
        if rising_width.is_nan() || rising_width <= 0.0 {
            rising_width = total_width / 2.0;
        } else {
            rising_width = mu - rising_width;
        }
        let sigma = ((falling_width / 2.355) + (rising_width / 2.355)) / 2.0;

        let lambda = 1.0;
        let mut this = Self::new(mu, sigma, amplitude, lambda);
        let (lambda, _score) = [-4.0, -2.0, -1.0, 1.0, 2.0, 4.0]
            .into_iter()
            .map(|lambda: f64| {
                this.lambda = lambda;
                (lambda, this.score(data))
            })
            .reduce(|state, next| if state.1 < next.1 { next } else { state })
            .unwrap();
        this.lambda = lambda;
        this
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

    fn gradient(&self, data: &PeakFitArgs, _constraints: Option<&FitConstraints>) -> Self {
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
        let total_width = data.time.last().copied().unwrap() - data.time.first().copied().unwrap();

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

    pub fn lagrangian(&self, data: &PeakFitArgs, constraints: &FitConstraints) -> Self {
        let width_bound = constraints.width_boundary;
        let (lb, ub) = constraints.center_boundaries;

        let mut sigma_rising_penalty = 0.0;
        let mut sigma_falling_penalty = 0.0;
        let mut mu_penalty = 0.0;
        let mut amplitude_penalty = 0.0;

        let sigma_falling = self.sigma_falling;
        let sigma_rising = self.sigma_rising;
        let width_bound = width_bound;

        let amp = self.amplitude;
        let mu = self.mu;

        for (x, _) in data.iter() {
            sigma_rising_penalty += if mu >= x && sigma_rising > width_bound || sigma_rising < 0.0 {
                2.0 * amp.powi(2) * sigma_rising
            } else {
                0.0
            };

            sigma_falling_penalty +=
                if mu >= x && sigma_falling > width_bound || sigma_falling < 0.0 {
                    2.0 * amp.powi(2) * sigma_falling
                } else {
                    0.0
                };

            amplitude_penalty +=
                if lb >= mu || mu >= ub {
                    2.0 * amp * mu.powi(2)
                } else {
                    0.0
                } + if mu >= x && sigma_falling > width_bound || sigma_falling < 0.0 {
                    2.0 * amp * sigma_falling.powi(2)
                } else {
                    0.0
                } + if mu >= x && sigma_rising > width_bound || sigma_rising < 0.0 {
                    2.0 * amp * sigma_rising.powi(2)
                } else {
                    0.0
                };

            mu_penalty += if lb >= mu || mu >= ub {
                2.0 * amp.powi(2) * mu
            } else {
                0.0
            };
        }

        Self::new(
            mu_penalty,
            sigma_rising_penalty,
            sigma_falling_penalty,
            amplitude_penalty,
        )
    }

    /// Compute the regularization term for the loss function
    pub fn regularization(&self) -> f64 {
        self.mu + self.sigma_falling + self.sigma_rising
    }

    /// Compute the gradient of the loss function for parameter optimization.
    pub fn gradient(
        &self,
        data: &PeakFitArgs,
        constraints: Option<&FitConstraints>,
    ) -> BiGaussianPeakShape {
        let mu = self.mu;
        let amp = self.amplitude;
        let sigma_rising = self.sigma_rising;
        let sigma_falling = self.sigma_falling;

        let sigma_rising_squared = sigma_rising.powi(2);
        let sigma_falling_squared = sigma_falling.powi(2);
        let sigma_rising_cubed = sigma_rising.powi(3);
        let sigma_falling_cubed = sigma_falling.powi(3);

        let mut gradient_mu = 0.0;
        let mut gradient_sigma_falling = 0.0;
        let mut gradient_sigma_rising = 0.0;
        let mut gradient_amplitude = 0.0;

        for (x, y) in data.iter() {
            let diff_mu_x_squared = (mu - x).powi(2);
            let amp_mu_x_diff_squared = amp * diff_mu_x_squared;

            // The rising arm, mu > x, e.g. x is before the mean, so we are increasing from left to right
            if mu >= x {
                let g0_rising = (-0.5 * diff_mu_x_squared / sigma_rising_squared).exp();
                let g1_rising = amp * g0_rising;
                let neg_2_y_diff_g1_rising = -2.0 * (y - g1_rising);
                gradient_sigma_falling += neg_2_y_diff_g1_rising * 0.0;
                gradient_sigma_rising +=
                    neg_2_y_diff_g1_rising * amp_mu_x_diff_squared * g0_rising / sigma_rising_cubed;

                gradient_mu +=
                    neg_2_y_diff_g1_rising * -0.5 * amp * (2.0 * mu - 2.0 * x) * g0_rising
                        / sigma_rising_squared;
                gradient_amplitude += neg_2_y_diff_g1_rising * g0_rising;
            } else {
                let g0_falling = (-0.5 * diff_mu_x_squared / sigma_falling_squared).exp();
                let g1_falling = amp * g0_falling;
                let neg_2_y_diff_g1_falling = -2.0 * (y - g1_falling);

                gradient_sigma_falling +=
                    neg_2_y_diff_g1_falling * amp_mu_x_diff_squared * g0_falling
                        / sigma_falling_cubed;
                gradient_sigma_rising += neg_2_y_diff_g1_falling * 0.0;

                gradient_mu +=
                    neg_2_y_diff_g1_falling * -0.5 * amp * (2.0 * mu - 2.0 * x) * g0_falling
                        / sigma_falling_squared;
                gradient_amplitude += neg_2_y_diff_g1_falling * g0_falling;
            }
        }

        // Regularization
        let n = data.len() as f64;
        gradient_mu += n;
        gradient_sigma_falling += n;
        gradient_sigma_rising += n;

        if let Some(constraints) = constraints {
            let penalty = self.lagrangian(data, constraints);
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

    fn loss(&self, data: &PeakFitArgs) -> f64 {
        data.iter()
            .map(|(t, i)| (i - self.density(t)).powi(2))
            .sum::<f64>()
            / data.len() as f64
            + self.regularization()
    }

    fn gradient(&self, data: &PeakFitArgs, constraints: Option<&FitConstraints>) -> Self {
        self.gradient(data, constraints)
    }
}
