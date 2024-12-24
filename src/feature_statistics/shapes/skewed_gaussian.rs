use std::f64::consts::{PI, SQRT_2};

use libm::erf;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};


use crate::peak_statistics::{
    fit_falling_side_width, fit_rising_side_width,
};

use super::{FitConstraints, PeakFitArgs, PeakShapeFitter, PeakShapeModel};

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

        // Estimate a good starting value of sigma
        let mut falling_width = fit_falling_side_width(&data.time, &data.intensity, idx, 1.0);
        let mut rising_width = fit_rising_side_width(&data.time, &data.intensity, idx, 1.0);
        let total_width = data.duration();

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

        // Create an initial state
        let mut this = Self::new(mu, sigma, amplitude, 1.0);

        // Search for a good initial skewing lambda initial parameter
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
    fn regularization(&self) -> f64 {
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

    #[inline]
    pub fn jacobian_iter<'a>(
        &'a self,
        data: &'a PeakFitArgs,
    ) -> impl Iterator<Item = (f64, f64, f64, f64)> + 'a {
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

        data.iter().map(move |(x, y)| {
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

            let dmu = delta_y
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

            let dsigma = delta_y
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

            let dlambda = delta_skew
                * mu_sub_x
                * delta_y
                * neg_half_mu_sub_x_squared_div_sigma_squared_exp
                * neg_half_lam_squared_mu_sub_x_squared_div_sigma_square_exp
                / sqrt_pi_sigma
                + 1.0;

            let damplitude = -2.0
                * delta_y
                * erf_sqrt_2_lam_mu_sub_x_div_two_sigma_plus_one
                * neg_half_mu_sub_x_squared_div_sigma_squared_exp;

            (dmu, dsigma, dlambda, damplitude)
        })
    }

    /// Compute the gradient of the loss function for parameter optimization.
    pub fn gradient(&self, data: &PeakFitArgs) -> Self {
        let (gradient_mu, gradient_sigma, gradient_lambda, gradient_amplitude) = self
            .jacobian_iter(data)
            .reduce(|acc, point| {
                (
                    acc.0 + point.0,
                    acc.1 + point.1,
                    acc.2 + point.2,
                    acc.3 + point.3,
                )
            })
            .unwrap();

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
    fn loss(&self, data: &PeakFitArgs, _constraints: Option<&FitConstraints>) -> f64 {
        data.iter()
            .map(|(t, i)| (i - self.density(t)).powi(2))
            .sum::<f64>()
            / data.len() as f64
            + self.regularization()
    }
}
