use libm::erf;
use std::{
    borrow::Cow,
    f64::consts::{PI, SQRT_2},
    fmt::Debug,
    iter::FusedIterator,
};

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

#[derive(Debug, Default, Clone)]
pub struct PeakFitArgs<'a, 'b> {
    pub time: Cow<'a, [f64]>,
    pub intensity: Cow<'b, [f32]>,
}

impl<'a, 'b> PeakFitArgs<'a, 'b> {
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

#[derive(Debug, Clone)]
pub struct FitConfig {
    max_iter: usize,
    learning_rate: f64,
    convergence: f64,
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
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            max_iter: 50_000,
            learning_rate: 1e-3,
            convergence: 1e-6,
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
    type ModelType: PeakShapeModel;

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

    fn fit_model(&self, model_params: &mut Self::ModelType, config: FitConfig) -> ModelFitResult {
        let mut params = model_params.clone();

        let mut last_loss = f64::INFINITY;
        let mut best_loss = f64::INFINITY;
        let mut best_params = model_params.clone();
        let mut iters = 0;
        let mut converged = false;
        let mut success = true;

        for it in 0..config.max_iter {
            iters = it;
            let loss = self.loss(&params);
            let gradient = self.gradient(&params);

            params.gradient_update(gradient, config.learning_rate);
            if loss < best_loss {
                best_loss = loss;
                best_params = params.clone();
            }

            if (last_loss - loss).abs() < config.convergence {
                converged = true;
                break;
            }
            last_loss = loss;

            if loss.is_nan() || loss.is_infinite() {
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

    fn residuals<'a, 'b>(&self, data: &PeakFitArgs<'a, 'b>) -> PeakFitArgs<'a, 'b> {
        let mut data = data.borrow();
        for (yhat, y) in self.predict_iter(data.time.iter().copied()).zip(data.intensity.to_mut().iter_mut()) {
            *y -= yhat as f32;
            if *y < 0.0 {
                *y =  0.0;
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

    fn fit(&mut self, args: PeakFitArgs) -> ModelFitResult {
        self.fit_with(args, Default::default())
    }

    fn fit_with(&mut self, args: PeakFitArgs, config: FitConfig) -> ModelFitResult {
        let fitter = Self::Fitter::from_args(args);
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

    pub fn gradient_update(&mut self, gradient: Self, learning_rate: f64) {
        self.mu -= gradient.mu * learning_rate;
        self.sigma -= gradient.sigma * learning_rate;
        self.amplitude -= gradient.amplitude * learning_rate;
    }
}

impl PeakShapeModel for GaussianPeakShape {
    type Fitter<'a, 'b> = GaussianPeakShapeFitter<'a, 'b>;

    fn density(&self, x: f64) -> f64 {
        self.density(x)
    }

    fn gradient_update(&mut self, gradient: Self, learning_rate: f64) {
        self.gradient_update(gradient, learning_rate);
    }
}

impl<'a, 'b> PeakShapeModelFitter<'a, 'b> for GaussianPeakShapeFitter<'a, 'b> {
    type ModelType = GaussianPeakShape;

    fn from_args(args: PeakFitArgs<'a, 'b>) -> Self {
        Self::new(args)
    }

    fn gradient(&self, params: &Self::ModelType) -> Self::ModelType {
        self.gradient(params)
    }

    fn loss(&self, params: &Self::ModelType) -> f64 {
        self.loss(params)
    }

    fn data(&self) -> &PeakFitArgs {
        &self.data
    }
}

#[derive(Debug)]
pub struct GaussianPeakShapeFitter<'a, 'b> {
    pub data: PeakFitArgs<'a, 'b>,
}

impl<'a, 'b> GaussianPeakShapeFitter<'a, 'b> {
    pub fn new(data: PeakFitArgs<'a, 'b>) -> Self {
        Self { data }
    }

    pub fn loss(&self, model: &GaussianPeakShape) -> f64 {
        let loss = self
            .iter()
            .map(|(t, y)| (y - model.density(t)).powf(2.0))
            .sum::<f64>()
            + (model.mu + model.sigma);
        loss / self.data.len() as f64
    }

    pub fn gradient(&self, model: &GaussianPeakShape) -> GaussianPeakShape {
        let mut g = [
            self.mu_gradient(&model),
            self.sigma_gradient(&model),
            self.amplitude_gradient(&model),
        ];
        let gradnorm: f64 = g.iter().map(|f| f.abs()).sum::<f64>() / g.len() as f64;
        if gradnorm > 1.0 {
            g[0] /= gradnorm;
            g[1] /= gradnorm;
        }
        GaussianPeakShape::from_slice(&g)
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn iter(&self) -> PeakFitArgsIter<'_> {
        self.data.iter()
    }

    pub fn apply(&self, param: &GaussianPeakShape) -> Vec<f64> {
        self.data.iter().map(|(t, _i)| param.density(t)).collect()
    }

    pub fn mu_gradient(&self, param: &GaussianPeakShape) -> f64 {
        let amp = param.amplitude;
        let mu = param.mu;
        let sigma = param.sigma;

        let two_mu = 2.0 * mu;
        let sigma_squared = sigma.powi(2);

        let grad: f64 = self
            .data
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

        grad / self.data.len() as f64
    }

    pub fn sigma_gradient(&self, param: &GaussianPeakShape) -> f64 {
        let amp = param.amplitude;
        let mu = param.mu;
        let sigma = param.sigma;

        let sigma_squared = sigma.powi(2);
        let sigma_cubed = sigma.powi(3);

        let grad: f64 = self
            .data
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

        grad / self.data.len() as f64
    }

    pub fn amplitude_gradient(&self, param: &GaussianPeakShape) -> f64 {
        let amp = param.amplitude;
        let mu = param.mu;
        let sigma = param.sigma;

        let sigma_squared = sigma.powi(2);

        let grad: f64 = self
            .data
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

        grad / self.data.len() as f64
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

    pub fn to_vec(&self) -> Vec<f64> {
        vec![self.mu, self.sigma, self.amplitude, self.lambda]
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
    type Fitter<'a, 'b> = SkewedGaussianPeakShapeFitter<'a, 'b>;

    fn density(&self, x: f64) -> f64 {
        self.density(x)
    }

    fn gradient_update(&mut self, gradient: Self, learning_rate: f64) {
        self.gradient_update(gradient, learning_rate);
    }
}

#[derive(Debug)]
pub struct SkewedGaussianPeakShapeFitter<'a, 'b> {
    pub data: PeakFitArgs<'a, 'b>,
}

impl<'a, 'b> SkewedGaussianPeakShapeFitter<'a, 'b> {
    pub fn new(data: PeakFitArgs<'a, 'b>) -> Self {
        Self { data }
    }

    pub fn loss(&self, model: &SkewedGaussianPeakShape) -> f64 {
        let loss = self
            .iter()
            .map(|(t, y)| (y - model.density(t)).powi(2))
            .sum::<f64>()
            + (model.mu + model.sigma);
        loss / self.data.len() as f64
    }

    pub fn gradient(&self, model: &SkewedGaussianPeakShape) -> SkewedGaussianPeakShape {
        let mut g = [
            self.mu_gradient(&model),
            self.sigma_gradient(&model),
            self.amplitude_gradient(&model),
            self.lambda_gradient(&model),
        ];
        let gradnorm: f64 = g.iter().map(|f| f.abs()).sum::<f64>() / g.len() as f64;
        if gradnorm > 1.0 {
            g[0] /= gradnorm;
            g[1] /= gradnorm;
            g[3] /= gradnorm;
        }
        SkewedGaussianPeakShape::from_slice(&g)
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn iter(&self) -> PeakFitArgsIter<'_> {
        self.data.iter()
    }

    pub fn apply(&self, param: &SkewedGaussianPeakShape) -> Vec<f64> {
        self.data.iter().map(|(t, _i)| param.density(t)).collect()
    }

    pub fn mu_gradient(&self, param: &SkewedGaussianPeakShape) -> f64 {
        let amp = param.amplitude;
        let mu = param.mu;
        let sigma = param.sigma;
        let lam = param.lambda;

        let two_sigma = sigma * 2.0;
        let sigma_square = sigma.powi(2);
        let skew = 2.0 * 1.4142135623731 * amp * lam;
        let sqrt_2_lam = SQRT_2 * lam;
        let sqrt_pi_sigma = PI.sqrt() * sigma;
        let neg_half_lam_squared = -1_f64 / 2.0 * lam.powi(2);

        let mut grad = 0.0;
        for (x, y) in self.data.iter() {
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
        grad / self.data.len() as f64
    }

    pub fn sigma_gradient(&self, param: &SkewedGaussianPeakShape) -> f64 {
        let amp = param.amplitude;
        let mu = param.mu;
        let sigma = param.sigma;
        let lam = param.lambda;

        let two_sigma = sigma * 2.0;
        let sigma_square = sigma.powi(2);
        let sigma_cubed = sigma.powi(3);
        let skew = 2.0 * 1.4142135623731 * amp * lam;
        let sqrt_2_lam = SQRT_2 * lam;
        let sqrt_pi_sigma_square = PI.sqrt() * sigma_square;
        let neg_half_lam_squared = -1_f64 / 2.0 * lam.powi(2);

        let mut grad = 0.0;
        for (x, y) in self.data.iter() {
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
        grad / self.data.len() as f64
    }

    pub fn amplitude_gradient(&self, param: &SkewedGaussianPeakShape) -> f64 {
        let amp = param.amplitude;
        let mu = param.mu;
        let sigma = param.sigma;
        let lam = param.lambda;

        let two_sigma = sigma * 2.0;
        let sigma_square = sigma.powi(2);
        let sqrt_2_lam = SQRT_2 * lam;

        let mut grad = 0.0;
        for (x, y) in self.data.iter() {
            grad += -2.0
                * (-amp
                    * (erf(sqrt_2_lam * (-mu + x) / (two_sigma)) + 1.0)
                    * (-0.5 * (-mu + x).powi(2) / sigma_square).exp()
                    + y)
                * (erf(sqrt_2_lam * (-mu + x) / (two_sigma)) + 1.0)
                * (-0.5 * (-mu + x).powi(2) / sigma_square).exp()
        }
        grad / self.data.len() as f64
    }

    pub fn lambda_gradient(&self, param: &SkewedGaussianPeakShape) -> f64 {
        let amp = param.amplitude;
        let mu = param.mu;
        let sigma = param.sigma;
        let lam = param.lambda;

        let two_sigma = sigma * 2.0;
        let sigma_square = sigma.powi(2);
        let delta_skew = -2.0 * 1.4142135623731 * amp;
        let sqrt_2_lam = SQRT_2 * lam;
        let sqrt_pi_sigma = PI.sqrt() * sigma;
        let neg_half_lam_squared = -1_f64 / 2.0 * lam.powi(2);

        let mut grad = 0.0;
        for (x, y) in self.data.iter() {
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
        grad / self.data.len() as f64
    }
}

impl<'a, 'b> PeakShapeModelFitter<'a, 'b> for SkewedGaussianPeakShapeFitter<'a, 'b> {
    type ModelType = SkewedGaussianPeakShape;

    fn from_args(args: PeakFitArgs<'a, 'b>) -> Self {
        Self::new(args)
    }

    fn gradient(&self, params: &Self::ModelType) -> Self::ModelType {
        self.gradient(params)
    }

    fn loss(&self, params: &Self::ModelType) -> f64 {
        self.loss(params)
    }

    fn data(&self) -> &PeakFitArgs {
        &self.data
    }
}

#[derive(Debug, Clone, Copy)]
pub enum PeakShape {
    Gaussian(GaussianPeakShape),
    SkewedGaussian(SkewedGaussianPeakShape),
}



#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_fit() {
        let time = vec![
            0., 0.20408163, 0.40816327, 0.6122449, 0.81632653, 1.02040816, 1.2244898, 1.42857143,
            1.63265306, 1.83673469, 2.04081633, 2.24489796, 2.44897959, 2.65306122, 2.85714286,
            3.06122449, 3.26530612, 3.46938776, 3.67346939, 3.87755102, 4.08163265, 4.28571429,
            4.48979592, 4.69387755, 4.89795918, 5.10204082, 5.30612245, 5.51020408, 5.71428571,
            5.91836735, 6.12244898, 6.32653061, 6.53061224, 6.73469388, 6.93877551, 7.14285714,
            7.34693878, 7.55102041, 7.75510204, 7.95918367, 8.16326531, 8.36734694, 8.57142857,
            8.7755102, 8.97959184, 9.18367347, 9.3877551, 9.59183673, 9.79591837, 10.,
        ];

        let mu = 5.0;
        let sigma = 1.5;
        let amplitude = 5000.0;

        let ref_model = GaussianPeakShape {
            mu,
            sigma,
            amplitude,
        };

        let intensity: Vec<_> = time.iter().map(|t| ref_model.density(*t) as f32).collect();

        let measures = PeakFitArgs::from((time, intensity.clone()));

        let initial = GaussianPeakShape::from_slice(&[4.0, 1.0, 1.0]);

        let mut params = initial.clone();
        let result = params.fit_with(measures.borrow(), FitConfig::default().max_iter(50_000));
        let score = params.score(&measures);

        eprintln!(
            "Final Params: {params:?}; Loss = {:0.3}, Score = {score:0.3}, Its = {}",
            result.loss, result.iterations
        )
    }

    #[test]
    fn test_skewed_fit() {
        let times = vec![
            0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,
            1.8, 1.9, 2., 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3., 3.1, 3.2, 3.3, 3.4, 3.5,
            3.6, 3.7, 3.8, 3.9, 4., 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5., 5.1, 5.2, 5.3,
            5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6., 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7., 7.1,
            7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8., 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8,
            8.9, 9., 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10., 10.1, 10.2, 10.3, 10.4,
            10.5, 10.6, 10.7, 10.8, 10.9, 11., 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8,
            11.9, 12., 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 13., 13.1, 13.2, 13.3,
            13.4, 13.5, 13.6, 13.7, 13.8, 13.9, 14., 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7,
            14.8, 14.9,
        ];

        let reference = SkewedGaussianPeakShape {
            mu: 4.0,
            sigma: 1.5,
            amplitude: 1500.0,
            lambda: 2.5,
        };

        let intensities: Vec<_> = times
            .iter()
            .copied()
            .map(|x| reference.density(x) as f32)
            .collect();

        let measures = PeakFitArgs::from((times.clone(), intensities.clone()));

        let initial = SkewedGaussianPeakShape::from_slice(&[4.0, 1.0, 1.0, 1.0]);
        let mut params = initial.clone();
        let result = params.fit_with(measures.borrow(), FitConfig::default().max_iter(50_000));
        let score = params.score(&measures);

        eprintln!(
            "Final Params: {params:?}; Loss = {:0.3}, Score = {score:0.3}, Its = {}",
            result.loss, result.iterations
        )
    }
}
