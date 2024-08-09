#![allow(unused)]
use std::{f64::consts::PI, fmt::Debug};

use argmin::core::State;
use nalgebra::{self, dvector, DMatrix, DVector};

#[allow(unused)]
use argmin::{
    core::observers::{Observe, ObserverMode},
    core::{CostFunction, Error, Executor, Gradient, Hessian, Jacobian, Operator},
    solver::{gaussnewton::GaussNewton, neldermead::NelderMead},
};

#[derive(Debug, Default, Clone)]
struct PeakFitArgs {
    time: Vec<f64>,
    intensity: Vec<f32>,
}

impl PeakFitArgs {
    fn new(time: Vec<f64>, intensity: Vec<f32>) -> Self {
        Self { time, intensity }
    }

    fn len(&self) -> usize {
        self.time.len()
    }
}

#[derive(Debug, Clone, Copy)]
struct Gaussian {
    mu: f64,
    sigma: f64,
    amplitude: f64,
}

impl Gaussian {
    fn gaussian(&self, x: f64) -> f64 {
        self.amplitude / (2.0 * self.sigma * PI).sqrt()
            * (-0.5 * (x - self.mu).powi(2) / self.sigma.powi(2)).exp()
    }

    fn gaussian_dsigma(&self, x: f64) -> f64 {
        let a = self.gaussian(x);
        let b = (x - self.mu).powi(2) / self.sigma.powi(3);
        let c = (b - 1.0 / self.sigma);
        a * c
    }

    fn gaussian_dmu(&self, x: f64) -> f64 {
        let b = (x - self.mu).powi(2) / self.sigma.powi(2);
        let a = self.gaussian(x);
        a * b
    }

    fn gaussian_damplitude(&self, x: f64) -> f64 {
        1.0 / (2.0 * self.sigma * PI).sqrt()
            * (-0.5 * (x - self.mu).powi(2) / self.sigma.powi(2)).exp()
    }

    fn from_slice(data: &[f64]) -> Self {
        Self {
            mu: data[0],
            sigma: data[1],
            amplitude: data[2],
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ChiSquareLoss {
    m: f64,
    c: f64,
}

impl ChiSquareLoss {
    fn new(m: f64, c: f64) -> Self {
        Self { m, c }
    }

    fn evaluate(&self, x: &[f64], y: &[f64]) -> f64 {
        x.iter()
            .zip(y.iter())
            .map(|(xi, yi)| (yi - self.m * xi - self.c).powi(2))
            .sum()
    }

    fn d_m(&self, x: &[f64], y: &[f64]) -> f64 {
        -2.0 * x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| xi * (yi - self.m * xi - self.c))
            .sum::<f64>()
    }

    fn d_c(&self, x: &[f64], y: &[f64]) -> f64 {
        -2.0 * x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| yi - self.m * xi - self.c)
            .sum::<f64>()
    }
}

#[derive(Debug)]
struct GaussianPeakShapeProblem {
    data: PeakFitArgs,
}

impl GaussianPeakShapeProblem {
    fn new(data: PeakFitArgs) -> Self {
        Self { data }
    }

    fn apply(&self, param: &Gaussian) -> Vec<f64> {
        self.data
            .time
            .iter()
            .zip(self.data.intensity.iter())
            .map(|(t, i)| (*i as f64) - param.gaussian(*t))
            .collect()
    }
}

impl Operator for GaussianPeakShapeProblem {
    type Param = DVector<f64>;

    type Output = DVector<f64>;

    fn apply(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let model = Gaussian::from_slice(&param.as_slice());

        Ok(self.apply(&model).into())
    }
}

impl Jacobian for GaussianPeakShapeProblem {
    type Param = DVector<f64>;

    type Jacobian = DMatrix<f64>;

    fn jacobian(&self, param: &Self::Param) -> Result<Self::Jacobian, Error> {
        let model = Gaussian::from_slice(param.as_slice());
        let resid = self.apply(&model);
        let jac = DMatrix::from_fn(self.data.len(), 3, |si, i| {
            if i == 0 {
                resid[si].powi(2) * model.gaussian_dmu(self.data.time[si])
            } else if i == 1 {
                resid[si].powi(2) * model.gaussian_dsigma(self.data.time[si])
            } else if i == 2 {
                resid[si].powi(2) * model.gaussian_damplitude(self.data.time[si])
            } else {
                panic!("Confused")
            }
        });

        Ok(jac)
    }
}

struct Printer {}

impl<I> Observe<I> for Printer
where
    I: State + Debug,
{
    fn observe_init(
        &mut self,
        _name: &str,
        _state: &I,
        _kv: &argmin::core::KV,
    ) -> Result<(), Error> {
        eprintln!("Init:: Name: {_name}, State: {_state:?}");
        Ok(())
    }

    fn observe_iter(&mut self, _state: &I, _kv: &argmin::core::KV) -> Result<(), Error> {
        eprintln!("Step:: State: {_state:?}");
        Ok(())
    }

    fn observe_final(&mut self, _state: &I) -> Result<(), Error> {
        eprintln!("Final:: State: {_state:?}");
        Ok(())
    }
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
        let intensity = vec![
            5.14092999,
            8.01643116,
            12.27103812,
            18.43921389,
            27.19971529,
            39.38648037,
            55.98747067,
            78.12596265,
            107.01897193,
            143.90869234,
            189.96523898,
            246.16262192,
            313.13451423,
            391.02149165,
            479.32620916,
            576.79645713,
            681.35720233,
            790.11076167,
            899.4188028,
            1005.07115014,
            1102.53531942,
            1187.26885115,
            1255.06580235,
            1302.40117016,
            1326.73418216,
            1326.73418216,
            1302.40117016,
            1255.06580235,
            1187.26885115,
            1102.53531942,
            1005.07115014,
            899.4188028,
            790.11076167,
            681.35720233,
            576.79645713,
            479.32620916,
            391.02149165,
            313.13451423,
            246.16262192,
            189.96523898,
            143.90869234,
            107.01897193,
            78.12596265,
            55.98747067,
            39.38648037,
            27.19971529,
            18.43921389,
            12.27103812,
            8.01643116,
            5.14092999,
        ];

        let mu = 5.0;
        let sigma = 1.5;
        let amplitude = 5000.0;

        let measures = PeakFitArgs::new(time, intensity);
        let problem = GaussianPeakShapeProblem::new(measures);

        let initial: DVector<f64> = dvector![3.0, 1.0, 1000.0];
        let solver: GaussNewton<f64> = GaussNewton::new();

        let res = Executor::new(problem, solver)
            .configure(|state| state.param(initial).max_iters(10))
            .add_observer(Printer {}, ObserverMode::Always)
            .run().unwrap();
        eprintln!("Best solution: {:?}", res.state.best_param);
    }
}
