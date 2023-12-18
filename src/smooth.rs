use std::collections::VecDeque;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, SubAssign};

use num_traits;
use num_traits::{Float, FromPrimitive};
use thiserror::Error;

#[derive(Debug, Clone)]
struct RingBuffer<F: Float + AddAssign, const N: usize> {
    buffer: VecDeque<F>,
}

impl<F: Float + AddAssign, const N: usize> Default for RingBuffer<F, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + AddAssign, const N: usize> RingBuffer<F, N> {
    pub fn new() -> Self {
        let mut buffer = VecDeque::new();
        buffer.extend([0.0; N].into_iter().map(|f| F::from(f).unwrap()));
        Self { buffer }
    }

    pub fn add(&mut self, value: F) -> Option<F> {
        let first = self.buffer.pop_front();
        self.buffer.push_back(value);
        first
    }

    #[allow(unused)]
    pub fn sum(&self) -> F {
        self.buffer.iter().fold(F::zero(), |acc, val| acc + *val)
    }
}

#[derive(Debug, Clone)]
struct MovingAverage<F: Float + AddAssign + SubAssign, const N: usize> {
    buffer: RingBuffer<F, N>,
    running_sum: F,
    divisor: F,
}

impl<F: Float + AddAssign + SubAssign, const N: usize> MovingAverage<F, N> {
    pub fn new() -> Self {
        let buffer = RingBuffer::new();
        let divisor = F::from(N).unwrap();
        Self {
            buffer,
            running_sum: F::zero(),
            divisor,
        }
    }

    pub fn add(&mut self, value: F) {
        self.running_sum += value;
        if let Some(last_value) = self.buffer.add(value) {
            self.running_sum -= last_value
        };
    }

    pub fn average(&self) -> F {
        self.running_sum / self.divisor
    }

    pub fn average_into(data: &[F], destination: &mut [F]) {
        let state = Self::new();
        let it = state.average_over(data.into_iter().copied());
        it.zip(destination.iter_mut()).for_each(|(a, d)| *d = a);
    }

    pub fn average_over<I: Iterator<Item = F>>(self, source: I) -> MovingAverageIter<F, N, I> {
        MovingAverageIter {
            state: self,
            source,
        }
    }
}

struct MovingAverageIter<F: Float + AddAssign + SubAssign, const N: usize, I: Iterator<Item = F>> {
    state: MovingAverage<F, N>,
    source: I,
}

impl<F: Float + AddAssign + SubAssign, const N: usize, I: Iterator<Item = F>> Iterator
    for MovingAverageIter<F, N, I>
{
    type Item = F;

    fn next(&mut self) -> Option<Self::Item> {
        self.source.next().and_then(|x| {
            self.state.add(x);
            Some(self.state.average())
        })
    }
}

pub fn moving_average<F: Float + AddAssign + SubAssign, const N: usize>(
    xdata: &[F],
    xout: &mut [F],
) {
    MovingAverage::<F, N>::average_into(xdata, xout)
}

#[derive(Debug, Clone, Copy, Error)]
pub enum SavitskyGolayError {
    #[error("The window length must be an odd number, received {0}")]
    WindowLengthNotOdd(usize),
    #[error(
        "The window length must be shorter than the data, received {0} window with {1} data points"
    )]
    WindowLengthTooLong(usize, usize),
    #[error("The polynomial order term {0} must be less than the window size {1}")]
    PolynomialOrderTooLarge(usize, usize),
    #[error("Failed to solve for coefficients: {0}")]
    FailedToSolveCoefficients(&'static str),
}

// Adapted from https://github.com/tpict/savgol-rs/
#[allow(unused)]
#[derive(Debug, Clone)]
struct SavitskyGolay<'a, F: Float> {
    data: &'a [F],
    window_length: usize,
    poly_order: usize,
    derivative: usize,
}

#[allow(unused)]
fn factorial(n: usize) -> usize {
    match n {
        0 => 1,
        1 => 1,
        _ => factorial(n - 1) * n,
    }
}

#[derive(Debug, Clone)]
pub struct Polynomial<F: Float + AddAssign + SubAssign> {
    coefficients: Vec<F>,
    order: usize,
}

impl<F: Float + AddAssign + SubAssign> Polynomial<F> {
    pub fn new(coefficients: Vec<F>, order: usize) -> Self {
        Self {
            coefficients,
            order,
        }
    }

    pub fn derivative(&self) -> Polynomial<F> {
        Polynomial::new(
            self.coefficients[1..]
                .iter()
                .enumerate()
                .map(|(i, c)| *c * F::from(i + 1).unwrap())
                .collect(),
            self.order.saturating_sub(1),
        )
    }

    pub fn derivative_to_zero(&self, derivative: usize) -> Polynomial<F> {
        (0..derivative).fold(self.clone(), |state, _| state.derivative())
    }

    pub fn iter(&self) -> std::slice::Iter<'_, F> {
        self.coefficients.iter()
    }

    pub fn eval(&self, values: &[F]) -> Vec<F> {
        values
            .iter()
            .map(|v| {
                self.iter()
                    .enumerate()
                    .fold(F::zero(), |y, (i, c)| y + *c * v.powf(F::from(i).unwrap()))
            })
            .collect()
    }
}

impl<F: Float + AddAssign + SubAssign> AsRef<[F]> for Polynomial<F> {
    fn as_ref(&self) -> &[F] {
        &self.coefficients
    }
}

#[cfg(feature = "nalgebra")]
mod nalgebra_impl {
    use nalgebra::{DMatrix, DVector};

    use super::*;

    impl<'a, F: Float + Debug + 'static + nalgebra::ComplexField + nalgebra::RealField>
        SavitskyGolay<'a, F>
    {
        fn new(
            data: &'a [F],
            window_length: usize,
            poly_order: usize,
            derivative: usize,
        ) -> Result<Self, SavitskyGolayError> {
            let inst = Self {
                data,
                window_length,
                poly_order,
                derivative,
            };
            inst.validate()?;
            Ok(inst)
        }

        fn validate(&self) -> Result<(), SavitskyGolayError> {
            let n = self.data.len();
            if self.window_length % 2 == 0 {
                Err(SavitskyGolayError::WindowLengthNotOdd(self.window_length))
            } else if self.window_length > n {
                Err(SavitskyGolayError::WindowLengthTooLong(
                    self.window_length,
                    n,
                ))
            } else if self.poly_order >= self.window_length {
                Err(SavitskyGolayError::PolynomialOrderTooLarge(
                    self.poly_order,
                    self.window_length,
                ))
            } else {
                Ok(())
            }
        }

        fn estimate_coefficients(&self) -> Result<DVector<F>, SavitskyGolayError> {
            let half_length = self.window_length / 2;
            let rem = self.window_length % 2;

            let pos = match rem {
                0 => F::from(half_length).unwrap() - F::from(0.5).unwrap(),
                _ => F::from(half_length).unwrap(),
            };
            if self.derivative > self.poly_order {
                Ok(DVector::from_element(self.window_length, F::zero()))
            } else {
                // Construct a Vandermond matrix
                let x = DVector::from_fn(self.window_length, |i, _| pos - F::from(i).unwrap());
                let order = DVector::from_fn(self.poly_order + 1, |i, _| F::from(i).unwrap());
                let vandermond =
                    DMatrix::from_fn(self.poly_order + 1, self.window_length, |i, j| {
                        Float::powf(x[j], order[i])
                    });

                let mut y = DVector::from_element(self.poly_order + 1, F::zero());
                y[self.derivative] = F::from(factorial(self.derivative)).unwrap();

                // Derived from https://github.com/strawlab/lstsq
                let svd = nalgebra::linalg::SVD::new(vandermond, true, true);
                let beta = match svd.solve(&y, F::from(1e-12).unwrap()) {
                    Ok(val) => val,
                    Err(err) => return Err(SavitskyGolayError::FailedToSolveCoefficients(err)),
                };
                Ok(beta)
            }
        }

        const fn half_length(&self) -> usize {
            self.window_length / 2
        }

        fn polyfit(&self, x: &[F], y: &[F]) -> Result<Vec<F>, SavitskyGolayError> {
            let nc = self.poly_order + 1;
            let nr = x.len();

            // Initialize system of equations for polynomial
            let mut system = DMatrix::<F>::zeros(nr, nc);
            x.iter().enumerate().for_each(|(row_i, x)| {
                system[(row_i, 0)] = F::one();
                (1..nc).for_each(|col_j| {
                    system[(row_i, col_j)] = Float::powf(*x, F::from(col_j).unwrap())
                });
            });

            let beta = DVector::from_row_slice(&y);
            let decomp = nalgebra::linalg::SVD::new(system, true, true);

            // Solve system of equations for polynomial coefficients
            let poly_coefs: Vec<_> = match decomp.solve(&beta, F::from(1e-18).unwrap()) {
                Ok(val) => val.data.into(),
                Err(e) => return Err(SavitskyGolayError::FailedToSolveCoefficients(e)),
            };
            Ok(poly_coefs)
        }

        fn fit_edge(
            &self,
            x: &DVector<F>,
            window_start: usize,
            window_stop: usize,
            interp_start: usize,
            interp_stop: usize,
            y: &mut Vec<F>,
        ) -> Result<(), SavitskyGolayError> {
            let x_edge = &x.as_slice()[window_start..window_stop];
            let y_edge: Vec<_> = (0..window_stop - window_start)
                .map(|i| F::from(i).unwrap())
                .collect();

            let poly_coefs = Polynomial::new(self.polyfit(&x_edge, &y_edge)?, self.poly_order);
            eprintln!("Initial polynomial {poly_coefs:?}");
            let poly_coefs = poly_coefs.derivative_to_zero(self.derivative);
            eprintln!("Derived polynomial {poly_coefs:?}");
            let i: Vec<_> = (0..interp_stop - interp_start)
                .map(|i| F::from(interp_start - window_start + i).unwrap())
                .collect();
            eprintln!("Evaluating polynomial");
            let values = poly_coefs.eval(&i);
            y.splice(interp_start..interp_stop, values);
            Ok(())
        }

        fn fit_edges(&self, x: &DVector<F>, y: &mut Vec<F>) -> Result<(), SavitskyGolayError>{
            eprintln!("Fitting forward edge");
            self.fit_edge(x, 0, self.window_length, self.half_length(), self.poly_order, y)?;
            let n = x.len();
            eprintln!("Fitting backward edge");
            self.fit_edge(x, n - self.window_length, n, n - self.half_length(), n, y)
        }

        fn smooth(&self) -> Result<Vec<F>, SavitskyGolayError> {
            eprintln!("Estimating coefficients");
            let coefs = self.estimate_coefficients()?;
            let x = DVector::from_vec(self.data.to_vec());
            eprintln!("Convolving kernel");
            eprintln!("Coefficients: {coefs:?}");
            // This seems to be a bottleneck
            let y = x.convolve_full(coefs);
            let padding = (y.len() - x.len()) / 2;
            let y = y.as_slice();
            let mut y = y[padding..y.len().saturating_sub(padding)].to_vec();
            eprintln!("Fitting edges");
            self.fit_edges(&x, &mut y)?;
            Ok(y)
        }
    }

    pub fn savitsky_golay<
        F: Float + Debug + 'static + nalgebra::ComplexField + nalgebra::RealField,
    >(
        data: &[F],
        window_length: usize,
        poly_order: usize,
        derivative: usize,
    ) -> Result<Vec<F>, SavitskyGolayError> {
        let state = SavitskyGolay::new(data, window_length, poly_order, derivative)?;
        state.smooth()
    }
}

#[cfg(feature = "nalgebra")]
#[deprecated = "WIP"]
pub use nalgebra_impl::savitsky_golay;
