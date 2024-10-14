//! An implementation of smoothing filters and transformations
//!
use std::collections::VecDeque;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Sub, SubAssign};

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
        buffer.extend([F::zero(); N]);
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

/// Moving average over a window of size `N`
#[derive(Debug, Clone)]
pub struct MovingAverage<F: Float + AddAssign + SubAssign, const N: usize> {
    buffer: RingBuffer<F, N>,
    running_sum: F,
    divisor: F,
}

impl<F: Float + AddAssign + SubAssign, const N: usize> Default for MovingAverage<F, N> {
    fn default() -> Self {
        Self::new()
    }
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

    /// Reset the internal state to an empty buffer and 0 mean
    pub fn reset(&mut self) {
        self.buffer = RingBuffer::new();
        self.running_sum = F::zero();
    }

    /// Add `value` to the ring buffer, ejecting the last value
    pub fn add(&mut self, value: F) {
        self.running_sum += value;
        if let Some(last_value) = self.buffer.add(value) {
            self.running_sum -= last_value
        };
    }

    /// Get the current average value
    pub fn average(&self) -> F {
        self.running_sum / self.divisor
    }


    /// Average the signal in `data` into `destination` with window size `N`
    pub fn average_into(data: &[F], destination: &mut [F]) {
        let state = Self::new();
        let it = data.iter().copied();
        let it = state.average_over(it).skip(N / 2);
        it.zip(destination.iter_mut()).for_each(|(a, d)| *d = a);
    }

    /// Return an iterator that successively computes the *next* averaged item, managing
    /// the cycling of the ring buffer.
    #[must_use]
    pub fn average_over<I: Iterator<Item = F>>(self, source: I) -> MovingAverageIter<F, N, I> {
        MovingAverageIter {
            state: self,
            source,
        }
    }
}


/// An iterator used internally by [`MovingAverage::average_into`] and exposed by [`MovingAverage::average_over`]
pub struct MovingAverageIter<F: Float + AddAssign + SubAssign, const N: usize, I: Iterator<Item = F>> {
    state: MovingAverage<F, N>,
    source: I,
}

impl<F: Float + AddAssign + SubAssign, const N: usize, I: Iterator<Item = F>> Iterator
    for MovingAverageIter<F, N, I>
{
    type Item = F;

    fn next(&mut self) -> Option<Self::Item> {
        self.source.next().map(|x| {
            self.state.add(x);
            self.state.average()
        })
    }
}

/// Compute a moving average over a window of size `N` on either side of each point using
/// a compile time-known window size.
pub fn moving_average<F: Float + AddAssign + SubAssign, const N: usize>(
    xdata: &[F],
    xout: &mut [F],
) {
    MovingAverage::<F, N>::average_into(xdata, xout)
}


/// Compute a moving average over a window of size `N` on either side of each point.
/// This uses dynamic dispatch to a fixed window size implementation. This maxes out
/// at 20 currently.
pub fn moving_average_dyn<F: Float + AddAssign + SubAssign>(xdata: &[F], xout: &mut [F], size: usize) {
    match size {
        1 => moving_average::<F, 1>(xdata, xout),
        2 => moving_average::<F, 1>(xdata, xout),
        3 => moving_average::<F, 3>(xdata, xout),
        4 => moving_average::<F, 4>(xdata, xout),
        5 => moving_average::<F, 5>(xdata, xout),
        6 => moving_average::<F, 6>(xdata, xout),
        7 => moving_average::<F, 7>(xdata, xout),
        8 => moving_average::<F, 8>(xdata, xout),
        9 => moving_average::<F, 9>(xdata, xout),
        10 => moving_average::<F, 10>(xdata, xout),
        11 => moving_average::<F, 11>(xdata, xout),
        12 => moving_average::<F, 12>(xdata, xout),
        13 => moving_average::<F, 13>(xdata, xout),
        14 => moving_average::<F, 14>(xdata, xout),
        15 => moving_average::<F, 15>(xdata, xout),
        16 => moving_average::<F, 16>(xdata, xout),
        17 => moving_average::<F, 17>(xdata, xout),
        18 => moving_average::<F, 18>(xdata, xout),
        19 => moving_average::<F, 19>(xdata, xout),
        20 => moving_average::<F, 20>(xdata, xout),
        _ => moving_average::<F, 20>(xdata, xout)
    }
}

/// All the ways a Savitsky-Golay filter can go wrong
#[derive(Debug, Clone, Error)]
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
    FailedToSolveCoefficients(String),
}


/// An opaque implementation of a non-negative [Savitsky-Golay](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter) filter.
/// The implementation details depend upon which linear algebra backend is enabled.
/// Implementation details were adapted from `<https://github.com/tpict/savgol-rs/>`
#[allow(unused)]
#[derive(Debug, Clone)]
pub struct SavitskyGolay<F: Float> {
    pub window_length: usize,
    pub poly_order: usize,
    pub derivative: usize,
    coefs: Vec<F>,
}

fn factorial(n: usize) -> usize {
    match n {
        0 => 1,
        1 => 1,
        _ => factorial(n - 1) * n,
    }
}


/// A basic opaque polynomial for Savitsky-Golay
#[derive(Debug, Clone)]
pub struct Polynomial<F: Float + AddAssign + SubAssign> {
    /// The coefficients of each polynomial
    pub coefficients: Vec<F>,
    /// The number of polynomial terms
    pub order: usize,
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

#[cfg(feature = "ndarray")]
mod ndarray_impl {
    use ndarray::{Array, Array1, Array2};
    use ndarray_linalg::{Lapack, Scalar, Solve, SVD};

    use super::*;

    impl<F: Float + Debug + 'static + Lapack + Scalar> SavitskyGolay<F> {
        pub fn new(
            window_length: usize,
            poly_order: usize,
            derivative: usize,
        ) -> Result<Self, SavitskyGolayError> {
            let mut inst = Self {
                window_length,
                poly_order,
                derivative,
                coefs: Vec::new(),
            };
            inst.validate()?;
            inst.coefs = inst.estimate_coefficients()?;
            Ok(inst)
        }

        fn validate(&self) -> Result<(), SavitskyGolayError> {
            if self.window_length % 2 == 0 {
                Err(SavitskyGolayError::WindowLengthNotOdd(self.window_length))
            } else if self.poly_order >= self.window_length {
                Err(SavitskyGolayError::PolynomialOrderTooLarge(
                    self.poly_order,
                    self.window_length,
                ))
            } else {
                Ok(())
            }
        }

        fn estimate_coefficients(&self) -> Result<Vec<F>, SavitskyGolayError> {
            let half_length = self.window_length / 2;
            let rem = self.window_length % 2;

            let pos = match rem {
                0 => F::from(half_length).unwrap() - F::from(0.5).unwrap(),
                _ => F::from(half_length).unwrap(),
            };
            if self.derivative > self.poly_order {
                let mut coefs = Vec::with_capacity(self.window_length);
                coefs.resize(self.window_length, F::zero());
                Ok(coefs)
            } else {
                // There is an error in here somewhere

                // Construct a Vandermond matrix
                let x = Array1::from_iter((0..self.window_length).into_iter().map(|i| pos - F::from(i).unwrap()));
                let order = Array1::from_iter((0..(self.poly_order + 1)).into_iter().map(|i| F::from(i).unwrap()));

                let vandermond = Array2::from_shape_fn((self.poly_order + 1, self.window_length), |(i, j)| {
                    Float::powf(x[j], order[i])
                });

                let mut y = Array1::from_shape_fn(self.poly_order + 1, |_| F::zero());
                y[self.derivative] = F::from(factorial(self.derivative)).unwrap();


                let (u, _s, _v) = vandermond.svd(true, true).unwrap();
                let u = u.unwrap();
                let beta = u.dot(&u.t()).dot(&y);

                Ok(beta.to_vec())
            }
        }

        const fn half_length(&self) -> usize {
            self.window_length / 2
        }

        fn polyfit(&self, x: &[F], y: &[F]) -> Result<Vec<F>, SavitskyGolayError> {
            let nc = self.poly_order + 1;
            let nr = x.len();

            // Initialize system of equations for polynomial
            let mut system = Array2::zeros((nr, nc));
            x.iter().enumerate().for_each(|(row_i, x)| {
                system[(row_i, 0)] = F::one();
                (1..nc).for_each(|col_j| {
                    system[(row_i, col_j)] = Float::powf(*x, F::from(col_j).unwrap())
                });
            });

            let beta = Array1::from_iter(y.iter().copied());

            let (u, _s, _v) = system.svd(true, true).unwrap();
            let u = u.unwrap();

            let poly_coefs = u.dot(&u.t()).dot(&beta);
            Ok(poly_coefs.to_vec())
        }

        fn fit_edge(
            &self,
            x: &Array1<F>,
            window_start: usize,
            window_stop: usize,
            interp_start: usize,
            interp_stop: usize,
            y: &mut Vec<F>,
        ) -> Result<(), SavitskyGolayError> {
            let x_edge = &x.as_slice().unwrap()[window_start..window_stop];
            let y_edge: Vec<_> = (0..window_stop - window_start)
                .map(|i| F::from(i).unwrap())
                .collect();

            let poly_coefs = Polynomial::new(self.polyfit(x_edge, &y_edge)?, self.poly_order);
            let poly_coefs = poly_coefs.derivative_to_zero(self.derivative);
            let i: Vec<_> = (0..interp_stop - interp_start)
                .map(|i| F::from(interp_start - window_start + i).unwrap())
                .collect();
            let values = poly_coefs.eval(&i);
            y.splice(interp_start..interp_stop, values);
            Ok(())
        }

        fn fit_edges(&self, x: &Array1<F>, y: &mut Vec<F>) -> Result<(), SavitskyGolayError> {
            self.fit_edge(
                x,
                0,
                self.window_length,
                self.half_length(),
                self.poly_order,
                y,
            )?;
            let n = x.len();
            self.fit_edge(x, n - self.window_length, n, n - self.half_length(), n, y)
        }

        fn convolve(&self, x: &Array1<F>, coefs: &Array1<F>) -> Array1<F> {
            let vec = x.len();
            let ker = coefs.len();

            let result_len = (x.shape()[0] + coefs.shape()[0]) - 1;
            let mut conv = Array1::<F>::zeros([result_len]);
            for i in 0..vec + ker - 1 {
                let u_i = if i > vec { i - ker } else { 0 };
                let u_f = i.min(vec - 1);
                if u_i == u_f {
                    conv[i] += x[u_i] * coefs[i - u_i];
                } else {
                    for u in u_i..u_f + 1 {
                        if i - u < ker {
                            conv[i] += x[u] * coefs[i - u];
                        }
                    }
                }
            }
            conv
        }

        pub fn smooth(&self, data: &[F]) -> Result<Vec<F>, SavitskyGolayError> {
            let n = data.len();
            if self.window_length > n {
                return Err(SavitskyGolayError::WindowLengthTooLong(
                    self.window_length,
                    n,
                ));
            }
            let coefs = Array1::from_iter(self.coefs.iter().copied());
            let x = Array1::from_iter(data.iter().copied());
            // This seems to be a bottleneck
            let y = self.convolve(&x, &coefs);
            let padding = (y.len() - x.len()) / 2;
            let y = y.as_slice().unwrap();
            let mut y = y[padding..y.len().saturating_sub(padding)].to_vec();
            self.fit_edges(&x, &mut y)?;
            let zero = F::zero();
            y.iter_mut().for_each(|y| {
                if *y < zero {
                    *y = zero;
                }
            });
            Ok(y)
        }
    }

    /// A wrapper around [`SavitskyGolay`] for the [`ndarray`] backend.
    pub fn savitsky_golay<
        F: Float + Debug + 'static + Lapack + Scalar,
    >(
        data: &[F],
        window_length: usize,
        poly_order: usize,
        derivative: usize,
    ) -> Result<Vec<F>, SavitskyGolayError> {
        let state = SavitskyGolay::new(window_length, poly_order, derivative)?;
        state.smooth(data)
    }
}

#[cfg(feature = "nalgebra")]
mod nalgebra_impl {
    use nalgebra::{DMatrix, DVector};

    use super::*;

    impl<F: Float + Debug + 'static + nalgebra::ComplexField + nalgebra::RealField> SavitskyGolay<F> {
        pub fn new(
            window_length: usize,
            poly_order: usize,
            derivative: usize,
        ) -> Result<Self, SavitskyGolayError> {
            let mut inst = Self {
                window_length,
                poly_order,
                derivative,
                coefs: Vec::new(),
            };
            inst.validate()?;
            inst.coefs = inst.estimate_coefficients()?;
            Ok(inst)
        }

        fn validate(&self) -> Result<(), SavitskyGolayError> {
            if self.window_length % 2 == 0 {
                Err(SavitskyGolayError::WindowLengthNotOdd(self.window_length))
            } else if self.poly_order >= self.window_length {
                Err(SavitskyGolayError::PolynomialOrderTooLarge(
                    self.poly_order,
                    self.window_length,
                ))
            } else {
                Ok(())
            }
        }

        fn estimate_coefficients(&self) -> Result<Vec<F>, SavitskyGolayError> {
            let half_length = self.window_length / 2;
            let rem = self.window_length % 2;

            let pos = match rem {
                0 => F::from(half_length).unwrap() - F::from(0.5).unwrap(),
                _ => F::from(half_length).unwrap(),
            };
            if self.derivative > self.poly_order {
                Ok(DVector::from_element(self.window_length, F::zero()).into_iter().copied().collect())
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
                    Err(err) => return Err(SavitskyGolayError::FailedToSolveCoefficients(err.to_string())),
                };
                Ok(beta.into_iter().copied().collect())
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

            let beta = DVector::from_row_slice(y);
            let decomp = nalgebra::linalg::SVD::new(system, true, true);

            // Solve system of equations for polynomial coefficients
            let poly_coefs: Vec<_> = match decomp.solve(&beta, F::from(1e-18).unwrap()) {
                Ok(val) => val.data.into(),
                Err(e) => return Err(SavitskyGolayError::FailedToSolveCoefficients(e.to_string())),
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

            let poly_coefs = Polynomial::new(self.polyfit(x_edge, &y_edge)?, self.poly_order);
            let poly_coefs = poly_coefs.derivative_to_zero(self.derivative);
            let i: Vec<_> = (0..interp_stop - interp_start)
                .map(|i| F::from(interp_start - window_start + i).unwrap())
                .collect();
            let values = poly_coefs.eval(&i);
            y.splice(interp_start..interp_stop, values);
            Ok(())
        }

        fn fit_edges(&self, x: &DVector<F>, y: &mut Vec<F>) -> Result<(), SavitskyGolayError> {
            self.fit_edge(
                x,
                0,
                self.window_length,
                self.half_length(),
                self.poly_order,
                y,
            )?;
            let n = x.len();
            self.fit_edge(x, n - self.window_length, n, n - self.half_length(), n, y)
        }

        pub fn smooth(&self, data: &[F]) -> Result<Vec<F>, SavitskyGolayError> {
            let n = data.len();
            if self.window_length > n {
                return Err(SavitskyGolayError::WindowLengthTooLong(
                    self.window_length,
                    n,
                ));
            }
            let coefs = DVector::from_row_slice(&self.coefs);
            let x = DVector::from_row_slice(data);
            // This seems to be a bottleneck
            let y = x.convolve_full(coefs);
            let padding = (y.len() - x.len()) / 2;
            let y = y.as_slice();
            let mut y = y[padding..y.len().saturating_sub(padding)].to_vec();
            self.fit_edges(&x, &mut y)?;
            let zero = F::zero();
            y.iter_mut().for_each(|y| {
                if *y < zero {
                    *y = zero;
                }
            });
            Ok(y)
        }
    }

    /// A wrapper around [`SavitskyGolay`] for the [`nalgebra`] backend.
    pub fn savitsky_golay<
        F: Float + Debug + 'static + nalgebra::ComplexField + nalgebra::RealField,
    >(
        data: &[F],
        window_length: usize,
        poly_order: usize,
        derivative: usize,
    ) -> Result<Vec<F>, SavitskyGolayError> {
        let state = SavitskyGolay::new(window_length, poly_order, derivative)?;
        state.smooth(data)
    }
}

#[cfg(feature = "nalgebra")]
pub use nalgebra_impl::savitsky_golay;

#[cfg(not(feature = "nalgebra"))]
#[cfg(feature = "ndarray")]
pub use ndarray_impl::savitsky_golay;

#[cfg(test)]
mod test {
    use super::*;
    use crate::test_data::{NOISE, X, Y};

    use std::{
        fs,
        io::{self, Write},
    };

    #[cfg(feature = "nalgebra")]
    #[test]
    fn test_savgol() -> io::Result<()> {
        let actual_y: Vec<_> = Y
            .iter()
            .zip(NOISE.iter())
            .map(|(y, o)| *y * 1000.0 + *o)
            .collect();

        let filter = SavitskyGolay::<f32>::new(5, 3, 0).unwrap();
        let smoothed = filter.smooth(&actual_y).unwrap();

        match fs::create_dir("tmp") { Ok(_) => {}, Err(_) => {} };

        let mut coeffh = fs::File::create("tmp/coefs.txt")?;
        filter.coefs.iter().for_each(|y| {
            coeffh.write(format!("{}\n", *y).as_bytes()).unwrap();
        });

        let mut rawfh = fs::File::create("tmp/raw.txt")?;
        actual_y.iter().for_each(|y| {
            rawfh.write(format!("{}\n", *y).as_bytes()).unwrap();
        });
        let mut rawfh = fs::File::create("tmp/savgol.txt")?;
        smoothed.iter().for_each(|y| {
            rawfh.write(format!("{}\n", *y).as_bytes()).unwrap();
        });

        let diff: f32 = actual_y
            .iter()
            .zip(smoothed.iter())
            .map(|(y, y2)| (*y - *y2).abs())
            .sum::<f32>()
            / smoothed.len() as f32;
        eprintln!("Difference {}", diff);

        assert!(
            (diff - 1761.9003).abs() < 1e-3,
            "Difference {diff} was {} (too large)",
            (diff - 1761.9003).abs()
        );
        Ok(())
    }
}
