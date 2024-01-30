use std::fmt::Debug;

use cfg_if::cfg_if;
use num_traits;
use num_traits::{Float, FromPrimitive};

#[cfg(feature = "ndarray")]
use ndarray::Array2;
#[cfg(feature = "ndarray-linalg")]
use ndarray_linalg::Inverse;

pub fn _isclose<T>(x: T, y: T, rtol: T, atol: T) -> bool
where
    T: Float,
{
    (x - y).abs() <= (atol + rtol * y.abs())
}

pub fn isclose<T>(x: T, y: T) -> bool
where
    T: Float + FromPrimitive,
{
    _isclose(x, y, T::from_f64(1e-5).unwrap(), T::from_f64(1e-8).unwrap())
}

pub fn aboutzero<T>(x: T) -> bool
where
    T: Float + FromPrimitive,
{
    isclose(x, T::zero())
}

const MINIMUM_SIGNAL_TO_NOISE: f32 = 4.0;
const MAX_WIDTH: f64 = 1.5;

/// Approximate signal to noise ratios for  the target intensity value
pub fn approximate_signal_to_noise<Y: Float + FromPrimitive>(
    target_val: Y,
    intensity_array: &[Y],
    index: usize,
) -> Y {
    let mut min_intensity_left: Y = Y::from_f64(0.0).unwrap();
    let mut min_intensity_right: Y = Y::from_f64(0.0).unwrap();
    let n = intensity_array.len() - 1;

    if aboutzero(target_val) || index == 0 || index >= n {
        return Y::from_f64(0.0).unwrap();
    }

    let mut finished = false;
    for i in (1..=index).rev() {
        if intensity_array[i + 1] >= intensity_array[i]
            && intensity_array[i - 1] > intensity_array[i]
        {
            min_intensity_left = intensity_array[i];
            finished = true;
            break;
        }
    }
    if !finished {
        min_intensity_left = intensity_array[0];
    }

    finished = false;
    for i in index..n {
        if intensity_array[i + 1] >= intensity_array[i]
            && intensity_array[i - 1] > intensity_array[i]
        {
            min_intensity_right = intensity_array[i];
            finished = true;
            break;
        }
    }
    if !finished {
        min_intensity_right = intensity_array[n];
    }

    if aboutzero(min_intensity_left) {
        if aboutzero(min_intensity_right) {
            target_val
        } else {
            target_val / min_intensity_right
        }
    }
    else if min_intensity_right < min_intensity_left && !aboutzero(min_intensity_right) {
        target_val / min_intensity_right
    } else {
        target_val / min_intensity_left
    }
}

pub fn curve_regression<T: Float + Into<f64> + Debug, U: Float + Into<f64> + Debug>(xs: &[T], ys: &[U], n: usize, terms: &mut [f64; 2]) -> f64 {
    cfg_if! {
        if #[cfg(feature = "ndarray")] {
            return curve_regression_ndarray(xs, ys, n, terms);
        }
        else if #[cfg(feature = "nalgebra")] {
            return curve_regression_nalgebra(xs, ys, n, terms);
        }
    }
}

#[allow(non_snake_case)]
#[cfg(feature = "ndarray")]
pub fn curve_regression_ndarray<T: Float + Into<f64>, U: Float + Into<f64>>(xs: &[T], ys: &[U], n: usize, terms: &mut [f64; 2]) -> f64 {
    // Based upon
    // https://github.com/PNNL-Comp-Mass-Spec/DeconTools/blob/0a7bde357af0551bedf44368c847cc15c70c7ace/DeconTools.Backend/ProcessingTasks/Deconvoluters/HornDeconvolutor/ThrashV1/PeakProcessing/PeakStatistician.cs#L227
    let n_terms: usize = 1;

    let mut A = Array2::<f64>::zeros((2, n));
    for i in 0..n {
        A[[0, i]] = 1.0;
        for j in 1..n_terms {
            A[[j, i]] = A[[j - 1, i]] * xs[i].into();
        }
    }

    let mut Z = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        Z[[i, 0]] = 1.0f64 * ys[i].into();
    }

    let A_t = A.t();
    let A_A_t = A.dot(&A_t);
    if let Ok(iA_A_t) = A_A_t.inv() {
        let iA_A_t_A = iA_A_t.dot(&A);
        let B = iA_A_t_A.dot(&Z);

        let mut mse: f64 = 0.0;
        // This seems to be just reading out the final dimension's terms
        for i in 0..n {
            terms[0] = B[[0, 0]];
            let mut y_fit = B[[0, 0]];
            let mut x_pow: f64 = xs[i].into();
            for j in 1..n_terms {
                terms[j] = B[[j, 0]];
                y_fit += B[[j, 0]] * x_pow;
                x_pow *= xs[i].into();
            }
            mse += ys[i].into() - y_fit;
        }
        mse
    } else {
        terms.fill(0.0);
        0.0
    }
}

#[allow(non_snake_case)]
#[cfg(feature = "nalgebra")]
pub fn curve_regression_nalgebra<T: Float + Into<f64> + Debug, U: Float + Into<f64> + Debug>(xs: &[T], ys: &[U], n: usize, terms: &mut [f64; 2]) -> f64 {
    use nalgebra;
    use nalgebra::{Matrix, Dim, dimension::Dyn};
    // Based upon
    // https://github.com/PNNL-Comp-Mass-Spec/DeconTools/blob/0a7bde357af0551bedf44368c847cc15c70c7ace/DeconTools.Backend/ProcessingTasks/Deconvoluters/HornDeconvolutor/ThrashV1/PeakProcessing/PeakStatistician.cs#L227
    let n_terms: usize = 1;

    let mut A: Matrix<f64, _, _, _> = Matrix::zeros_generic(Dyn::from_usize(2usize), Dyn::from_usize(n));//::<f64>::zeros((2, n));
    for i in 0..n {
        A[(0, i)] = 1.0;
        for j in 1..n_terms {
            A[(j, i)] = A[(j - 1, i)] * xs[i].into();
        }
    }

    // let mut Z = Array2::<f64>::zeros((n, 1));
    let mut Z: Matrix<f64, _, _, _> = Matrix::zeros_generic(Dyn::from_usize(n), Dyn::from_usize(1));
    for i in 0..n {
        Z[(i, 0)] = 1.0f64 * ys[i].into();
    }

    let A_t = A.transpose();
    let A_A_t = (&A) * A_t;
    if let Some(iA_A_t) = A_A_t.try_inverse() {
        let iA_A_t_A = iA_A_t * A;
        let B = iA_A_t_A * (&Z);

        let mut mse: f64 = 0.0;
        // This seems to be just reading out the final dimension's terms
        for i in 0..n {
            terms[0] = B[(0, 0)];
            let mut y_fit = B[(0, 0)];
            let mut x_pow: f64 = xs[i].into();
            for j in 1..n_terms {
                terms[j] = B[(j, 0)];
                y_fit += B[(j, 0)] * x_pow;
                x_pow *= xs[i].into();
            }
            mse += ys[i].into() - y_fit;
        }
        mse
    } else {
        terms.fill(0.0);
        0.0
    }
}

#[derive(Default, Debug, Clone)]
pub struct WidthFit {
    pub right_width: f64,
    pub left_width: f64,
    pub full_width_at_half_max: f64,
}

/// Fits the left side of a peak
pub fn fit_rising_side_width(
    mz_array: &[f64],
    intensity_array: &[f32],
    data_index: usize,
    signal_to_noise: f32,
) -> f64 {
    let peak = intensity_array[data_index];
    let peak_half = peak / 2.0;
    let mz = mz_array[data_index];
    let mut last_y1 = peak;

    if peak == 0.0 {
        return mz
    }

    let mut upper = mz_array[0];
    for index in (0..=data_index).rev() {
        let current_mz = mz_array[index];
        let y1 = intensity_array[index];
        // We found a point below the half-max intensity threshold or the end
        // of the array, or we've gotten too far away from the centroid of the
        // peak, or the signal to noise is bad and it just looks like we're near
        // signal that will be increasing
        if (y1 < peak_half)
            || (y1 > last_y1)
            || (mz - current_mz).abs() > MAX_WIDTH
            || ((index < 1 || intensity_array[index - 1] > y1)
                && (index < 2 || intensity_array[index - 2] > y1)
                && (signal_to_noise < MINIMUM_SIGNAL_TO_NOISE))
        {
            let y2 = intensity_array[index + 1];
            let x1 = mz_array[index];
            let x2 = mz_array[index + 1];

            if !aboutzero(y2 - y1) && y1 < peak_half {
                // Use linear interpolation to find the m/z coordinate of the halfway point
                upper = x1 - (x1 - x2) * ((peak_half - y1) / (y2 - y1)) as f64;
            } else {
                // Use regression to fit the peak shape.
                // NOTE: The regression is taken from Decon2LS which is a bit
                // weird looking.
                upper = x1;
                let points = data_index - index + 1;
                if points >= 3 {
                    let iv = (data_index - points)..=data_index;
                    if intensity_array[iv.clone()]
                        .windows(2)
                        .all(|w| (w[0] - w[1]).abs() < 1e-3)
                    {
                        upper = mz;
                    } else {
                        let mut coefs = [0.0; 2];
                        curve_regression(
                            &intensity_array[iv.clone()],
                            &mz_array[iv.clone()],
                            points,
                            &mut coefs,
                        );
                        upper = coefs[1] * peak_half as f64 + coefs[0];
                    }
                }
            }
            break;
        }
        last_y1 = y1;
    }
    upper
}

/// Fits the right side of a peak
pub fn fit_falling_side_width(
    mz_array: &[f64],
    intensity_array: &[f32],
    data_index: usize,
    signal_to_noise: f32,
) -> f64 {
    let peak = intensity_array[data_index];
    let peak_half = peak / 2.0;
    let mz = mz_array[data_index];
    let n = mz_array.len() - 1;
    let mut lower = mz_array[n];
    let mut last_y1 = peak;

    if peak == 0.0 {
        return mz
    }

    for index in data_index..n {
        let current_mz = mz_array[index];
        let y1 = intensity_array[index];
        // We found a point below the half-max intensity threshold or the end
        // of the array, or we've gotten too far away from the centroid of the
        // peak, or the signal to noise is bad and it just looks like we're near
        // signal that will be increasing
        if (y1 < peak_half)
            || ((mz - current_mz).abs() > MAX_WIDTH)
            || (y1 > last_y1)
            || (((index > n - 1) || intensity_array[index + 1] > y1)
                && ((index > n - 2) || intensity_array[index + 2] > y1)
                && signal_to_noise < MINIMUM_SIGNAL_TO_NOISE)
        {
            let y2 = intensity_array[index - 1];
            let x1 = mz_array[index];
            let x2 = mz_array[index - 1];

            if !aboutzero(y2 - y1) && y1 < peak_half {
                lower = x1 - (x1 - x2) * ((peak_half - y1) / (y2 - y1)) as f64;
            } else {
                lower = x1;
                let points = index - data_index + 1;
                if points >= 3 {
                    let iv = (index - points)..=index;
                    if intensity_array[iv.clone()]
                        .windows(2)
                        .all(|w| (w[0] - w[1]).abs() < 1e-3)
                    {
                        lower = mz;
                    } else {
                        let mut coefs = [0.0; 2];
                        curve_regression(
                            &intensity_array[iv.clone()],
                            &mz_array[iv.clone()],
                            points,
                            &mut coefs,
                        );
                        lower = coefs[1] * peak_half as f64 + coefs[0];
                    }
                }
            }
            break;
        }
        last_y1 = y1;
    }
    lower
}

pub fn full_width_at_half_max(
    mz_array: &[f64],
    intensity_array: &[f32],
    data_index: usize,
    signal_to_noise: f32,
) -> WidthFit {
    let peak = intensity_array[data_index];
    let mz = mz_array[data_index];
    let mut fit = WidthFit::default();

    if aboutzero(peak) {
        return fit;
    }

    let n = mz_array.len() - 1;
    if data_index == 0 || data_index > n {
        return fit;
    }

    let rising_side_width =
        fit_rising_side_width(mz_array, intensity_array, data_index, signal_to_noise);

    fit.left_width = (mz - rising_side_width).abs();

    let falling_side_width =
        fit_falling_side_width(mz_array, intensity_array, data_index, signal_to_noise);
    fit.right_width = (falling_side_width - mz).abs();

    if aboutzero(rising_side_width) {
        fit.full_width_at_half_max = 2.0 * fit.right_width;
    } else if aboutzero(falling_side_width) {
        fit.full_width_at_half_max = 2.0 * fit.left_width;
    } else {
        fit.full_width_at_half_max = falling_side_width - rising_side_width;
    }
    fit
}

/// Fit a gaussian peak shape at `index`.
pub fn quadratic_fit(mz_array: &[f64], intensity_array: &[f32], index: usize) -> f64 {
    let n = mz_array.len() - 1;
    if index < 1 {
        mz_array[0]
    } else if index > n {
        mz_array[n]
    } else {
        let x1 = mz_array[index - 1];
        let x2 = mz_array[index];
        let x3 = mz_array[index + 1];
        let y1 = intensity_array[index - 1] as f64;
        let y2 = intensity_array[index] as f64;
        let y3 = intensity_array[index + 1] as f64;
        let d = (y2 - y1) * (x3 - x2) - (y3 - y2) * (x2 - x1);
        if aboutzero(d) {
            x2
        } else {
            // mz_fit
            ((x1 + x2) - ((y2 - y1) * (x3 - x2) * (x1 - x3)) / d) / 2.0
        }
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use crate::test_data::{read_1col, read_cols};
    use std::io;
    use std::io::prelude::*;
    use std::fs;

}