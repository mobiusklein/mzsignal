use num_traits::{Float, FromPrimitive, ToPrimitive, Zero};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::arrayops::minmax;

pub fn percentile<T: Float + ToPrimitive>(values: &[T], percent: f64) -> T {
    let k = (values.len() - 1) as f64 * percent;
    let f = k.floor();
    let c = k.ceil();
    if (f - c).abs() < 1e-6 {
        return values[k as usize];
    }
    let d0 = values[f as usize] * T::from(c - k).unwrap();
    let d1 = values[c as usize] * T::from(k - f).unwrap();
    d0 + d1
}

pub fn freedman_diaconis_bin_width<T: Float + ToPrimitive>(values: &[T]) -> f64 {
    let q75 = percentile(values, 0.75);
    let q25 = percentile(values, 0.25);
    let iqr = (q75 - q25).to_f64().unwrap();
    2.0 * iqr * (values.len() as f64).powf(-1.0 / 3.0)
}

pub fn sturges_bin_width<T: Float + ToPrimitive>(values: &[T]) -> f64 {
    let d = (values.len() as f64 + 1.0).log2();
    let (min, max) = minmax(values);
    (max - min).to_f64().unwrap() / d
}

#[derive(Default, Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Histogram<T: Float + Default + FromPrimitive> {
    pub bin_count: Vec<usize>,
    pub bin_edges: Vec<T>,
}

impl<T: Float + Default + FromPrimitive> Histogram<T> {
    pub fn new(values: &[T], bins: usize) -> Histogram<T> {
        let mut hist = Histogram::default();
        hist.populate(values, bins);
        hist
    }

    fn from_values_and_width(values: &[T], width: f64) -> Self {
        if width.is_zero() {
            return Histogram::new(values, 1)
        }
        let (min, max) = minmax(values);
        let bins = ((max - min).to_f64().unwrap() / width).to_usize().unwrap() + 1;
        Self::new(values, bins)
    }

    pub fn new_freedman_diaconis(values: &[T]) -> Histogram<T> {
        let width = freedman_diaconis_bin_width(values);
        Self::from_values_and_width(values, width)
    }

    pub fn new_sturges(values: &[T]) -> Histogram<T> {
        let width = sturges_bin_width(values);
        Self::from_values_and_width(values, width)
    }

    pub fn clear(&mut self) {
        self.bin_count.clear();
        self.bin_edges.clear();
    }

    pub fn populate(self: &mut Histogram<T>, values: &[T], bins: usize) {
        let (mut min, mut max) = minmax(values);
        if min == max {
            min = min - T::from(0.5).unwrap();
            max = max + T::from(0.5).unwrap();
        }

        let binwidth = (max - min) / T::from(bins).unwrap();

        for i in 0..(bins + 1) {
            self.bin_edges.push(T::from(i).unwrap() * binwidth);
            if i < bins {
                self.bin_count.push(0);
            }
        }

        for x in values.iter() {
            let mut hit = false;
            for j in 1..bins + 1 {
                let binwidth = self.bin_edges[j];
                if x < &binwidth {
                    hit = true;
                    self.bin_count[j - 1] += 1;
                    break;
                }
            }

            if !hit {
                let j = self.bin_count.len().saturating_sub(1);
                self.bin_count[j] += 1;
            }
        }
    }

    pub fn len(&self) -> usize {
        self.bin_count.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod test {
    use crate::gridspace;

    use super::*;

    #[test]
    fn test_fd() {
        let xval = gridspace(0.0, 10.5, 0.5);
        let width = freedman_diaconis_bin_width(&xval);
        let err = width - 3.6246012433429744;
        assert!(err.abs() < 1e-3, "{width} off by {err}");

        let hist = Histogram::new_freedman_diaconis(&xval);
        assert_eq!(hist.len(), 3);
    }

    #[test]
    fn test_sturges() {
        let xval = gridspace(0.0, 10.5, 0.5);
        let width = sturges_bin_width(&xval);
        let err = width - 2.2424382421757545;
        assert!(err.abs() < 1e-3, "{width} off by {err}");

        let hist = Histogram::new_sturges(&xval);
        assert_eq!(hist.len(), 5);
    }
}