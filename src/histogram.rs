use num_traits::{Float, FromPrimitive, ToPrimitive};

pub fn percentile<T: Float + ToPrimitive>(values: &[T], percent: f64) -> T {
    let k = (values.len() - 1) as f64 * percent;
    let f = k.floor();
    let c = k.ceil();
    if f == c {
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

pub fn minmax<T: Float>(values: &[T]) -> (T, T) {
    let mut max = -T::infinity();
    let mut min = T::infinity();

    for v in values.iter() {
        if *v > max {
            max = *v;
        }
        if *v < min {
            min = *v
        }
    }
    (min, max)
}

#[derive(Default, Debug, Clone)]
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
                let j = self.bin_count.len() - 1;
                self.bin_count[j] += 1;
            }
        }
    }

    pub fn len(&self) -> usize {
        self.bin_count.len()
    }
}
