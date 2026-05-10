use std::{borrow::Cow, collections::HashSet};

use mzpeaks::{
    IndexedCoordinate, IntensityMeasurement, IntensityMeasurementMut, KnownCharge, MZ, Mass, PeakCollection, Tolerance, peak_set::PeakSetVec
};

use crate::arrayops::{self, ArrayPairLike, ArrayPairSplit};

#[derive(Debug, thiserror::Error)]
pub enum BinnedSimilarityError {
    #[error("Expected equal sizes, but lhs was {lhs} and rhs was {rhs}")]
    UnequalCoordinateDimension { lhs: usize, rhs: usize },
    #[error("Expected equal coordinate domains, but lhs was {lhs} and rhs was {rhs}")]
    UnalignedCoordinateDimension { lhs: f64, rhs: f64 },
}

/// A binned spectrum for comparing (resolution dependent) similarity.
pub struct BinnedProfile<'a, 'b> {
    bins: ArrayPairSplit<'a, 'b>,
    cached_self_normalizer: Option<f32>,
}

impl<'a, 'b> From<ArrayPairSplit<'a, 'b>> for BinnedProfile<'a, 'b> {
    fn from(value: ArrayPairSplit<'a, 'b>) -> Self {
        Self {
            bins: value,
            cached_self_normalizer: None,
        }
    }
}

impl<'a, 'b> ArrayPairLike for BinnedProfile<'a, 'b> {
    fn mz_array(&self) -> &[f64] {
        <ArrayPairSplit<'a, 'b> as ArrayPairLike>::mz_array(&self.bins)
    }

    fn intensity_array(&self) -> &[f32] {
        <ArrayPairSplit<'a, 'b> as ArrayPairLike>::intensity_array(&self.bins)
    }

    fn to_owned(self) -> crate::ArrayPair<'static> {
        <ArrayPairSplit<'a, 'b> as ArrayPairLike>::to_owned(self.bins)
    }
}

impl<'a, 'b> BinnedProfile<'a, 'b> {
    pub fn new(array_pair_split: ArrayPairSplit<'a, 'b>) -> Self {
        Self {
            bins: array_pair_split,
            cached_self_normalizer: None,
        }
    }

    pub fn self_product_normalizer(&self) -> f32 {
        let mut chunks = self.intensity_array().chunks_exact(8);
        let mut self_sim_acc = [0.0f32; 8];

        while let Some(chunk) = chunks.next() {
            for i in 0..8 {
                self_sim_acc[i] += chunk[i].powi(2);
            }
        }

        let rem = chunks.remainder();
        for i in 0..rem.len() {
            self_sim_acc[i] += rem[i].powi(2);
        }
        let normalizer: f32 = self_sim_acc.iter().sum();
        normalizer
    }

    pub fn precompute_self_product_normalizer(&mut self) {
        self.cached_self_normalizer = Some(self.self_product_normalizer());
    }

    pub fn empty_for_range(start: f64, end: f64, step: f64) -> BinnedProfile<'static, 'static> {
        let mz = arrayops::gridspace(start, end, step);
        let intens = vec![0.0; mz.len()];
        BinnedProfile::new(ArrayPairSplit::from((mz, intens)))
    }

    /// Create a new copy of this profile, reusing the m/z array but smoothing the
    /// intensity array using a moving average kernel of width `width`.
    pub fn smooth(&self, width: usize) -> BinnedProfile<'a, 'static> {
        let mut new_intensity = vec![0.0f32; self.bins.len()];
        crate::smooth::moving_average_dyn(self.intensity_array(), &mut new_intensity, width);
        ArrayPairSplit::new(self.bins.mz_array.clone(), Cow::Owned(new_intensity)).into()
    }

    /// Fill the binned signal with 0.0s, clearing the existing values while leaving
    /// the bins unchanged
    pub fn clear(&mut self) {
        let bins = self.bins.intensity_array.to_mut();
        bins.fill(0.0);
        self.cached_self_normalizer = None;
    }

    /// Implementation details of the dot-product
    #[inline(always)]
    fn dot_product(
        mut chunks_lhs: core::slice::ChunksExact<'_, f32>,
        mut chunks_rhs: core::slice::ChunksExact<'_, f32>,
        self_needs_norm: bool,
        other_needs_norm: bool,
    ) -> (f32, f32, f32) {
        let mut acc = [0.0f32; 8];
        let mut denom_lhs_acc = [0.0f32; 8];
        let mut denom_rhs_acc = [0.0f32; 8];

        while let (Some(lhs), Some(rhs)) = (chunks_lhs.next(), chunks_rhs.next()) {
            match (self_needs_norm, other_needs_norm) {
                (true, false) => {
                    for i in 0..8 {
                        acc[i] += lhs[i] * rhs[i];
                        denom_lhs_acc[i] += lhs[i].powi(2);
                    }
                }
                (false, true) => {
                    for i in 0..8 {
                        acc[i] += lhs[i] * rhs[i];
                        denom_rhs_acc[i] += rhs[i].powi(2);
                    }
                }
                (true, true) => {
                    for i in 0..8 {
                        acc[i] += lhs[i] * rhs[i];
                        denom_lhs_acc[i] += lhs[i].powi(2);
                        denom_rhs_acc[i] += rhs[i].powi(2);
                    }
                }
                (false, false) => {
                    for i in 0..8 {
                        acc[i] += lhs[i] * rhs[i];
                    }
                }
            }
        }
        let rem_lhs = chunks_lhs.remainder();
        let rem_rhs = chunks_rhs.remainder();
        for i in 0..rem_lhs.len() {
            acc[i] += rem_lhs[i] * rem_rhs[i];
            if self_needs_norm {
                denom_lhs_acc[i] += rem_lhs[i].powi(2);
            }
            if other_needs_norm {
                denom_rhs_acc[i] += rem_rhs[i].powi(2);
            }
        }

        (
            acc.iter().sum(),
            denom_lhs_acc.iter().sum(),
            denom_rhs_acc.iter().sum(),
        )
    }

    /// Fill this binned profile from the provided source, adding the
    /// intensity points to each corresponding bin.
    pub fn fill_from<T: ArrayPairLike>(&mut self, source: &T) {
        self.cached_self_normalizer = None;
        if self.bins.mz_array.is_empty() {
            return;
        }
        let mz_bins_n = self.bins.mz_array().len();
        let intensity_bins = self.bins.intensity_array.to_mut();
        let mut thresh_i = 0;
        let mut thresh_mz = self.bins.mz_array[thresh_i];
        'z: for (mz, int) in source
            .mz_array()
            .iter()
            .copied()
            .zip(source.intensity_array().iter().copied())
        {
            while mz > thresh_mz {
                thresh_i += 1;
                if thresh_i < mz_bins_n {
                    thresh_mz = self.bins.mz_array[thresh_i];
                } else {
                    break 'z;
                }
            }
            if mz <= thresh_mz {
                intensity_bins[thresh_i] += int;
            }
        }
    }

    /// Build a new [`BinnedProfile`] from `source` using the grid spacing `step`, copying the
    /// intensity array into the appropriate bin
    pub fn bin_from<T: ArrayPairLike>(source: &T, step: f64) -> BinnedProfile<'static, 'static> {
        let start = source.mz_array().first().copied().unwrap_or_default();
        let end = source.mz_array().last().copied().unwrap_or_default();
        let mut this = Self::empty_for_range(start, end, step);
        this.fill_from(source);
        this
    }

    /// Build a new [`BinnedProfile`] using the signal in `source` but re-using the bins in `bins`
    pub fn bin_along<T: ArrayPairLike>(source: &T, bins: &'a [f64]) -> BinnedProfile<'a, 'static> {
        let ints: Cow<'_, [f32]> = Cow::Owned(vec![0.0f32; bins.len()]);
        let mzs = Cow::Borrowed(bins);
        let mut this: BinnedProfile<'a, 'static> = ArrayPairSplit::new(mzs, ints).into();
        this.fill_from(source);
        this
    }

    /// Compute the cosine similarity with the other `other`
    ///
    /// TODO: Write the equation
    pub fn cosine_similarity(
        &self,
        other: &BinnedProfile<'_, '_>,
    ) -> Result<f32, BinnedSimilarityError> {
        if self.len() != other.len() {
            return Err(BinnedSimilarityError::UnequalCoordinateDimension {
                lhs: self.len(),
                rhs: other.len(),
            });
        }
        if self.len() == 0 {
            return Ok(0.0);
        }

        let lhs = *self.mz_array().first().unwrap();
        let rhs = *other.mz_array().first().unwrap();
        if lhs != rhs {
            return Err(BinnedSimilarityError::UnalignedCoordinateDimension { lhs, rhs });
        }
        let lhs = *self.mz_array().last().unwrap();
        let rhs = *other.mz_array().last().unwrap();
        if lhs != rhs {
            return Err(BinnedSimilarityError::UnalignedCoordinateDimension { lhs, rhs });
        }

        let chunks_lhs = self.intensity_array().chunks_exact(8);
        let chunks_rhs = other.intensity_array().chunks_exact(8);
        let self_needs_norm = self.cached_self_normalizer.is_none();
        let other_needs_norm = other.cached_self_normalizer.is_none();

        #[cfg(target_arch = "x86_64")]
        let (total, denom_lhs, denom_rhs) = if std::arch::is_x86_feature_detected!("avx2") {
            Self::dot_product(chunks_lhs, chunks_rhs, self_needs_norm, other_needs_norm)
        } else {
            Self::dot_product(chunks_lhs, chunks_rhs, self_needs_norm, other_needs_norm)
        };
        #[cfg(not(target_arch = "x86_64"))]
        let (total, denom_lhs, denom_rhs) = Self::dot_product(chunks_lhs, chunks_rhs, self_needs_norm, other_needs_norm);

        let denom_lhs: f32 = if self_needs_norm {
            denom_lhs
        } else {
            self.cached_self_normalizer.unwrap()
        };

        let denom_rhs: f32 = if other_needs_norm {
            denom_rhs
        } else {
            other.cached_self_normalizer.unwrap()
        };
        let cosine = total / (denom_lhs.sqrt() * denom_rhs.sqrt());
        Ok(cosine)
    }
}

const PROTON: f64 = 1.00727646677;

#[inline]
pub fn mass_charge_ratio(mass: f64, z: i32) -> f64 {
    (mass + z as f64 * PROTON) / (z.abs() as f64)
}

pub trait AlignablePeakSetLike<P: IndexedCoordinate<C> + IntensityMeasurement, C> {
    fn peaks(&self) -> &PeakSetVec<P, C>;
    fn iter(&self) -> core::slice::Iter<'_, P>;
    fn parent_mass(&self) -> f64;
    fn parent_charge(&self) -> Option<i32>;
    fn transform(&mut self, transform: impl FnMut(&mut P) -> bool);
    fn precompute_normalizer(&mut self);

    fn as_slice<'a>(&'a self) -> &'a [P] where C: 'a {
        self.peaks().as_slice()
    }

    fn squared_normalizer(&self) -> f32 {
        self.iter().map(|p| p.intensity().powi(2)).sum()
    }

    fn get_normalizer_or_compute(&self) -> f32 {
        self.squared_normalizer()
    }

    fn parent_mz(&self) -> f64 {
        self.parent_charge()
            .map(|z| mass_charge_ratio(self.parent_mass(), z))
            .unwrap_or(self.parent_mass())
    }

    fn len(&self) -> usize {
        self.peaks().len()
    }

    fn is_empty(&self) -> bool {
        self.peaks().is_empty()
    }

    fn search_sorted_all_indices(
        &self,
        queries: &[P],
        error_tolerance: Tolerance,
        offset: f64,
    ) -> Vec<(usize, usize)> {
        let mut acc = Vec::new();
        self.search_sorted_all_indices_into(queries, error_tolerance, offset, &mut acc);
        acc
    }

    fn search_sorted_all_indices_into(
        &self,
        queries: &[P],
        error_tolerance: Tolerance,
        offset: f64,
        accumulator: &mut Vec<(usize, usize)>,
    ) {
        let mut checkpoint: usize = 0;
        let n = self.peaks().len();
        for (query_i, query) in queries.iter().enumerate() {
            let (lb, ub) = error_tolerance.bounds(query.coordinate());
            let lb_offset = lb + offset;
            let ub_offset = ub + offset;
            let lb_for_check = lb.min(lb_offset);
            let ub_for_check = ub.max(ub_offset);
            for (p, ref_i) in self
                .peaks()
                .get_slice(checkpoint..n)
                .iter()
                .zip(checkpoint..n)
            {
                let coord = p.coordinate();
                if coord < lb_for_check {
                    checkpoint = ref_i;
                } else if (coord > lb && coord < ub) || (coord > lb_offset && coord < ub_offset) {
                    accumulator.push((query_i, ref_i))
                } else if coord > ub_for_check {
                    break;
                }
            }
        }


    }

    fn overlaps(&self, other: &Self, error_tolerance: Tolerance) -> Vec<(usize, usize)> {
        let offset = self.parent_mass() - other.parent_mass();
        self.search_sorted_all_indices(other.as_slice(), error_tolerance, offset)
    }

    fn _similarity_score(&self, other: &Self, mut iis: Vec<(usize, usize)>) -> f32 {
        iis.sort_by(|a, b| {
            let aw = self.peaks().get_item(a.0).intensity() * other.peaks().get_item(a.1).intensity();
            let bw = self.peaks().get_item(b.0).intensity() * other.peaks().get_item(b.1).intensity();
            bw.total_cmp(&aw)
        });
        let mut mask: HashSet<(Option<usize>, Option<usize>)> =
            HashSet::with_capacity(iis.len() / 2);
        let mut score = 0.0;
        for (i, j) in iis {
            if mask.contains(&(Some(i), None)) || mask.contains(&(None, Some(j))) {
                continue;
            }
            mask.insert((Some(i), None));
            mask.insert((None, Some(j)));
            score += self.peaks().get_item(i).intensity() * other.peaks().get_item(j).intensity();
        }
        let lhs_norm = self
            .get_normalizer_or_compute().sqrt();

        let rhs_norm = other
            .get_normalizer_or_compute().sqrt();
        score / (lhs_norm * rhs_norm)
    }

    fn similarity(&self, other: &Self, error_tolerance: Tolerance) -> f32 {
        let iis = self.overlaps(other, error_tolerance);
        self._similarity_score(other, iis)
    }
}

#[derive(Debug, Clone, Default)]
pub struct AlignablePeakSet<P: IndexedCoordinate<MZ> + IntensityMeasurement> {
    pub peaks: PeakSetVec<P, MZ>,
    pub parent_mass: f64,
    pub parent_charge: Option<i32>,
    normalizer: Option<f32>,
}

impl<P: IndexedCoordinate<MZ> + IntensityMeasurement> AlignablePeakSetLike<P, MZ> for AlignablePeakSet<P> {
    fn peaks(&self) -> &PeakSetVec<P, MZ> {
        &self.peaks
    }

    fn iter(&self) -> core::slice::Iter<'_, P> {
        self.peaks.iter()
    }

    fn parent_mass(&self) -> f64 {
        self.parent_mass
    }

    fn parent_charge(&self) -> Option<i32> {
        self.parent_charge
    }

    fn transform(&mut self, transform: impl FnMut(&mut P) -> bool) {
        self.normalizer = None;
        self.peaks.peaks.retain_mut(transform);
    }

    fn precompute_normalizer(&mut self) {
        self.normalizer = Some(self.squared_normalizer());
    }
}

impl<P: IndexedCoordinate<MZ> + IntensityMeasurement + IntensityMeasurementMut>
    AlignablePeakSet<P>
    {

    pub fn new(peaks: PeakSetVec<P, MZ>, parent_mass: f64, parent_charge: Option<i32>) -> Self {
            Self {
                peaks,
                parent_mass,
                parent_charge,
                normalizer: None,
            }
        }

    pub fn similarity(&self, other: &Self, error_tolerance: Tolerance) -> f32 {
        AlignablePeakSetLike::similarity(self, other, error_tolerance)
    }
}

#[derive(Debug, Clone, Default)]
pub struct DeconvolvedAlignablePeakSet<P: IndexedCoordinate<Mass> + IntensityMeasurement + KnownCharge> {
    pub peaks: PeakSetVec<P, Mass>,
    pub parent_mass: f64,
    pub parent_charge: Option<i32>,
    normalizer: Option<f32>,
}

impl<P: IndexedCoordinate<Mass> + IntensityMeasurement + KnownCharge> DeconvolvedAlignablePeakSet<P> {
    pub fn new(peaks: PeakSetVec<P, Mass>, parent_mass: f64, parent_charge: Option<i32>, normalizer: Option<f32>) -> Self {
        Self { peaks, parent_mass, parent_charge, normalizer }
    }

    pub fn similarity(&self, other: &Self, error_tolerance: Tolerance) -> f32 {
        AlignablePeakSetLike::similarity(self, other, error_tolerance)
    }
}

impl<P: IndexedCoordinate<Mass> + IntensityMeasurement + KnownCharge> AlignablePeakSetLike<P, Mass> for DeconvolvedAlignablePeakSet<P> {
    fn peaks(&self) -> &PeakSetVec<P, Mass> {
        &self.peaks
    }

    fn iter(&self) -> core::slice::Iter<'_, P> {
        self.peaks.iter()
    }

    fn parent_mass(&self) -> f64 {
        self.parent_mass
    }

    fn parent_charge(&self) -> Option<i32> {
        self.parent_charge
    }

    fn transform(&mut self, transform: impl FnMut(&mut P) -> bool) {
        self.normalizer = None;
        self.peaks.peaks.retain_mut(transform);
    }

    fn precompute_normalizer(&mut self) {
        self.normalizer = Some(self.squared_normalizer());
    }

    fn search_sorted_all_indices_into(
        &self,
        queries: &[P],
        error_tolerance: Tolerance,
        offset: f64,
        accumulator: &mut Vec<(usize, usize)>,
    )
    {
        let mut checkpoint: usize = 0;
        let n = self.peaks.len();
        for (query_i, query) in queries.iter().enumerate() {
            let (lb, ub) = error_tolerance.bounds(query.coordinate());
            let lb_offset = lb + offset;
            let ub_offset = ub + offset;
            let lb_for_check = lb.min(lb_offset);
            let ub_for_check = ub.max(ub_offset);
            let q_charge = query.charge();
            for (p, ref_i) in self
                .peaks
                .get_slice(checkpoint..n)
                .iter()
                .zip(checkpoint..n)
            {
                let coord = p.coordinate();
                if coord < lb_for_check {
                    checkpoint = ref_i;
                } else if ((coord > lb && coord < ub) || (coord > lb_offset && coord < ub_offset))
                    && q_charge == p.charge()
                {
                    accumulator.push((query_i, ref_i))
                } else if coord > ub_for_check {
                    break;
                }
            }
        }
    }
}



#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        pick_peaks,
        test_data::{NOISE, X, Y},
        FittedPeak,
    };
    use mzpeaks::MZ;

    #[test]
    fn test_bin_construction() {
        let data = ArrayPairSplit::wrap(&X, &Y);
        let bins = BinnedProfile::bin_from(&data, 0.1);
        eprintln!(
            "generated {} bins from {} to {}",
            bins.mz_array().len(),
            bins.mz_array().first().unwrap(),
            bins.mz_array().last().unwrap()
        );
        assert_eq!(bins.mz_array().len(), 59);
        eprintln!("{}-{}", data.min_mz, data.max_mz);
    }

    #[test_log::test]
    fn test_self_similarity() {
        let data = ArrayPairSplit::wrap(&X, &Y);
        let mut bins = BinnedProfile::bin_from(&data, 0.1);
        let sim = bins.cosine_similarity(&bins).unwrap();
        let e = (sim - 1.0).abs();
        assert!(e < 1e-6, "{sim} was {e} off the expected value (1.0)");

        bins.precompute_self_product_normalizer();
        let sim2 = bins.cosine_similarity(&bins).unwrap();
        let e = (sim2 - 1.0).abs();
        assert!(e < 1e-6, "{sim2} was {e} off the expected value (1.0)");

        let e = (sim - sim2).abs();
        assert!(e < 1e-6, "{sim2} was {e} off {sim}");
    }

    #[test_log::test]
    fn test_with_noise() {
        let data = ArrayPairSplit::wrap(&X, &Y);
        let mut bins = BinnedProfile::bin_from(&data, 0.1);
        let mut bins2 = BinnedProfile::bin_from(&data, 0.1);
        bins2.fill_from(&ArrayPairSplit::wrap(&X, &NOISE));
        let sim = bins.cosine_similarity(&bins2).unwrap();
        let e = (0.99197996 - sim).abs();
        assert!(e < 1e-6, "{sim} was {e} off the expected value");

        bins.precompute_self_product_normalizer();
        bins2.precompute_self_product_normalizer();

        let sim2 = bins.cosine_similarity(&bins2).unwrap();
        let e = (0.99197996 - sim2).abs();
        assert!(e < 1e-6, "{sim2} was {e} off the expected value");

        let e = (sim - sim2).abs();
        assert!(e < 1e-6, "{sim2} was {e} off {sim}");
    }

    fn prepare_peaks() -> PeakSetVec<FittedPeak, MZ> {
        let yhat: Vec<f32> = Y
            .iter()
            .zip(NOISE.iter())
            .map(|(y, e)| y * 50.0 + e * 20.0)
            .collect();
        let mut peaks = pick_peaks(&X, &yhat).unwrap();
        peaks.sort_by(|a, b| a.mz.total_cmp(&b.mz));
        peaks.into()
    }

    #[test]
    fn test_offset() {
        let peaks = prepare_peaks();
        let apeaks = AlignablePeakSet::new(peaks, 500.0, Some(1));
        let bpeaks = apeaks.clone();
        let iis = apeaks.search_sorted_all_indices(bpeaks.as_slice(), Tolerance::PPM(10.0), 0.0);
        assert_eq!(iis.len(), apeaks.len());

        let mut peaks2 = prepare_peaks();
        peaks2.iter_mut().for_each(|p| {
            p.mz += 10.0;
        });
        let bpeaks = AlignablePeakSet::new(peaks2, 510.0, Some(1));
        let iis = apeaks.overlaps(&bpeaks, Tolerance::PPM(10.0));
        assert_eq!(iis.len(), bpeaks.len());
        let iis = bpeaks.overlaps(&apeaks, Tolerance::PPM(10.0));
        assert_eq!(iis.len(), apeaks.len());

    }

    #[test]
    fn test_peak_similarity() {
        let peaks = prepare_peaks();
        let apeaks = AlignablePeakSet::new(peaks, 500.0, Some(1));
        let s = apeaks.similarity(&apeaks, Tolerance::PPM(10.0));
        assert_eq!(s, 1.0);

        let mut peaks2 = prepare_peaks();
        peaks2.iter_mut().for_each(|p| {
            p.mz += 10.0;
        });
        let mut bpeaks = AlignablePeakSet::new(peaks2, 510.0, Some(1));
        let s = apeaks.similarity(&bpeaks, Tolerance::PPM(10.0));
        assert_eq!(s, 1.0);

        bpeaks.precompute_normalizer();
        let s = apeaks.similarity(&bpeaks, Tolerance::PPM(10.0));
        assert_eq!(s, 1.0);

        bpeaks.parent_mass = 500.0;
        let s = apeaks.similarity(&bpeaks, Tolerance::PPM(10.0));
        assert_eq!(s, 0.0);
    }
}
