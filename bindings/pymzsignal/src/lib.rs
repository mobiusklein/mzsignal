use std::borrow::Cow;
use std::ops::Index;

use mzpeaks::prelude::PeakCollectionMut;
use ndarray::prelude::Dim;
use pyo3::exceptions::{PyException, PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyLong, PySlice, PyString};

use numpy::{PyArray, PyArray1, PyReadonlyArray1, ToPyArray};

use mzsignal::average::average_signal;
use mzsignal::denoise::denoise;
use mzsignal::peak_statistics::{approximate_signal_to_noise, full_width_at_half_max};
use mzsignal::reprofile::reprofile;
use mzsignal::smooth::{moving_average, savitsky_golay};
use mzsignal::{ArrayPair, FittedPeak, PeakPicker, PeakPickerError};

use mzpeaks::coordinate::MZ;
use mzpeaks::peak_set::{PeakCollection, PeakSetVec};
use mzpeaks::Tolerance as _Tolerance;
use mzpeaks::{CoordinateLike, IndexedCoordinate, IntensityMeasurement};

#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Tolerance(_Tolerance);

impl Into<_Tolerance> for Tolerance {
    fn into(self) -> _Tolerance {
        self.0
    }
}

#[allow(non_snake_case)]
#[pymethods]
impl Tolerance {
    #[staticmethod]
    fn Da(value: f64) -> PyResult<Self> {
        Ok(Self(_Tolerance::Da(value)))
    }

    #[staticmethod]
    fn PPM(value: f64) -> PyResult<Self> {
        Ok(Self(_Tolerance::PPM(value)))
    }

    fn __repr__(&self) -> String {
        self.0.to_string()
    }
}

#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct PyFittedPeak(FittedPeak);

impl PyFittedPeak {
    pub fn new(peak: FittedPeak) -> Self {
        Self(peak)
    }
}

impl From<FittedPeak> for PyFittedPeak {
    fn from(value: FittedPeak) -> Self {
        Self::new(value)
    }
}

impl CoordinateLike<MZ> for PyFittedPeak {
    fn coordinate(&self) -> f64 {
        self.mz()
    }
}

impl IndexedCoordinate<MZ> for PyFittedPeak {
    fn get_index(&self) -> mzpeaks::IndexType {
        self.0.index
    }

    fn set_index(&mut self, index: mzpeaks::IndexType) {
        self.0.index = index
    }
}

impl IntensityMeasurement for PyFittedPeak {
    fn intensity(&self) -> f32 {
        self.0.intensity
    }
}

#[pymethods]
impl PyFittedPeak {
    #[getter]
    fn mz(&self) -> f64 {
        self.0.mz
    }

    #[getter]
    fn intensity(&self) -> f32 {
        self.0.intensity
    }

    #[getter]
    fn index(&self) -> u32 {
        return self.0.index;
    }

    #[getter]
    fn signal_to_noise(&self) -> f32 {
        self.0.signal_to_noise
    }

    #[getter]
    fn full_width_at_half_max(&self) -> f32 {
        self.0.full_width_at_half_max
    }

    fn __repr__(&self) -> String {
        format!(
            "PyFittedPeak({:0.4}, {:0.4}, {}, {:0.4}, {:0.4})",
            self.mz(),
            self.intensity(),
            self.index(),
            self.signal_to_noise(),
            self.full_width_at_half_max()
        )
    }
}

#[pyclass(sequence)]
#[derive(Debug, Clone, PartialEq)]
pub struct PyPeakSet(PeakSetVec<PyFittedPeak, MZ>);

impl Index<usize> for PyPeakSet {
    type Output = PyFittedPeak;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl PeakCollection<PyFittedPeak, MZ> for PyPeakSet {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn get_item(&self, i: usize) -> &PyFittedPeak {
        self.0.get_item(i)
    }

    fn get_slice(&self, i: std::ops::Range<usize>) -> &[PyFittedPeak] {
        self.0.get_slice(i)
    }

    fn search_by(&self, query: f64) -> Result<usize, usize> {
        self.0.search_by(query)
    }

    fn iter(&self) -> impl Iterator<Item = &PyFittedPeak> {
        self.0.iter()
    }
}

impl PeakCollectionMut<PyFittedPeak, MZ> for PyPeakSet {
    fn push(&mut self, peak: PyFittedPeak) -> mzpeaks::peak_set::OrderUpdateEvent {
        self.0.push(peak)
    }

    fn sort(&mut self) {
        self.0.sort()
    }
}

impl PyPeakSet {
    pub fn new(peaks: Vec<PyFittedPeak>) -> Self {
        PyPeakSet(PeakSetVec::new(peaks))
    }
}

#[derive(Debug)]
pub struct ErrorToleranceArg(Tolerance);

impl From<f64> for ErrorToleranceArg {
    fn from(value: f64) -> Self {
        Self(Tolerance::PPM(value).unwrap())
    }
}

impl<'a> FromPyObject<'a> for ErrorToleranceArg {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        if ob.is_instance_of::<PyFloat>()? {
            Ok(ErrorToleranceArg(ob.extract()?))
        } else if ob.is_instance_of::<Tolerance>()? {
            Ok(ErrorToleranceArg(ob.extract()?))
        } else if ob.is_instance_of::<PyString>()? {
            let s: &str = ob.extract()?;
            match s.parse::<_Tolerance>() {
                Ok(v) => Ok(ErrorToleranceArg(Tolerance(v))),
                Err(e) => Err(PyValueError::new_err(format!(
                    "Failed to parse error tolerance: {}",
                    e
                ))),
            }
        } else {
            Err(PyTypeError::new_err(
                "Could not convert object to tolerance",
            ))
        }
    }
}

impl Into<_Tolerance> for ErrorToleranceArg {
    fn into(self) -> _Tolerance {
        self.0.into()
    }
}

#[pymethods]
impl PyPeakSet {
    #[new]
    pub fn py_new(peaks: Vec<PyFittedPeak>) -> Self {
        PyPeakSet(PeakSetVec::new(peaks))
    }

    #[pyo3(signature=(mz, error_tolerance=10.0f64.into()))]
    pub fn has_peak(&self, mz: f64, error_tolerance: ErrorToleranceArg) -> Option<PyFittedPeak> {
        if let Some(peak) = self.0.has_peak(mz, error_tolerance.into()) {
            Some(peak.clone())
        } else {
            None
        }
    }

    pub fn all_peaks_for(
        &self,
        mz: f64,
        error_tolerance: ErrorToleranceArg,
    ) -> PyResult<Vec<PyFittedPeak>> {
        let res = self
            .0
            .all_peaks_for(mz, error_tolerance.into())
            .iter()
            .copied()
            .collect::<Vec<_>>();
        Ok(res)
    }

    pub fn between(&self, m1: f64, m2: f64) -> PyResult<Vec<PyFittedPeak>> {
        Ok(self
            .0
            .between(m1, m2, _Tolerance::PPM(10.0))
            .into_iter()
            .copied()
            .collect())
    }

    fn __getitem__(&self, i: &PyAny) -> PyResult<PyFittedPeak> {
        if i.is_instance_of::<PySlice>()? {
            Err(PyTypeError::new_err("Could not select indices by slice"))
        } else if i.is_instance_of::<PyLong>()? {
            let i: usize = i.extract()?;
            if i >= self.len() {
                Err(PyIndexError::new_err(i))
            } else {
                Ok(self[i].clone())
            }
        } else {
            Err(PyTypeError::new_err("Could not select indices from input"))
        }
    }

    fn __len__(&self) -> usize {
        self.len()
    }

    fn __repr__(&self) -> String {
        format!("PyPeakSet({} peaks)", self.len())
    }
}

#[pyfunction]
#[pyo3(name = "approximate_signal_to_noise")]
fn py_approximate_signal_to_noise(
    intensity_array: PyReadonlyArray1<f32>,
    index: usize,
) -> PyResult<f32> {
    let tmp = intensity_array.as_slice()?;
    let result = approximate_signal_to_noise(tmp[index], tmp, index);
    return Ok(result);
}

#[pyfunction]
#[pyo3(name = "fit_full_width_at_half_max")]
fn py_fit_full_width_at_half_max(
    mz_array: PyReadonlyArray1<f64>,
    intensity_array: PyReadonlyArray1<f32>,
    index: usize,
) -> PyResult<(f64, f64, f64)> {
    let intensity_view = intensity_array.as_slice()?;
    let mz_view = mz_array.as_slice()?;
    let signal_to_noise = approximate_signal_to_noise(intensity_view[index], intensity_view, index);
    let fwhm = full_width_at_half_max(mz_view, intensity_view, index, signal_to_noise);
    let res = (
        fwhm.full_width_at_half_max,
        fwhm.left_width,
        fwhm.right_width,
    );
    return Ok(res);
}

#[pyfunction]
#[pyo3(name = "denoise", signature = (mz_array, intensity_array, scale = 5.0, inplace = true))]
fn py_denoise<'py>(
    mz_array: PyReadonlyArray1<f64>,
    intensity_array: &'py PyArray1<f32>,
    scale: f32,
    inplace: bool,
) -> PyResult<Py<PyArray1<f32>>> {
    let mz_view = mz_array.as_slice()?;
    if inplace {
        unsafe {
            let view = intensity_array.as_slice_mut()?;
            let res = denoise(mz_view, view, scale);
            match res {
                Ok(_view) => Ok(intensity_array.into()),
                Err(_) => Err(PyException::new_err("An error occurred while denoising")),
            }
        }
    } else {
        let mut intensity_vec: Vec<f32> = intensity_array.extract()?;
        let res = denoise(mz_view, intensity_vec.as_mut_slice(), scale);
        match res {
            Ok(_view) => {
                let pyarray = Python::with_gil(|py| -> Py<PyArray1<f32>> {
                    let result = PyArray1::from_iter(py, intensity_vec);
                    result.to_owned()
                });
                Ok(pyarray)
            }
            Err(_) => Err(PyException::new_err("An error occurred while denoising")),
        }
    }
}

#[pyfunction]
#[pyo3(name = "average_signal", signature = (array_pairs, dx = 0.005))]
fn py_average_signal(
    array_pairs: Vec<(PyReadonlyArray1<f64>, PyReadonlyArray1<f32>)>,
    dx: f64,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f32>>)> {
    let wrapped_pairs: Vec<ArrayPair> = array_pairs
        .iter()
        .map(|(x, y)| {
            ArrayPair::new(
                Cow::Borrowed(x.as_slice().unwrap()),
                Cow::Borrowed(y.as_slice().unwrap()),
            )
        })
        .collect();

    let new_arrays = average_signal(&wrapped_pairs, dx);
    Python::with_gil(|py| {
        let x = new_arrays.mz_array.to_pyarray(py).into();
        let y = new_arrays.intensity_array.to_pyarray(py).into();
        Ok((x, y))
    })
}

#[pyfunction]
#[pyo3(name = "pick_peaks", signature = (mz_array, intensity_array, signal_to_noise_threshold=1.0))]
fn py_pick_peaks(
    mz_array: PyReadonlyArray1<f64>,
    intensity_array: PyReadonlyArray1<f32>,
    signal_to_noise_threshold: f32,
) -> PyResult<PyPeakSet> {
    let picker = PeakPicker {
        signal_to_noise_threshold,
        ..PeakPicker::default()
    };

    let mut acc = Vec::new();
    let peaks_res =
        picker.discover_peaks(mz_array.as_slice()?, intensity_array.as_slice()?, &mut acc);
    match peaks_res {
        Ok(_) => {
            let pypeaks: Vec<PyFittedPeak> = acc.into_iter().map(|p| PyFittedPeak(p)).collect();
            return Ok(PyPeakSet::new(pypeaks));
        }
        Err(err) => match err {
            PeakPickerError::IntervalTooSmall => Err(PyException::new_err("Interval is too small")),
            PeakPickerError::MZIntensityMismatch => Err(PyException::new_err(
                "mz_array and intensity_array must have the same size",
            )),
            PeakPickerError::Unknown => Err(PyException::new_err("Unknown error")),
            PeakPickerError::MZNotSorted => Err(PyException::new_err("mz_array is not sorted")),
        },
    }
}

#[pyfunction]
#[pyo3(name = "reprofile", signature = (peaks, dx = 0.005))]
fn py_reprofile(
    peaks: Py<PyPeakSet>,
    dx: f64,
) -> (
    Py<PyArray<f64, Dim<[usize; 1]>>>,
    Py<PyArray<f32, Dim<[usize; 1]>>>,
) {
    let (mz_array, intensity_array) = Python::with_gil(|py| {
        let peak_set = peaks.borrow(py);
        let pair = reprofile(peak_set.0.iter().map(|p| &p.0), dx);
        let mz_array = pair.mz_array.to_pyarray(py).to_owned();
        let intensity_array = pair.intensity_array.to_pyarray(py).to_owned();
        (mz_array, intensity_array)
    });
    (mz_array, intensity_array)
}

#[pyfunction]
#[pyo3(name = "moving_average", signature = (mz_array, intensity_array))]
fn py_moving_average(
    #[allow(unused)] mz_array: PyReadonlyArray1<f64>,
    intensity_array: PyReadonlyArray1<f32>,
) -> PyResult<Py<PyArray<f32, Dim<[usize; 1]>>>> {
    let intensity_array_ = intensity_array.as_slice()?;
    let mut new_intensity = Vec::with_capacity(intensity_array_.len());
    new_intensity.resize(intensity_array_.len(), 0.0);
    moving_average::<f32, 3>(intensity_array_, &mut new_intensity);
    let new_py_intensity_array: Py<PyArray<f32, Dim<[usize; 1]>>> =
        Python::with_gil(|py| new_intensity.to_pyarray(py).to_owned());
    Ok(new_py_intensity_array)
}

#[pyfunction]
#[pyo3(name = "savitsky_golay", signature = (mz_array, intensity_array, window_length = 5, poly_order=3, derivative = 0))]
fn py_savitsky_golay(
    #[allow(unused)] mz_array: PyReadonlyArray1<f64>,
    intensity_array: PyReadonlyArray1<f32>,
    window_length: usize,
    poly_order: usize,
    derivative: usize,
) -> PyResult<Py<PyArray<f32, Dim<[usize; 1]>>>> {
    let res = savitsky_golay::<f32>(
        intensity_array.as_slice()?,
        window_length,
        poly_order,
        derivative,
    );
    let res = match res {
        Ok(arr) => arr,
        Err(e) => return Err(PyValueError::new_err(e.to_string())),
    };

    let py_res = Python::with_gil(|py| res.to_pyarray(py).to_owned());
    Ok(py_res)
}

/// A Python module implemented in Rust.
#[pymodule]
fn pymzsignal(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_approximate_signal_to_noise, m)?)?;
    m.add_function(wrap_pyfunction!(py_pick_peaks, m)?)?;
    m.add_function(wrap_pyfunction!(py_denoise, m)?)?;
    m.add_function(wrap_pyfunction!(py_average_signal, m)?)?;
    m.add_function(wrap_pyfunction!(py_fit_full_width_at_half_max, m)?)?;
    m.add_function(wrap_pyfunction!(py_reprofile, m)?)?;
    m.add_function(wrap_pyfunction!(py_moving_average, m)?)?;
    m.add_function(wrap_pyfunction!(py_savitsky_golay, m)?)?;
    m.add_class::<PyFittedPeak>()?;
    Ok(())
}
