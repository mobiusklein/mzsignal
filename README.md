# mzsignal - Low Level Signal Processing For Mass Spectra
`mzsignal` is a library for performing low-level signal processing on
mass spectra en-route to converting a continuous profile-mode spectrum
into a centroided peak list.

The peak picking facility can be used directly with `PeakPicker` which
implements a simple gaussian peak shape fitter. There are a some threshold
criteria that can be manipulated to control which fits are reported, see its
documentation for more details.

When one spectrum is insufficient, averaging the signal from multiple spectra
together can be better. The `average` sub-module includes components
for merging together multiple profile spectra.

# Usage
```rust
use std::fs;
use std::io;
use std::io::prelude::*;

use mzsignal;

// Read in signal arrays from a text file
let mut mz_array: Vec<f64> = Vec::new();
let mut intensity_array: Vec<f32> = Vec::new();
let reader = io::BufReader::new(fs::File::open("./test/data/test.txt").unwrap());
for line in reader.lines() {
    let line = line.unwrap();
    let pref = line.trim();
    let chunks: Vec<&str> = pref.split("\t").collect();
    mz_array.push(chunks[0].parse::<f64>().expect("Expected number for m/z"));
    intensity_array.push(chunks[1].parse::<f32>().expect("Expected number for intensity"));
}

// Create a peak picker
let picker = mzsignal::PeakPicker::default();

// Create an accumulator
let mut acc = Vec::new();

// Pick peaks
let peak_count = picker.discover_peaks(&mz_array, &intensity_array, &mut acc).unwrap();
assert_eq!(peak_count , 4);

for peak in acc.iter() {
    println!("{}", peak);
}
```
## Building
This library depends upon `ndarray-linalg`, which means it needs a LAPACK implementation
as a backend for `ndarray-linalg`. These are enabled by passing one of the supported backends
as a `feature` to `cargo` e.g.: