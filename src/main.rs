use mzsignal::peak::FittedPeak;
use mzsignal::peak_picker;
use mzsignal::test_data::{NOISE, X, Y};

fn main() {
    let mut picker = peak_picker::PeakPicker::default();
    picker.signal_to_noise_threshold = 10.0;
    picker.intensity_threshold = 1.0;
    let mut acc: Vec<FittedPeak> = Vec::new();
    let yhat: Vec<f32> = Y
        .iter()
        .zip(NOISE.iter())
        .map(|(y, e)| y * 5.0 + e)
        .collect();
    let count = picker.discover_peaks(&X, &yhat, &mut acc);
    match count {
        Ok(count) => {
            println!("Found {} peaks", count);
            for peak in acc.iter() {
                println!("\t{}", peak);
            }
        }
        Err(err) => println!("Encountered error {:?}", err),
    };
}
