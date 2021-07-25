use std::io;
use std::time::{Instant};

use mzsignal::peak::FittedPeak;
use mzsignal::peak_picker;
use mzsignal::test_data::{NOISE, X, Y};
use mzsignal::average::{SignalAverager, ArrayPair};

fn main() -> io::Result<()> {
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

    let mut averager = SignalAverager::new(X[0], X[X.len() - 1], 0.00001);
    averager.push(ArrayPair::new(&X, &Y));
    let start = Instant::now();
    let rebinned = averager.interpolate();
    println!("Rebinning took milliseconds {}", (Instant::now() - start).as_millis());
    picker.signal_to_noise_threshold = 1.0;
    let mut acc2 = Vec::new();
    let count = picker.discover_peaks(&averager.mz_grid, &rebinned, &mut acc2);
    match count {
        Ok(count) => {
            println!("Found {} peaks after re-binning", count);
            for peak in acc2.iter() {
                println!("\t{}", peak);
            }
        }
        Err(err) => println!("Encountered error {:?}", err),
    };
    let start = Instant::now();
    let rebinned_3 = averager.interpolate_chunks(3);
    println!("Rebinning took milliseconds {}", (Instant::now() - start).as_millis());
    let mut acc3 = Vec::new();
    let count = picker.discover_peaks(&averager.mz_grid, &rebinned_3, &mut acc3);
    match count {
        Ok(count) => {
            println!("Found {} peaks after re-binning", count);
            for peak in acc3.iter() {
                println!("\t{}", peak);
            }
        }
        Err(err) => println!("Encountered error {:?}", err),
    };

    #[cfg(feature = "parallelism")]
    {
        println!("Parallel Method");
        let start = Instant::now();
        let rebinned_5 = averager.interpolate_chunks_parallel(6);
        println!("Rebinning took milliseconds {}", (Instant::now() - start).as_millis());
        let mut acc5 = Vec::new();
        let count = picker.discover_peaks(&averager.mz_grid, &rebinned_5, &mut acc5);
        match count {
            Ok(count) => {
                println!("Found {} peaks after re-binning", count);
                for peak in acc5.iter() {
                    println!("\t{}", peak);
                }
            }
            Err(err) => println!("Encountered error {:?}", err),
        };
    }
    Ok(())
}
