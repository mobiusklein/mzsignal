#![allow(dead_code, unused_variables, unused_imports)]
use std::env;
use std::fs;
use std::io;
use std::io::prelude::*;
use std::process;

use mzsignal::denoise::denoise;
use mzsignal::FittedPeak;
use mzsignal::peak_picker;

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let path = if args.len() > 1 {
        &args[1]
    } else {
        println!("Usage: mzsignal < path OR - >");
        println!(r#"""
Will read the input path or STDIN as a tab-separated series of (m/z, intensity) datapoints,
pick peaks from those points, and write the resulting peaks to STDOUT in tab separated format.
"""#);
        process::exit(1);
    };
    let mut mz_array: Vec<f64> = Vec::new();
    let mut intensity_array: Vec<f32> = Vec::new();
    if path != "-" {
        let reader = io::BufReader::new(fs::File::open(path)?);
        for line in reader.lines() {
            let line = line.unwrap();
            let pref = line.trim();
            let chunks: Vec<&str> = pref.split('\t').collect();
            mz_array.push(chunks[0].parse::<f64>().expect("Expected number for m/z"));
            intensity_array.push(
                chunks[1]
                    .parse::<f32>()
                    .expect("Expected number for intensity"),
            );
        }
        eprintln!("Read {} items from {}", mz_array.len(), path);
    } else {
        let stream = io::stdin();
        let reader = stream.lock();
        for line in reader.lines() {
            let line = line.unwrap();
            let pref = line.trim();
            let chunks: Vec<&str> = pref.split('\t').collect();
            mz_array.push(chunks[0].parse::<f64>().expect("Expected number for m/z"));
            intensity_array.push(
                chunks[1]
                    .parse::<f32>()
                    .expect("Expected number for intensity"),
            );
        }
        eprintln!("Read {} items from STDIN", mz_array.len());
    }
    let picker = peak_picker::PeakPicker {
        signal_to_noise_threshold: 3.0,
        ..Default::default()
    };
    let mut acc = Vec::new();
    match picker.discover_peaks(&mz_array, &intensity_array, &mut acc) {
        Ok(count) => {
            eprintln!("Found {} peaks", count);
        }
        Err(msg) => {
            eprintln!("Encountered an error while picking peaks: {:?}", msg);
        }
    }
    let outstream = io::stdout();
    let mut writer = outstream.lock();
    writer.write_all(b"mz\tintensity\tsnr\tfwhm\n")?;
    for peak in acc {
        writer.write_all(
            format!(
                "{}\t{}\t{}\t{}\n",
                peak.mz, peak.intensity, peak.signal_to_noise, peak.full_width_at_half_max
            )
            .as_bytes(),
        )?;
    }
    Ok(())
}
