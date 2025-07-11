# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.7] - 2025-07-10

### Fixed

- Fix another hole in FWHM calculation

## [1.1.6] - 2025-07-09

### Fixed

- Fix width calculation

## [1.1.5] - 2025-07-09

### Fixed

- Fix problem with overbroad peaks leading to invalid centroids

## [1.1.4] - 2025-03-09

### Added

- Add better error message when no backend is selected

### Changed

- Change make fewer copies and allocations during feature extraction

## [1.1.3] - 2025-02-24

### Changed

- Change sorting efficiency during feature map build

## [1.1.2] - 2025-02-16

### Fixed

- Handle ill-defined peak fits more explicitly but non-recoverably

## [1.1.1] - 2025-02-09

### Added

- Appease clippy
- Add feature trimming for low mass accuracy peaks

## [1.1.0] - 2025-01-26

### Added

- Add `ChargedIonMobilityFeature` map builder

### Changed

- Change `feature_mapping` to use `PeakSeries` trait to build features instead of decaying to raw tuples

## [1.0.7] - 2024-12-29

### Added

- Add a `prelude` module where traits will begin to accumulate. This remains unstable.

### Changed

- Changed `FeatureExtracterType::extract_features` to treat unused peaks as features of length 1 to include during feature merging.

## [1.0.6] - 2024-12-28

### Fixed

- Fix more broken imports in `feature_mapping::graph`

## [1.0.5] - 2024-12-28

### Fixed

- Fix broken import paths in `feature_mapping`

## [1.0.4] - 2024-12-28

### Added

- Add `FeatureTransform`

### Changed

- Upgrade ot `thiserror` v2
- Change the granularity of the centroid search in `lorentzian_fit` from dx/100 to dx/500

### Fixed

- Fix peak re-use in `feature_mapping`
- Fix test precision

## [1.0.3] - 2024-12-23

### Fixed

- Refactor `feature_statistics` to try to reduce pathological solution occurrences
- More fixes to the `BiGaussian` shape flipping during early re-implementation

### Removed

- Remove import

## [1.0.2] - 2024-12-19

### Changed

- Change peak shape model fitting defaults

## [1.0.1] - 2024-12-17

### Added

- Adjust version

### Fixed

- Fix feature graph merging process, was repeatedly skipping the second feature when merging connected components

## [1.0.0] - 2024-12-13

### Added

- Add `serde` feature support
- Add code coverage

### Changed

- Change `PeakPickerBuilder` to be a consuming builder
- Change `FittedPeak` default conversion from 0 to 0.005
- Upgrade to `mzpeaks` 1.0.0

### Fixed

- Fix test dependencies

### Removed

- Remove `search::nearest_binary`

## [0.27.0] - 2024-11-10

### Fixed

- Fix edge weighting algorithm for `FeatureExtracterType`

### Removed

- Remove test log file

## [0.26.0] - 2024-11-10

### Fixed

- Prevent the same peak being used in multiple features by `FeatureExtracterType`

## [0.25.0] - 2024-10-14

### Fixed

- Upgraded mzpeaks version

## [0.24.0] - 2024-10-14

### Added

- Add `feature_statistics` module for elution profile peak shape fitting (#2)

### Fixed

- Fix moving average methods to use central average

## [0.23.0] - 2024-09-06

### Added

- Add AVX intrinsic implementations for high density operations. (#1)

## [0.22.0] - 2024-09-01

### Fixed

- Revert change to `SignalAverager` to investigate  downstream  crashing

## [0.21.0] - 2024-08-30

### Added

- Add optimizers to `SignalAverager::interpolate_into`
- Add faster signal averaging algorithm

## [0.20.0] - 2024-08-09

### Fixed

- Fix mzpeaks version incompatibility

## [0.19.0] - 2024-08-09

### Added

- Add `PeakPickerBuilder` to top-level import

### Changed

- Upgrade `mzpeaks`

### Fixed

- Fix unsigned overflow in `denoise`

## [0.17.0] - 2024-07-15

### Fixed

- Fix overflow in peak width fitting
- Skip empty blocks when averaging spectra

## [0.16.0] - 2024-07-12

### Added

- Support higher versions of mzpeaks

## [0.15.0] - 2024-06-26

### Added

- Add grid-segmented averaging strategy

### Changed

- Change `arrayops` to be crate-only, export types and traits at the top level

### Removed

- Remove plotters implementation

## [0.14.0] - 2024-05-25

### Changed

- Changed quadratic fit to use incremental probing along m/z axis for high density data like rebinned or averaged signal

## [0.13.0] - 2024-05-17

### Added

- Add feature extraction algorithm

[1.1.7]: https://github.com/mobiusklein/mzsignal/compare/v1.1.6..v1.1.7
[1.1.6]: https://github.com/mobiusklein/mzsignal/compare/v1.1.5..v1.1.6
[1.1.5]: https://github.com/mobiusklein/mzsignal/compare/v1.1.4..v1.1.5
[1.1.4]: https://github.com/mobiusklein/mzsignal/compare/v1.1.3..v1.1.4
[1.1.3]: https://github.com/mobiusklein/mzsignal/compare/v1.1.2..v1.1.3
[1.1.2]: https://github.com/mobiusklein/mzsignal/compare/v1.1.1..v1.1.2
[1.1.1]: https://github.com/mobiusklein/mzsignal/compare/v1.1.0..v1.1.1
[1.1.0]: https://github.com/mobiusklein/mzsignal/compare/v1.0.7..v1.1.0
[1.0.7]: https://github.com/mobiusklein/mzsignal/compare/v1.0.6..v1.0.7
[1.0.6]: https://github.com/mobiusklein/mzsignal/compare/v1.0.5..v1.0.6
[1.0.5]: https://github.com/mobiusklein/mzsignal/compare/v1.0.4..v1.0.5
[1.0.4]: https://github.com/mobiusklein/mzsignal/compare/v1.0.3..v1.0.4
[1.0.3]: https://github.com/mobiusklein/mzsignal/compare/v1.0.2..v1.0.3
[1.0.2]: https://github.com/mobiusklein/mzsignal/compare/v1.0.1..v1.0.2
[1.0.1]: https://github.com/mobiusklein/mzsignal/compare/v1.0.0..v1.0.1
[1.0.0]: https://github.com/mobiusklein/mzsignal/compare/v0.27.0..v1.0.0
[0.27.0]: https://github.com/mobiusklein/mzsignal/compare/v0.26.0..v0.27.0
[0.26.0]: https://github.com/mobiusklein/mzsignal/compare/v0.25.0..v0.26.0
[0.25.0]: https://github.com/mobiusklein/mzsignal/compare/v0.24.0..v0.25.0
[0.24.0]: https://github.com/mobiusklein/mzsignal/compare/v0.23.0..v0.24.0
[0.23.0]: https://github.com/mobiusklein/mzsignal/compare/v0.22.0..v0.23.0
[0.22.0]: https://github.com/mobiusklein/mzsignal/compare/v0.21.0..v0.22.0
[0.21.0]: https://github.com/mobiusklein/mzsignal/compare/v0.20.0..v0.21.0
[0.20.0]: https://github.com/mobiusklein/mzsignal/compare/v0.19.0..v0.20.0
[0.19.0]: https://github.com/mobiusklein/mzsignal/compare/v0.17.0..v0.19.0
[0.17.0]: https://github.com/mobiusklein/mzsignal/compare/v0.16.0..v0.17.0
[0.16.0]: https://github.com/mobiusklein/mzsignal/compare/v0.15.0..v0.16.0
[0.15.0]: https://github.com/mobiusklein/mzsignal/compare/v0.14.0..v0.15.0
[0.14.0]: https://github.com/mobiusklein/mzsignal/compare/v0.13.0..v0.14.0
[0.13.0]: https://github.com/mobiusklein/mzsignal/compare/v0.12.0..v0.13.0

<!-- generated by git-cliff -->
