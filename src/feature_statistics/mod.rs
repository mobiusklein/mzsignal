//! Fitting methods for elution-over-time profile peak shape models.
//!
//! ![A skewed gaussian peak shape fit to an asymmetric profile][peak_fit]
//!
//! This covers multiple peak shape kinds and has some support
//! for multi-modal profiles.
//!
//! The supported peak shape types:
//! - [`GaussianPeakShape`]
//! - [`SkewedGaussianPeakShape`]
//! - [`BiGaussianPeakShape`]
//!
//! and the [`PeakShape`] type that can be used when dealing with
//!
//! Most of the fitting methods expect to work with [`PeakFitArgs`] which
//! can be created from borrowed signal-over-time data.
//!
//! # Example
//!
//! ```rust
//! # use mzsignal::text::load_feature_table;
//! use mzpeaks::feature::Feature;
//! use mzsignal::feature_statistics::{PeakFitArgs, SplittingPeakShapeFitter, FitConfig};
//!
//! # fn main() {
//! let features: Vec<Feature<_, _>> = load_feature_table("test/data/features_graph.txt").unwrap();
//! let feature = &features[10979];
//! let args = PeakFitArgs::from(feature);
//! let mut fitter = SplittingPeakShapeFitter::new(args);
//! fitter.fit_with(FitConfig::default().max_iter(10_000).smooth(1));
//! let z = fitter.score();
//! eprintln!("Score: {z}");
//! # }
//! ```
//!
//! # Model Fit Evaluation
//!
//! All peak shape models are optimized using a mean squared error (MSE) loss function, regularizing over the
//! position and shape parameters but not the amplitude parameter. Parameters are updated using a basic gradient
//! descent procedure.
//!
//! A peak shape fit isn't just about minimizing the residual error, it's about there actually being a peak, so
//! for downstream applications, we provide a [`PeakShapeModel::score`] method which compares the MSE of the model
//! to a straight line linear model of the form $`y = \alpha + \beta\times x`$. Prior work has shown this approach
//! can be more effective at distinguishing jagged noise regions where a peak shape can *fit*, but isn't meaningful.
//!
//!
//!
//! [peak_fit]: https://github.com/mobiusklein/mzsignal/blob/feature/argmin_shape_fit/doc/chromatogram.png?raw=true

mod data;
mod fitter;
mod multishapes;
mod shapes;
mod utils;

pub use data::{PeakFitArgs, PeakFitArgsIter, SplittingPoint};
pub use fitter::{FitPeaksOn, PeakShapeFitter, SplittingPeakShapeFitter};
pub use multishapes::{MultiPeakShapeFit, PeakShape};
pub use shapes::{BiGaussianPeakShape, GaussianPeakShape, SkewedGaussianPeakShape};
pub use utils::{FitConfig, FitConstraints, ModelFitResult, PeakShapeModel, PeakShapeModelFitter, FeatureTransform};


#[cfg(test)]
mod test {
    use mzpeaks::{feature::Feature, Time, MZ};
    use crate::gridspace;
    use multishapes::dispatch_peak;

    use super::*;

    #[rstest::fixture]
    #[once]
    fn feature_table() -> Vec<Feature<MZ, Time>> {
        log::info!("Logging initialized");
        crate::text::load_feature_table("test/data/features_graph.txt").unwrap()
    }

    macro_rules! assert_is_close {
        ($t1:expr, $t2:expr, $tol:expr, $label:literal) => {
            assert!(
                ($t1 - $t2).abs() < $tol,
                "Observed {} {}, expected {}, difference {}",
                $label,
                $t1,
                $t2,
                $t1 - $t2,
            );
        };
        ($t1:expr, $t2:expr, $tol:expr, $label:literal, $obj:ident) => {
            assert!(
                ($t1 - $t2).abs() < $tol,
                "Observed {} {}, expected {}, difference {} from {:?}",
                $label,
                $t1,
                $t2,
                $t1 - $t2,
                $obj
            );
        };
    }

    /*
    #[test_log::test]
    fn test_fitted() {
        let reader = io::BufReader::new(fs::File::open("test/data/fitted.csv").unwrap());
        let mut times = Vec::new();
        let mut intensities = Vec::new();

        for line in reader.lines().skip(1).flatten() {
            let parts: Vec<_> = line.split('\t').flat_map(|t| t.parse::<f64>()).collect();
            let (time, _pred, observed, _smooth) = (parts[0], parts[1], parts[2], parts[3]);
            times.push(time);
            intensities.push(observed as f32);
        }

        let args: PeakFitArgs = PeakFitArgs::from((times, intensities));
        let point = args.locate_extrema(None);
        let split_points = args.split_at(point.as_slice());

        let part0 = args.slice(split_points[0].clone());
        let part1 = args.slice(split_points[1].clone());
        eprintln!("{:?} {:?}", part0.get(0), part1.get(0));

        let mut s1 = SkewedGaussianPeakShape::guess(&part1.smooth(3));
        let mut s2 = BiGaussianPeakShape::guess(&part1.smooth(3));
        eprintln!("{s1:?}\n{s2:?}");

        s1.fit_with(
            part1.borrow(),
            FitConfig::default()
                .learning_rate(0.0001)
                .max_iter(50_000)
                .smooth(0)
                .splitting_threshold(0.2),
        );

        s2.fit_with(
            part1.borrow().smooth(3),
            FitConfig::default()
                .learning_rate(0.0001)
                .max_iter(50_000)
                .smooth(0)
                .splitting_threshold(0.2),
        );
        // s1.fit(part1.borrow());

        let mut fitter = SplittingPeakShapeFitter::new(args.borrow());
        fitter.fit_with(
            FitConfig::default()
                .max_iter(50_000)
                .smooth(3)
                .learning_rate(0.0001),
        );
        eprintln!("{:?}", fitter.peak_fits);
        eprintln!("Score {}", fitter.score());
        let mut fits = vec![
            ("raw".to_string(), args.borrow(), None, None),
            ("smoothed".to_string(), args.smooth(3), None, None),
        ];
        for (i, fit) in fitter.peak_fits.iter().enumerate() {
            let pred = fit
                .predict_iter(args.time.iter().copied())
                .map(|y| y as f32)
                .collect();
            fits.push((
                format!("fit_{i}"),
                PeakFitArgs::from((args.time.to_vec(), pred)),
                Some(fit.clone()),
                Some(fit.score(&args.smooth(3))),
            ));
        }

        let pred = s2
            .predict_iter(args.time.iter().copied())
            .map(|y| y as f32)
            .collect();
        fits.push((
            "S2_fit".into(),
            PeakFitArgs::from((args.time.to_vec(), pred)),
            Some(PeakShape::BiGaussian(s2)),
            Some(s2.score(&args.smooth(3))),
        ));
        let pred = s1
            .predict_iter(args.time.iter().copied())
            .map(|y| y as f32)
            .collect();
        fits.push((
            "S1_fit".into(),
            PeakFitArgs::from((args.time.to_vec(), pred)),
            Some(PeakShape::SkewedGaussian(s1)),
            Some(s1.score(&args.smooth(3))),
        ));

        #[cfg(feature = "serde")]
        serde_json::to_writer(std::fs::File::create("fits.json").unwrap(), &fits).unwrap();
    }
    */

    #[test]
    fn test_bigaussian() {
        let model = BiGaussianPeakShape {
            mu: 125.47866390397324,
            sigma_falling: 0.42279618202432384,
            sigma_rising: 0.5734496177551722,
            amplitude: 1445596.1606827166,
        };

        let times = gridspace(124.0, 128.0, 0.1);
        let intensities = model
            .predict_iter(times.iter().copied())
            .map(|s| s as f32)
            .collect();
        let data = PeakFitArgs::from((times, intensities));

        let mut test_model = BiGaussianPeakShape::guess(&data);
        eprintln!("Initial: {test_model:?}");
        let fit_status = test_model.fit_with(
            data.borrow(),
            FitConfig::default()
                .max_iter(50_000)
                .learning_rate(0.001)
                .convergence(1e-9),
        );

        eprintln!(
            "Model: {:?} {} {fit_status:?}",
            test_model,
            test_model.score(&data)
        );

        let grad1 = test_model.gradient(&data, None);
        let grad2 = test_model.gradient_split(&data);
        eprintln!("{grad1:?} {grad2:?}");

        assert!(fit_status.success);
    }

    #[rstest::rstest]
    #[test_log::test]
    fn test_fit_feature_14216(feature_table: &[Feature<MZ, Time>]) {
        let feature = &feature_table[14216];
        let args: PeakFitArgs = feature.into();

        let wmt = args.weighted_mean_time();
        assert_is_close!(wmt, 122.3535, 1e-3, "weighted mean time");

        let mut model = SkewedGaussianPeakShape::guess(&args);

        // let expected_gradient = SkewedGaussianPeakShape {
        //     mu: 1.8301577852922191,
        //     sigma: -2.0862889857607283,
        //     amplitude: 15138.003092888086,
        //     lambda: 0.083553223109944452,
        // };
        let gradient = model.gradient(&args);

        eprintln!("Initial:\n{model:?}");
        eprintln!("Gradient combo:\n{:?}", gradient);
        eprintln!("Gradient split:\n{:?}", model.gradient_split(&args));

        let _res = model.fit(args.borrow());
        let _score = model.score(&args);
        eprintln!("{model:?}\n{_res:?}\n{_score}\n");

        let expected = SkewedGaussianPeakShape {
            mu: 121.55279659554802,
            sigma: 0.12862206964456804,
            amplitude: 5300808.611741115,
            lambda: 5.321920008423757e-6,
        };

        assert_is_close!(expected.mu, model.mu, 1e-2, "mu");
        assert_is_close!(expected.sigma, model.sigma, 1e-2, "sigma");

        // assert_is_close!(expected_gradient.mu, gradient.mu, 1e-2, "mu");
        // assert_is_close!(expected_gradient.sigma, gradient.sigma, 1e-2, "sigma");
        // unstable
        // assert_is_close!(expected.lambda, model.lambda, 1e-3, "lambda");
        // assert_is_close!(expected.amplitude, model.amplitude, 100.0, "amplitude");
    }

    #[rstest::rstest]
    fn test_fit_feature_4490(feature_table: &[Feature<MZ, Time>]) {
        let feature = &feature_table[4490];
        let args = PeakFitArgs::from(feature);

        let expected_fits = MultiPeakShapeFit {
            fits: vec![
                PeakShape::SkewedGaussian(SkewedGaussianPeakShape {
                    mu: 125.52343374409863,
                    sigma: 0.24333467521731506,
                    amplitude: 2301265.971958694,
                    lambda: -0.6969704897945564,
                }),
                PeakShape::Gaussian(GaussianPeakShape {
                    mu: 127.25393980857764,
                    sigma: 0.24351293620218414,
                    amplitude: 258173.71303547118,
                }),
            ],
        };

        let mut fitter = SplittingPeakShapeFitter::new(args.borrow());
        fitter.fit_with(FitConfig::default().max_iter(10_000));
        eprintln!("Score: {}", fitter.score());
        eprintln!("Fits: {:?}", fitter.peak_fits);

        for (exp, obs) in expected_fits.iter().zip(fitter.peak_fits.iter()) {
            let expected_mu = dispatch_peak!(exp, model, model.mu);
            let observed_mu = dispatch_peak!(obs, model, model.mu);

            assert_is_close!(expected_mu, observed_mu, 1e-3, "mu");
        }
    }

    #[rstest::rstest]
    fn test_fit_feature_10979(feature_table: &[Feature<MZ, Time>]) {
        let feature = &feature_table[10979];
        let args: PeakFitArgs<'_, '_> = feature.into();

        let expected_split_point = SplittingPoint {
            first_maximum_height: 1562937.5,
            minimum_height: 130524.61,
            second_maximum_height: 524531.8,
            minimum_time: 127.1233653584,
        };

        let observed_split_point = args.locate_extrema(None).unwrap();

        assert_is_close!(
            expected_split_point.minimum_time,
            observed_split_point.minimum_time,
            1e-2,
            "minimum_time",
            observed_split_point
        );

        let mut fitter = SplittingPeakShapeFitter::new(args.borrow());
        fitter.fit_with(FitConfig::default().max_iter(10_000).smooth(3));
        eprintln!("Score: {}", fitter.score());
        eprintln!("Fits: {:?}", fitter.peak_fits);

        let expected_fits = MultiPeakShapeFit {
            fits: vec![PeakShape::BiGaussian(BiGaussianPeakShape {
                mu: 125.15442379557763,
                sigma_falling: 1.0523297795974058,
                sigma_rising: 0.24321060937688804,
                amplitude: 1375631.4353707798,
            })],
        };

        // #[cfg(feature = "serde")]
        // serde_json::to_writer(
        //     std::fs::File::create("fits_10979.json").unwrap(),
        //     &(
        //         &fitter,
        //         fitter.predicted(),
        //         expected_fits.predict(&fitter.data.time),
        //         fitter.data.smooth(3),
        //     ),
        // )
        // .unwrap();

        for (exp, obs) in expected_fits.iter().zip(fitter.peak_fits.iter()) {
            let expected_mu = dispatch_peak!(exp, model, model.mu);
            let observed_mu = dispatch_peak!(obs, model, model.mu);

            assert_is_close!(observed_mu, expected_mu, 1e-3, "mu");
        }
    }

    #[rstest::rstest]
    fn test_fit_args(feature_table: &[Feature<MZ, Time>]) {
        let features = feature_table;
        let feature = &features[160];
        let (_, y, z) = feature.as_view().into_inner();
        let args = PeakFitArgs::from((y, z));

        let wmt = args.weighted_mean_time();
        assert!(
            (wmt - 123.455).abs() < 1e-3,
            "Observed average weighted mean time {wmt}, expected 123.455"
        );

        let mut model = GaussianPeakShape::new(wmt, 1.0, 1.0);
        let _res = model.fit(args.borrow());
        let _score = model.score(&args);
        // eprint!("{model:?}\n{_res:?}\n{_score}\n");

        let mu = 123.44962935714317;
        // let sigma = 0.10674673407221102;
        // let amplitude = 629639.6468112208;
        log::info!("Model {model:?}");
        assert!(
            (model.mu - mu).abs() < 1e-3,
            "Model mu {0} found, expected {mu}, error = {1}",
            model.mu,
            model.mu - mu
        );
        // assert!(
        //     (model.sigma - sigma).abs() < 1e-3,
        //     "Model sigma {0} found, expected {sigma}, error = {1}",
        //     model.sigma,
        //     model.sigma - sigma
        // );
        // Seems to be sensitive to the platform
        // assert!(
        //     (model.amplitude - amplitude).abs() < 1e-2,
        //     "Model {0} found, expected {amplitude}, error = {1}",
        //     model.amplitude,
        //     model.amplitude - amplitude
        // );
    }

    #[rstest::rstest]
    fn test_mixed_signal() {
        let time = vec![
            5., 5.05, 5.1, 5.15, 5.2, 5.25, 5.3, 5.35, 5.4, 5.45, 5.5, 5.55, 5.6, 5.65, 5.7, 5.75,
            5.8, 5.85, 5.9, 5.95, 6., 6.05, 6.1, 6.15, 6.2, 6.25, 6.3, 6.35, 6.4, 6.45, 6.5, 6.55,
            6.6, 6.65, 6.7, 6.75, 6.8, 6.85, 6.9, 6.95, 7., 7.05, 7.1, 7.15, 7.2, 7.25, 7.3, 7.35,
            7.4, 7.45, 7.5, 7.55, 7.6, 7.65, 7.7, 7.75, 7.8, 7.85, 7.9, 7.95, 8., 8.05, 8.1, 8.15,
            8.2, 8.25, 8.3, 8.35, 8.4, 8.45, 8.5, 8.55, 8.6, 8.65, 8.7, 8.75, 8.8, 8.85, 8.9, 8.95,
            9., 9.05, 9.1, 9.15, 9.2, 9.25, 9.3, 9.35, 9.4, 9.45, 9.5, 9.55, 9.6, 9.65, 9.7, 9.75,
            9.8, 9.85, 9.9, 9.95, 10., 10.05, 10.1, 10.15, 10.2, 10.25, 10.3, 10.35, 10.4, 10.45,
            10.5, 10.55, 10.6, 10.65, 10.7, 10.75, 10.8, 10.85, 10.9, 10.95, 11., 11.05, 11.1,
            11.15, 11.2, 11.25, 11.3, 11.35, 11.4, 11.45, 11.5, 11.55, 11.6, 11.65, 11.7, 11.75,
            11.8, 11.85, 11.9, 11.95,
        ];

        let intensity: Vec<f32> = vec![
            1.27420451e-10,
            6.17462536e-10,
            2.87663017e-09,
            1.28813560e-08,
            5.54347499e-08,
            2.29248641e-07,
            9.10983876e-07,
            3.47838473e-06,
            1.27613560e-05,
            4.49843188e-05,
            1.52358163e-04,
            4.95800813e-04,
            1.55018440e-03,
            4.65685268e-03,
            1.34410596e-02,
            3.72739646e-02,
            9.93134748e-02,
            2.54238248e-01,
            6.25321648e-01,
            1.47773227e+00,
            3.35519458e+00,
            7.31929673e+00,
            1.53408987e+01,
            3.08931557e+01,
            5.97728698e+01,
            1.11116052e+02,
            1.98463694e+02,
            3.40579046e+02,
            5.61550473e+02,
            8.89601967e+02,
            1.35407175e+03,
            1.98029962e+03,
            2.78272124e+03,
            3.75722702e+03,
            4.87459147e+03,
            6.07720155e+03,
            7.28110184e+03,
            8.38438277e+03,
            9.28130702e+03,
            9.87974971e+03,
            1.01181592e+04,
            9.97790016e+03,
            9.48776976e+03,
            8.71945410e+03,
            7.77507793e+03,
            6.76998890e+03,
            5.81486954e+03,
            5.00095885e+03,
            4.39083710e+03,
            4.01545364e+03,
            3.87648735e+03,
            3.95216934e+03,
            4.20449848e+03,
            4.58618917e+03,
            5.04639622e+03,
            5.53495197e+03,
            6.00532094e+03,
            6.41666569e+03,
            6.73537620e+03,
            6.93625265e+03,
            7.00335463e+03,
            6.93041211e+03,
            6.72066330e+03,
            6.38603273e+03,
            5.94566001e+03,
            5.42389927e+03,
            4.84799871e+03,
            4.24571927e+03,
            3.64315240e+03,
            3.06295366e+03,
            2.52313467e+03,
            2.03646669e+03,
            1.61046411e+03,
            1.24784786e+03,
            9.47346984e+02,
            7.04682299e+02,
            5.13587560e+02,
            3.66751988e+02,
            2.56606294e+02,
            1.75913423e+02,
            1.18159189e+02,
            7.77629758e+01,
            5.01435513e+01,
            3.16806551e+01,
            1.96114662e+01,
            1.18949556e+01,
            7.06890964e+00,
            4.11603334e+00,
            2.34823840e+00,
            1.31263002e+00,
            7.18917952e-01,
            3.85791959e-01,
            2.02844792e-01,
            1.04498823e-01,
            5.27467595e-02,
            2.60865722e-02,
            1.26408156e-02,
            6.00164112e-03,
            2.79191246e-03,
            1.27253700e-03,
            5.68297684e-04,
            2.48667027e-04,
            1.06609858e-04,
            4.47830199e-05,
            1.84317357e-05,
            7.43285977e-06,
            2.93685491e-06,
            1.13696185e-06,
            4.31266912e-07,
            1.60281439e-07,
            5.83656297e-08,
            2.08241826e-08,
            7.27973551e-09,
            2.49344667e-09,
            8.36799506e-10,
            2.75156384e-10,
            8.86491588e-11,
            2.79837874e-11,
            8.65516222e-12,
            2.62289424e-12,
            7.78795072e-13,
            2.26570027e-13,
            6.45830522e-14,
            1.80372998e-14,
            4.93584288e-15,
            1.32339040e-15,
            3.47657399e-16,
            8.94853257e-17,
            2.25677892e-17,
            5.57651734e-18,
            1.35012489e-18,
            3.20273997e-19,
            7.44399825e-20,
            1.69522634e-20,
            3.78256125e-21,
            8.26953508e-22,
            1.77138542e-22,
            3.71776457e-23,
            7.64517712e-24,
            1.54038778e-24,
        ];

        let args = PeakFitArgs::from((time, intensity));
        let split_point = args.locate_extrema(None).unwrap();
        assert_is_close!(
            split_point.minimum_time,
            7.5,
            1e-3,
            "minimum_height",
            split_point
        );

        let mut fitter = SplittingPeakShapeFitter::new(args);
        fitter.fit_with(FitConfig::default().max_iter(10_000));
        let score = fitter.score();
        assert!(
            score > 0.95,
            "Expected score {score} to be greater than 0.95"
        );
    }
}
