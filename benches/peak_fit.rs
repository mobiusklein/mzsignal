use criterion::Criterion;

use mzsignal::{feature_statistics::{BiGaussianPeakShape, SkewedGaussianPeakShape, FitPeaksOn}, text};


fn bigaussian_fitting(c: &mut Criterion) {
    let features = text::load_feature_table("test/data/features_graph.txt").unwrap();
    let feature = &features[10979];
    let args = feature.as_peak_shape_args().smooth(1);

    let init = BiGaussianPeakShape::guess(&args);
    c.bench_function("bigaussian_reference", |b| {
        b.iter(|| init.gradient_split(&args))
    });
    c.bench_function("bigaussian_optimized", |b| {
        b.iter(|| init.gradient(&args))
    });
}


fn skewed_gaussian_fitting(c: &mut Criterion) {
    let features = text::load_feature_table("test/data/features_graph.txt").unwrap();
    let feature = &features[10979];
    let args = feature.as_peak_shape_args().smooth(1);

    let init = SkewedGaussianPeakShape::guess(&args);
    c.bench_function("skewed_gaussian_reference", |b| {
        b.iter(|| init.gradient_split(&args))
    });
    c.bench_function("skewed_gaussian_optimized", |b| {
        b.iter(|| init.gradient(&args))
    });
}


fn fitting(c: &mut Criterion) {
    bigaussian_fitting(c);
    skewed_gaussian_fitting(c);
}


criterion::criterion_group!(benches, fitting);
criterion::criterion_main!(benches);