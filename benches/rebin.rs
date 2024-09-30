use criterion::{black_box, Criterion};

use mzsignal::{average, text};


fn direct_indexing(averager: &average::SignalAverager) -> f32 {
    let yhat = averager.interpolate();
    black_box(yhat.into_iter().sum())
}

fn iter_mixing(averager: &average::SignalAverager) -> f32 {
    let yhat = averager.interpolate_iter();
    black_box(yhat.into_iter().sum())
}

fn averaging(c: &mut Criterion) {
    let arrays = text::arrays_from_file("test/0.txt").unwrap();
    let mut averager = average::SignalAverager::new(arrays.min_mz, arrays.max_mz, 0.001);
    averager.push(arrays.borrow());

    c.bench_function("direct_indexing", |b| {
        b.iter(|| direct_indexing(&averager))
    });

    c.bench_function("iter_mixing", |b| {
        b.iter(|| iter_mixing(&averager))
    });
}


criterion::criterion_group!(benches, averaging);
criterion::criterion_main!(benches);