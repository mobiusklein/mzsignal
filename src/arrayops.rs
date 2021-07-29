use std::ops::{Add, Index};
use std::iter::Sum;

use num_traits::{Float, AsPrimitive, Zero, ToPrimitive};

pub fn gridspace<T: Float + ToPrimitive>(start: T, end: T, step: T) -> Vec<T> {
    let distance = end - start;
    let steps = (distance / step).to_usize().unwrap();
    let mut result = Vec::with_capacity(steps);
    for i in 0..steps {
        result.push(start + T::from(i).unwrap() * step);
    }
    result
}


pub fn trapz<A: Float + Clone + AsPrimitive<B> + 'static, B: Float + Clone + AsPrimitive<A> + 'static + Sum>(x: &[A], y: &[B]) -> B {
    // let result = B::from(0.0).unwrap();
    let n = x.len();
    (0..n - 2).map(|i| {
        let delta = x[i + 1] - x[i];
        delta.as_() * B::from(0.5).unwrap() * (y[i + 1] + y[i])
    }).sum()
}
