use num_traits::Float;

type MZ = f64;

/// Perform a linear local search looking to the left (decreasing)
pub fn nearest_left<T: Float>(vec: &[T], target_val: T, start_index: usize) -> usize {
    let mut nearest_index = start_index;
    let mut next_index = start_index;
    if next_index == 0 {
        return 0;
    }
    let next_val = vec[next_index];
    let mut best_distance = (next_val - target_val).abs();
    while next_val > target_val {
        next_index -= 1;
        let next_val = vec[next_index];
        let dist = (next_val - target_val).abs();
        if dist < best_distance {
            best_distance = best_distance;
            nearest_index = next_index;
        }
        if next_index == 0 {
            break;
        }
    }
    nearest_index
}

/// Perform a linear local search looking to the right (increasing)
pub fn nearest_right<T: Float>(vec: &[T], target_val: T, start_index: usize) -> usize {
    let mut nearest_index = start_index;
    let mut next_index = start_index;
    let n = vec.len() - 1;
    if next_index >= n {
        return n;
    }
    let next_val = vec[next_index];
    let mut best_distance = (next_val - target_val).abs();
    while next_val < target_val {
        next_index += 1;
        let dist = (next_val - target_val).abs();
        if dist < best_distance {
            best_distance = best_distance;
            nearest_index = next_index;
        }
        if next_index == n {
            break;
        }
    }
    nearest_index
}

pub fn nearest_binary<T: Float>(
    vec: &[T],
    target_val: T,
    start_index: usize,
    stop_index: usize,
) -> usize {
    let mut start_index = start_index;
    let mut stop_index = stop_index;

    let cval = vec[start_index];
    if cval > target_val {
        return start_index;
    }
    loop {
        let min_val = vec[start_index];
        let max_val = vec[stop_index];
        if (stop_index - start_index) <= 1 && (target_val >= min_val) && (target_val <= max_val) {
            if (min_val - target_val).abs() < (max_val - target_val).abs() {
                return start_index;
            } else {
                return stop_index;
            }
        }
        let ratio =
            (max_val - target_val).to_f64().unwrap() / (max_val - min_val).to_f64().unwrap();
        // Interpolation search
        let mid_index = (start_index as f64 * ratio + stop_index as f64 * (1.0 - ratio)) as usize;

        let mid_val = vec[mid_index];

        if mid_val >= target_val {
            stop_index = mid_index;
        } else if mid_index + 1 == stop_index {
            if (mid_val - target_val).abs() < (max_val - target_val).abs() {
                return mid_index;
            } else {
                return stop_index;
            }
        } else {
            let mid_next_val = vec[mid_index + 1];
            if target_val >= mid_val && target_val <= mid_next_val {
                if (target_val - mid_val) < (mid_next_val - mid_val) {
                    return mid_index;
                }
                return mid_index + 1;
            }
            start_index = mid_index + 1;
        }
    }
}

pub fn nearest(vec: &[MZ], target_val: MZ, _start_index: usize) -> usize {
    let n = vec.len() - 1;

    if target_val > vec[n] {
        return n;
    } else if target_val < vec[0] {
        return 0;
    }

    // let (domain, is_lower) = if target_val <= vec[start_index] {
    //     (&vec[..start_index + 1], true)
    // } else {
    //     (&vec[start_index..], false)
    // };

    let near = match vec.binary_search_by(|x| x.partial_cmp(&target_val).unwrap()) {
        Ok(i) => i,
        Err(i) => i,
    };
    if near <= n {
        if vec[near] <= target_val {
            nearest_right(vec, target_val, near)
        } else {
            nearest_left(vec, target_val, near)
        }
    } else {
        n
    }
}

pub fn binsearch<T: Float>(array: &[T], q: T) -> usize {
    match array.binary_search_by(|x| x.partial_cmp(&q).unwrap()) {
        Ok(i) => i,
        Err(i) => i,
    }
}

pub fn find_between<T: Float>(array: &[T], lo: T, hi: T) -> (usize, usize) {
    let n = array.len();
    let mut lo_i = binsearch(array, lo);
    if lo_i == n {
        lo_i -= 1;
    }
    if lo - array[lo_i] > T::from(0.1).unwrap() {
        if lo_i < array.len() - 1 {
            lo_i += 1;
        }
    }
    let mut hi_i = binsearch(array, hi);
    if hi_i == n {
        hi_i -= 1;
    }
    if array[hi_i] - hi > T::from(0.1).unwrap() {
        if hi_i > 0 {
            hi_i -= 1;
        }
    }
    if lo_i > hi_i {
        hi_i = lo_i;
    }
    return (lo_i, hi_i);
}
