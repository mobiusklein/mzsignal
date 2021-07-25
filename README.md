# mzsignal - Low Level Signal Processing For Mass Spectra

## Usage

```rust
use mzsignal::pick_peaks

```

## Building
This library depends upon `ndarray-linalg`, which means it needs a LAPACK implementation
as a backend for `ndarray-linalg`. These are enabled by passing one of the supported backends
as a `feature` to `cargo` e.g.:

```bash
cargo test --features intel-mkl
```
