//! Scalar and SIMD backends for reciprocal-space Ewald inner loops.
//!
//! k-vectors and coefficients are stored as f32, halving memory bandwidth;
//! structure factors and energy sums are f64.
//!
//! e^{ik·r} factorizes over axes (see `Phases`), so a particle needs three f64
//! sin_cos calls rather than one per k-vector. That beats a per-k-vector SIMD
//! polynomial on both x86_64 and aarch64.

pub(super) mod scalar;
