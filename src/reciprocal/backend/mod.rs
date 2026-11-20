//! Scalar and SIMD backends for reciprocal-space Ewald inner loops.
//!
//! k-vectors and coefficients are stored as f32, halving memory bandwidth;
//! structure factors and energy sums are f64.
//!
//! The scalar backend factorizes e^{ik·r} over axes (see `Phases`), so it needs
//! three f64 sin_cos calls per particle rather than one per k-vector.
//!
//! The SIMD backend (x86_64 only, `simd` feature) instead evaluates a polynomial
//! sin_cos per k-vector in f32.

#[cfg_attr(all(feature = "simd", target_arch = "x86_64"), allow(dead_code))]
pub(super) mod scalar;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub(super) mod simd;
