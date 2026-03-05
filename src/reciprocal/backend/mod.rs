//! Scalar and SIMD backends for reciprocal-space Ewald inner loops.
//!
//! Uses mixed precision: k-vectors and coefficients are stored as f32 (halving
//! memory bandwidth), trigonometric evaluations use f32 arithmetic, while
//! structure factor accumulators and energy sums use f64 for numerical safety.
//!
//! The SIMD backend (x86_64 only, `simd` feature) uses a polynomial sin_cos
//! that avoids per-lane glibc calls. On Apple Silicon, the platform libm is
//! faster than any polynomial, so the scalar backend is used unconditionally.

#[cfg_attr(all(feature = "simd", target_arch = "x86_64"), allow(dead_code))]
pub(super) mod scalar;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub(super) mod simd;
