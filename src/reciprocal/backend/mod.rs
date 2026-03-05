/// Scalar and SIMD backends for reciprocal-space Ewald inner loops.
///
/// All hot loops iterate over k-vectors computing `sin_cos(k·r)` and accumulating
/// structure factors. The scalar backend uses `f64::sin_cos`; the SIMD backend
/// processes 4 k-vectors per iteration using `wide::f64x4`.

#[cfg_attr(feature = "simd", allow(dead_code))]
pub(super) mod scalar;

#[cfg(feature = "simd")]
pub(super) mod simd;
