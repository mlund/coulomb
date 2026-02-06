// Copyright 2023 Björn Stenqvist and Mikael Lund
//
// Converted to Rust with modification from the C++ library "CoulombGalore":
// https://zenodo.org/doi/10.5281/zenodo.3522058
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.
// You may obtain a copy of the license at
//
//     http://www.apache.org/licenses/license-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the license is distributed on an "as is" basis,
// without warranties or conditions of any kind, either express or implied.
// See the license for the specific language governing permissions and
// limitations under the license.

//! Pairwise interaction schemes implementing the `ShortRangeFunction` trait.

pub(crate) mod ewald;
pub(crate) mod ewald_truncated;
pub(crate) mod plain;
pub(crate) mod poisson;
pub(crate) mod reactionfield;

/// Test utilities for pairwise schemes
#[cfg(test)]
pub(crate) mod test_utils {
    /// Assert that a vector (convertible to Vector3) has zero magnitude
    macro_rules! assert_vec_zero {
        ($vec:expr, $eps:expr) => {{
            let v: crate::NalgebraVector3 = $vec.into();
            approx::assert_relative_eq!(v.norm(), 0.0, epsilon = $eps);
        }};
    }

    /// Assert vector components match expected [x, y, z] values
    macro_rules! assert_vec3_eq {
        ($vec:expr, [$x:expr, $y:expr, $z:expr], $eps:expr) => {{
            let v: crate::NalgebraVector3 = $vec.into();
            approx::assert_relative_eq!(v[0], $x, epsilon = $eps);
            approx::assert_relative_eq!(v[1], $y, epsilon = $eps);
            approx::assert_relative_eq!(v[2], $z, epsilon = $eps);
        }};
    }

    /// Assert vector x-component equals its norm (for vectors along x-axis)
    macro_rules! assert_vec_x_equals_norm {
        ($vec:expr, $expected:expr, $eps:expr) => {{
            let v: crate::NalgebraVector3 = $vec.into();
            approx::assert_relative_eq!(v[0], $expected, epsilon = $eps);
            approx::assert_relative_eq!(v.norm(), $expected, epsilon = $eps);
        }};
    }

    pub(crate) use assert_vec3_eq;
    pub(crate) use assert_vec_x_equals_norm;
    pub(crate) use assert_vec_zero;
}
