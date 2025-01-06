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

#[cfg(test)]
use approx::assert_relative_eq;

/// Approximation of erfc-function
///
/// # Arguments
/// * `x` - Value for which erfc should be calculated
///
/// # Details
/// Reference for this approximation is found in Abramowitz and Stegun,
/// Handbook of mathematical functions, eq. 7.1.26
///
/// erf(x) = 1 - (a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5)e^{-x^2} + epsilon(x)
/// t = 1 / (1 + px)
/// |epsilon(x)| <= 1.5 * 10^-7
///
/// # Warning
/// Needs modification if x < 0
#[inline]
pub(crate) fn erfc_x(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5)))) * f64::exp(-x * x)
}

/// Approximation of erf-function, see `erfc_x` for details
#[inline]
pub(crate) fn erf_x(x: f64) -> f64 {
    1.0 - erfc_x(x)
}

pub(crate) fn _q_pochhammer_symbol(q: f64, l: i32, p: i32) -> f64 {
    let ct = (1..=p)
        .map(|n| (1..=(n + l)).map(|k| q.powi(k - 1)).sum::<f64>())
        .product::<f64>();
    let dt = (1.0 - q).powi(p);
    ct * dt
}

/// Computes the derivative of the q-Pochhammer Symbol.
pub(crate) fn _q_pochhammer_symbol_derivative(q: f64, l: i32, p: i32) -> f64 {
    let ct = (1..=p)
        .map(|n| (1..=(n + l)).map(|k| q.powi(k - 1)).sum::<f64>())
        .product::<f64>();

    // evaluates to derivative of ∏(∑(q^k), k = 0 to n+l, n = 1 to P)
    let fraction = |n: i32| -> f64 {
        let nom = (2..=(n + l))
            .map(|k| f64::from(k - 1) * q.powi(k - 2))
            .sum::<f64>();
        let denom = 1.0 + (2..=(n + l)).map(|k| q.powi(k - 1)).sum::<f64>();
        nom / denom
    };
    let dct = (1..=p).map(fraction).sum::<f64>();
    let dt = (1.0 - q).powi(p); // (1-q)^P
    let ddt = if p > 0 {
        f64::from(-p) * (1.0 - q).powi(p - 1) // derivative of (1-q)^P
    } else {
        0.0
    };
    ct * (ddt + dct * dt)
}

pub(crate) fn _q_pochhammer_symbol_second_derivative(q: f64, l: i32, p: i32) -> f64 {
    let mut ct = 1.0; // evaluates to \prod_{n=1}^P\sum_{k=0}^{n+l}q^k
    let mut ds = 0.0;
    let mut d_ds = 0.0;
    for n in 1..=p {
        let mut tmp = 0.0;
        for k in 1..=n + l {
            tmp += q.powi(k - 1);
        }
        ct *= tmp;
        let mut nom = 0.0;
        let mut denom = 1.0;
        for k in 2..=n + l {
            nom += (k - 1) as f64 * q.powi(k - 2);
            denom += q.powi(k - 1);
        }
        ds += nom / denom;
        let mut diff_nom = 0.0;
        let mut diff_denom = 1.0;
        for k in 3..=n + l {
            diff_nom += (k - 1) as f64 * (k - 2) as f64 * q.powi(k - 3);
            diff_denom += (k - 1) as f64 * q.powi(k - 2);
        }
        d_ds += (diff_nom * denom - nom * diff_denom) / denom / denom;
    }
    let d_ct = ct * ds; // derivative of \prod_{n=1}^P\sum_{k=0}^{n+l}q^k
    let dd_ct = d_ct * ds + ct * d_ds; // second derivative of \prod_{n=1}^P\sum_{k=0}^{n+l}q^k
    let dt = (1.0 - q).powi(p); // (1-q)^P
    let d_dt = if p > 0 {
        -p as f64 * (1.0 - q).powi(p - 1)
    } else {
        0.0
    }; // derivative of (1-q)^P
    let dd_dt = if p > 1 {
        p as f64 * (p - 1) as f64 * (1.0 - q).powi(p - 2)
    } else {
        0.0
    }; // second derivative of (1-q)^P
    ct * dd_dt + 2.0 * d_ct * d_dt + dd_ct * dt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q_pochhammer_symbol() {
        assert_eq!(_q_pochhammer_symbol(0.5, 0, 0), 1.0);
        assert_eq!(_q_pochhammer_symbol(0.0, 0, 1), 1.0);
        assert_eq!(_q_pochhammer_symbol(1.0, 0, 1), 0.0);
        assert_eq!(_q_pochhammer_symbol(1.0, 1, 2), 0.0);
        assert_relative_eq!(_q_pochhammer_symbol(0.75, 0, 2), 0.109375);
        assert_relative_eq!(
            _q_pochhammer_symbol(2.0 / 3.0, 2, 5),
            0.4211104676,
            epsilon = 1e-5
        );
        assert_relative_eq!(_q_pochhammer_symbol(0.125, 1, 1), 0.984375);
        assert_relative_eq!(_q_pochhammer_symbol_derivative(0.75, 0, 2), -0.8125);
        assert_relative_eq!(
            _q_pochhammer_symbol_derivative(2.0 / 3.0, 2, 5),
            -2.538458169,
            epsilon = 1e-5
        );
        assert_relative_eq!(_q_pochhammer_symbol_derivative(0.125, 1, 1), -0.25);
        assert_relative_eq!(_q_pochhammer_symbol_second_derivative(0.75, 0, 2), 2.5);
        assert_relative_eq!(
            _q_pochhammer_symbol_second_derivative(2.0 / 3.0, 2, 5),
            -1.444601767,
            epsilon = 1e-5
        );
        assert_relative_eq!(_q_pochhammer_symbol_second_derivative(0.125, 1, 1), -2.0);
        // assert_relative_eq!(q_pochhammer_symbol_third_derivative(0.75, 0, 2), 6.0);
        // assert_relative_eq!(
        //     q_pochhammer_symbol_third_derivative(2.0 / 3.0, 2, 5),
        //     92.48631425
        // );
        // assert_relative_eq!(q_pochhammer_symbol_third_derivative(0.125, 1, 1), 0.0);
        // assert_relative_eq!(
        //     q_pochhammer_symbol_third_derivative(0.4, 3, 7),
        //     -32.80472205);
    }
}
