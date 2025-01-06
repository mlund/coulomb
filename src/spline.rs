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

//! Spline functions
//!
//! ## Todo
//!
//! This is a conversion from Fortran → C++ → Rust 😱
//! Serious refactoring is needed!

#![allow(dead_code)]

use anyhow::Result;
#[cfg(test)]
use approx::assert_relative_eq;

use itertools::{Itertools, Position};
use std::f64;
use std::iter::zip;
use std::vec::Vec;

// First derivative with respect to x
fn first_derivative(func: impl Fn(f64) -> f64, x: f64, dx: f64) -> f64 {
    (func(x + dx * 0.5) - func(x - dx * 0.5)) / dx
}

// Second derivative with respect to x
fn second_derivative(func: impl Fn(f64) -> f64, x: f64, dx: f64) -> f64 {
    (first_derivative(&func, x + dx * 0.5, dx) - first_derivative(&func, x - dx * 0.5, dx)) / dx
}

/// Returns the 0th, 1st, and 2nd derivative at x
fn derivatives(func: impl Fn(f64) -> f64, x: f64, dx: f64) -> (f64, f64, f64) {
    (
        func(x),
        first_derivative(&func, x, dx),
        second_derivative(&func, x, dx),
    )
}

#[derive(Default, PartialEq, Clone)]
pub struct Knots {
    r2: Vec<f64>,    // r2 for intervals
    coeff: Vec<f64>, // c for coefficients
    pub rmin2: f64,  // useful to save these with table
    pub rmax2: f64,
}

impl Knots {
    pub fn is_empty(&self) -> bool {
        self.r2.is_empty() && self.coeff.is_empty()
    }

    pub fn len(&self) -> usize {
        self.r2.len()
    }
}

/* base class for all tabulators - no dependencies */
#[derive(PartialEq, Clone)]
pub struct Spline {
    /// Tolerance on the spline values
    tolerance: f64,
    /// Tolerance on the derivative of the spline values
    derivative_tolerance: f64,
    /// Step size used for numerical differentiation
    diff_step_size: f64,
    /// Spline knots
    pub knots: Knots,
}

impl Default for Spline {
    fn default() -> Self {
        Spline {
            tolerance: 1.0e-5,
            derivative_tolerance: -1.0,
            diff_step_size: 0.0001,
            knots: Knots::default(),
        }
    }
}

impl Spline {
    pub fn check_tolerance(&self) -> Result<(), anyhow::Error> {
        if self.derivative_tolerance != -1.0 && self.derivative_tolerance <= 0.0 {
            return Err(anyhow::Error::msg("ftol too small"));
        }
        Ok(())
    }

    /// Set the tolerance for the spline values and the derivative of the spline values
    pub fn set_tolerance(&mut self, utol: f64, ftol: f64) {
        self.tolerance = utol;
        self.derivative_tolerance = ftol;
    }

    pub fn set_differentiation_step_size(&mut self, step_size: f64) {
        self.diff_step_size = step_size;
    }
}

/// Andrea table with logarithmic search
///
/// Code mainly from MolSim (Per Linse) with some upgrades
/// Reference: doi:10/frzp4d
#[derive(Clone, PartialEq)]
pub struct Andrea {
    spline: Spline,
    /// Max number of control points
    max_num_ctrl_points: u32,
    /// Max number of trials to downscale dr
    max_num_downscales: u32,
    /// Downscaling factor for dr
    downscale_factor: f64,
}

impl Default for Andrea {
    fn default() -> Self {
        Self {
            spline: Spline::default(),
            max_num_ctrl_points: 1200,
            max_num_downscales: 100,
            downscale_factor: 0.9,
        }
    }
}

impl Andrea {
    pub fn get_coefficients(
        lower_x: f64,
        upper_x: f64,
        lower_diff: (f64, f64, f64),
        upper_diff: (f64, f64, f64),
    ) -> Vec<f64> {
        // Zero potential and force return no coefficients
        if lower_diff.0.abs() < 1e-9 && lower_diff.1.abs() < 1e-9 {
            return vec![0.0; 6];
        }

        let dx = upper_x - lower_x;
        let dx_squared = dx * dx;
        let dx_cubed = dx_squared * dx;
        let a = 6.0
            * (upper_diff.0 - lower_diff.0 - lower_diff.1 * dx - 0.5 * lower_diff.2 * dx_squared)
            / dx_cubed;
        let b = 2.0 * (upper_diff.1 - lower_diff.1 - lower_diff.2 * dx) / dx_squared;
        let c = (upper_diff.2 - lower_diff.2) / dx;
        let c3 = (10.0 * a - 12.0 * b + 3.0 * c) / 6.0;
        let c4 = (-15.0 * a + 21.0 * b - 6.0 * c) / (6.0 * dx);
        let c5 = (2.0 * a - 3.0 * b + c) / (2.0 * dx_squared);

        vec![
            lower_x,
            lower_diff.0,
            lower_diff.1,
            0.5 * lower_diff.2,
            c3,
            c4,
            c5,
        ]
    }

    pub fn check_coefficients(
        &self,
        ubuft: &[f64],
        rlow: f64,
        rupp: f64,
        func: impl Fn(f64) -> f64,
    ) -> (bool, bool) {
        // Number of points to control
        let ncheck = 11;
        let dr = (rupp - rlow) / (ncheck - 1) as f64;
        let mut error_codes = (false, false);

        for i in 0..ncheck {
            let r1 = rlow + dr * (i as f64);
            let r2 = r1 * r1;
            let u0 = func(r2);
            let u1 = first_derivative(&func, r2, self.spline.diff_step_size);
            let dz = r2 - rlow * rlow;
            let sum = ubuft[1]
                + dz * (ubuft[2]
                    + dz * (ubuft[3] + dz * (ubuft[4] + dz * (ubuft[5] + dz * ubuft[6]))));

            let derivative_sum = ubuft[2]
                + dz * (2.0 * ubuft[3]
                    + dz * (3.0 * ubuft[4] + dz * (4.0 * ubuft[5] + dz * (5.0 * ubuft[6]))));

            if (sum - u0).abs() > self.spline.tolerance {
                return error_codes;
            }

            if self.spline.derivative_tolerance != -1.0
                && (derivative_sum - u1).abs() > self.spline.derivative_tolerance
            {
                return error_codes;
            }
        }

        error_codes.0 = true;
        error_codes
    }

    pub fn generate<F>(&mut self, func: &F, xmin_squared: f64, xmax_squared: f64) -> Result<Knots>
    where
        F: Fn(f64) -> f64,
    {
        self.spline.check_tolerance().unwrap();

        let mut knots = Knots {
            rmin2: xmin_squared,
            rmax2: xmax_squared,
            r2: Vec::new(),
            coeff: Vec::new(),
        };

        let xmin = f64::sqrt(xmin_squared);
        let mut rumin = xmin;
        let mut highx = f64::sqrt(xmax_squared);
        let mut highx_squared = xmax_squared;
        let mut high_repulsion = false;

        knots.r2.push(highx_squared);

        for attempt in (0..self.max_num_ctrl_points).with_position() {
            let mut lowx = highx;
            let mut lowx_squared = 0.0;
            let mut coeff: Vec<f64> = Vec::new();

            let mut dx = highx - xmin;

            for attempt in (0..self.max_num_downscales).with_position() {
                highx_squared = highx * highx;
                lowx = highx - dx;
                if rumin > lowx {
                    lowx = rumin;
                }

                lowx_squared = lowx * lowx;

                let low_diff = derivatives(func, lowx_squared, self.spline.diff_step_size);
                let upp_diff = derivatives(func, highx_squared, self.spline.diff_step_size);
                coeff = Andrea::get_coefficients(lowx_squared, highx_squared, low_diff, upp_diff);

                let error_codes = self.check_coefficients(&coeff, lowx, highx, func);
                high_repulsion = error_codes.1;
                if error_codes.0 {
                    highx = lowx;
                    break;
                }
                dx *= self.downscale_factor;
                if let Position::Last(_) = attempt {
                    return Err(anyhow::Error::msg("increase tolerance"));
                }
            }

            if coeff.len() != 7 {
                return Err(anyhow::Error::msg("invalid number of coefficients"));
            }

            knots.r2.push(lowx_squared);
            knots.coeff.extend(coeff.iter().skip(1));

            if high_repulsion {
                rumin = lowx;
                knots.rmin2 = lowx * lowx;
            }
            if lowx <= rumin || high_repulsion {
                break;
            }
            if let Position::Last(_) = attempt {
                return Err(anyhow::Error::msg("increase tolerance"));
            }
        }

        Self::swap_coefficients(&mut knots.coeff);
        knots.r2.reverse();
        Ok(knots)
    }

    /// Swap coefficients in packets of six, starting from both ends
    ///
    /// ⏐❶…❻⏐1…6⏐🂡…🂦⏐①…⑥⏐ → ⏐①…⑥⏐🂡…🂦⏐1…6⏐❶…❻⏐
    fn swap_coefficients(coeff: &mut [f64]) {
        assert_eq!(coeff.len() % 6, 0);
        let middle = coeff.len() / 2;
        let (left, right) = coeff.split_at_mut(middle);
        for (chunk_1, chunk_2) in zip(left.chunks_mut(6), right.chunks_mut(6).rev()) {
            chunk_1.swap_with_slice(chunk_2);
        }
    }

    pub fn eval(&self, data: &Knots, r2: f64) -> f64 {
        // @todo: replace with linear search? usually there are only few knots
        let ndx = match data.r2.binary_search_by(|elem| elem.total_cmp(&r2)) {
            Ok(index) => index,
            Err(index) => index,
        } - 1;
        let ndx6 = ndx * 6;
        assert!(ndx6 + 5 < data.coeff.len());
        let dz = r2 - data.r2[ndx];
        data.coeff[ndx6]
            + dz * (data.coeff[ndx6 + 1]
                + dz * (data.coeff[ndx6 + 2]
                    + dz * (data.coeff[ndx6 + 3]
                        + dz * (data.coeff[ndx6 + 4] + dz * data.coeff[ndx6 + 5]))))
    }

    pub fn eval_derivative(&self, knots: &Knots, r2: f64) -> f64 {
        let ndx = match knots
            .r2
            .binary_search_by(|elem| elem.partial_cmp(&r2).unwrap())
        {
            Ok(index) => index - 1,
            Err(index) => index - 1,
        };
        // let pos = d.r2.binary_search_by(|x| x.total_cmp(&r2)).unwrap_or(0);
        let ndx6 = ndx * 6;
        assert!(ndx6 + 5 < knots.coeff.len());
        let dz = r2 - knots.r2[ndx];
        knots.coeff[ndx6 + 1]
            + dz * (2.0 * knots.coeff[ndx6 + 2]
                + dz * (3.0 * knots.coeff[ndx6 + 3]
                    + dz * (4.0 * knots.coeff[ndx6 + 4] + dz * (5.0 * knots.coeff[ndx6 + 5]))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spline_andrea() {
        let func = |x: f64| 0.5 * x * x.sin() + 2.0;
        let mut spline = Andrea::default();
        spline.spline.set_tolerance(2.0e-6, 1.0e-4); // ftol carries no meaning
        let knots = spline.generate(&func, 0.0, 10.0).unwrap();

        assert_relative_eq!(spline.downscale_factor, 0.9);
        assert_eq!(spline.max_num_downscales, 100);
        assert_eq!(spline.max_num_ctrl_points, 1200);

        assert_eq!(knots.r2.len(), 19);
        assert_eq!(knots.coeff.len(), 108);
        assert_eq!(knots.len(), 19);
        assert_relative_eq!(knots.rmin2, 0.0);
        assert_relative_eq!(knots.rmax2, 10.0);

        assert_relative_eq!(knots.r2[0], 0.0);
        assert_relative_eq!(knots.r2[1], 0.212991, epsilon = 1e-5);
        assert_relative_eq!(knots.r2[2], 0.782554, epsilon = 1e-5);
        assert_relative_eq!(*knots.r2.last().unwrap(), 10.0);

        assert_relative_eq!(knots.coeff[0], 2.0);
        assert_relative_eq!(knots.coeff[1], 0.0);
        assert_relative_eq!(knots.coeff[2], 0.5, epsilon = 1e-6);
        assert_relative_eq!(*knots.coeff.last().unwrap(), -0.0441931, epsilon = 1e-5);

        assert_relative_eq!(func(1.0e-9), spline.eval(&knots, 1e-9));
        assert_relative_eq!(func(5.0), spline.eval(&knots, 5.0), epsilon = 1e-5);
        assert_relative_eq!(func(10.0), spline.eval(&knots, 10.0), epsilon = 1e-5);

        // Check if numerical derivation of *splined* function
        // matches the analytical solution in `eval_der()`.
        let f_prime = |x: f64, dx: f64| {
            (spline.eval(&knots, x + dx) - spline.eval(&knots, x - dx)) / (2.0 * dx)
        };
        let x = 1e-9;
        assert_relative_eq!(
            f_prime(x, 1e-10),
            spline.eval_derivative(&knots, x),
            epsilon = 1e-6
        );
        let x = 1.0;
        assert_relative_eq!(
            f_prime(x, 1e-10),
            spline.eval_derivative(&knots, x),
            epsilon = 1e-6
        );
        let x = 5.0;
        assert_relative_eq!(
            f_prime(x, 1e-10),
            spline.eval_derivative(&knots, x),
            epsilon = 1e-6
        );

        // Check if analytical spline derivative matches
        // derivative of original function
        let f_prime_exact = |x: f64, dx: f64| (func(x + dx) - func(x - dx)) / (2.0 * dx);
        let x = 1e-9;
        assert_relative_eq!(
            f_prime_exact(x, 1e-10),
            spline.eval_derivative(&knots, x),
            epsilon = 1e-5
        );
        let x = 1.0;
        assert_relative_eq!(
            f_prime_exact(x, 1e-10),
            spline.eval_derivative(&knots, x),
            epsilon = 1e-5
        );
        let x = 5.0;
        assert_relative_eq!(
            f_prime_exact(x, 1e-10),
            spline.eval_derivative(&knots, x),
            epsilon = 1e-5
        );
    }
}
