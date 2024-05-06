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
use super::ShortRangeFunction;
use crate::{Cutoff, Matrix3, Vector3};

/// Electric potential from point multipoles.
///
/// The units of the returned potentials is [ ( input charge ) / ( input length ) ]
pub trait MultipolePotential: ShortRangeFunction + Cutoff {
    #[inline]
    /// Electrostatic potential from a point charge.
    fn ion_potential(&self, charge: f64, distance: f64) -> f64 {
        if distance >= self.cutoff() {
            return 0.0;
        }
        let q = distance / self.cutoff();
        charge / distance
            * self.short_range_f0(q)
            * self.kappa().map_or(1.0, |kappa| (-kappa * distance).exp())
    }

    /// Electrostatic potential from a point dipole.
    ///
    /// Parameters:
    /// - `dipole`: Dipole moment, UNIT: [ ( input length ) x ( input charge ) ]
    /// - `r`: Distance vector from the dipole, UNIT: [ input length ]
    ///
    /// Returns:
    /// - Dipole potential, UNIT: [ ( input charge ) / ( input length ) ]
    ///
    /// The potential from a point dipole is described by the formula:
    /// Phi(mu, r) = (mu dot r) / (|r|^2) * [s(q) - q * s'(q)] * exp(-kr)
    fn dipole_potential(&self, dipole: &Vector3, r: &Vector3) -> f64 {
        let r2 = r.norm_squared();
        if r2 >= self.cutoff_squared() {
            return 0.0;
        }
        let r1 = r2.sqrt(); // |r|
        let q = r1 / self.cutoff();
        let srf0 = self.short_range_f0(q);
        let srf1 = self.short_range_f1(q);
        dipole.dot(r) / (r2 * r1)
            * if let Some(kappa) = self.kappa() {
                (srf0 * (1.0 + kappa * r1) - q * srf1) * (-kappa * r1).exp()
            } else {
                srf0 - q * srf1
            }
    }

    /// Electrostatic potential from a point quadrupole.
    fn quadrupole_potential(&self, quad: &Matrix3, r: &Vector3) -> f64 {
        let r2 = r.norm_squared();
        if r2 >= self.cutoff_squared() {
            return 0.0;
        }
        let r1 = r.norm();
        let q = r1 / self.cutoff();
        let srf0 = self.short_range_f0(q);
        let srf1 = self.short_range_f1(q);
        let srf2 = self.short_range_f2(q);
        let trace = quad.trace();
        let f = 3.0 / r2 * (r.transpose() * quad * r)[0] - trace;

        0.5 / (r1 * r2)
            * if let Some(kappa) = self.kappa() {
                let kr = kappa * r1;
                let kr2 = kr * kr;
                let a = srf0 * (1.0 + kr + kr2 / 3.0) - q * srf1 * (1.0 + 2.0 / 3.0 * kr)
                    + q * q / 3.0 * srf2;
                let b = (srf0 * kr2 - 2.0 * kr * q * srf1 + srf2 * q * q) / 3.0;
                (f * a + trace * b) * (-kr).exp()
            } else {
                let a = srf0 - q * srf1 + q * q / 3.0 * srf2;
                let b = (srf2 * q * q) / 3.0;
                f * a + trace * b
            }
    }
}
