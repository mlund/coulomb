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

use super::MultipoleField;
use crate::{Matrix3, Vector3};

/// Force between multipoles.
pub trait MultipoleForce: MultipoleField {
    /// Force between two point charges.
    ///
    /// Parameters:
    /// - `charge1`: Point charge (input charge) [UNIT: input charge]
    /// - `charge2`: Point charge (input charge) [UNIT: input charge]
    /// - `r`: Distance vector between charges (input length) [UNIT: input length]
    ///
    /// Returns:
    /// Interaction force (input charge^2 / input length^2) [UNIT: (input charge)^2 / (input length)^2]
    ///
    /// The force between two point charges is described by the formula:
    /// F(z1, z2, r) = z2 * E(z1, r)
    ///
    /// where:
    /// - `charge1`: Point charge
    /// - `charge2`: Point charge
    /// - `r`: Distance vector between charges
    /// - `E(zA, r)`: Field from ion A at the location of ion B
    fn ion_ion_force(&self, charge1: f64, charge2: f64, r: &Vector3) -> Vector3 {
        charge2 * self.ion_field(charge1, r)
    }
    /// Interaction force between a point charge and a point dipole.
    ///
    /// Parameters:
    /// - `charge`: Charge (input charge) [UNIT: input charge]
    /// - `dipole`: Dipole moment (input length x input charge) [UNIT: (input length) x (input charge)]
    /// - `r`: Distance vector between dipole and charge (input length) [UNIT: input length]
    ///
    /// Returns:
    /// Interaction force (input charge^2 / input length^2) [UNIT: (input charge)^2 / (input length)^2]
    ///
    /// The force between an ion and a dipole is described by the formula:
    /// F(charge, mu, r) = charge * E(mu, r)
    ///
    /// where:
    /// - `charge`: Charge
    /// - `mu`: Dipole moment
    /// - `r`: Distance vector between dipole and charge
    /// - `E(mu, r)`: Field from the dipole at the location of the ion
    fn ion_dipole_force(&self, charge: f64, dipole: &Vector3, r: &Vector3) -> Vector3 {
        charge * self.dipole_field(dipole, r)
    }

    /// Interaction force between two point dipoles.
    ///
    /// Parameters:
    /// - `mu1`: Dipole moment of particle 1 (input length x input charge) [UNIT: (input length) x (input charge)]
    /// - `mu2`: Dipole moment of particle 2 (input length x input charge) [UNIT: (input length) x (input charge)]
    /// - `r`: Distance vector between dipoles (input length) [UNIT: input length]
    ///
    /// Returns:
    /// Interaction force (input charge^2 / input length^2) [UNIT: (input charge)^2 / (input length)^2]
    ///
    /// The force between two dipoles is described by the formula:
    /// F(mu1, mu2, r) = FD(mu1, mu2, r) * (s(q) - q * s'(q) + (q^2 / 3) * s''(q))
    ///                  + FI(mu1, mu2, r) * (s''(q) - q * s'''(q)) * q^2 * exp(-kr)
    fn dipole_dipole_force(&self, mu1: &Vector3, mu2: &Vector3, r: &Vector3) -> Vector3 {
        let r2 = r.norm_squared();
        if r2 >= self.cutoff_squared() {
            return Vector3::zeros();
        }
        let r1 = r.norm();
        let rh = r / r1;
        let q = r1 / self.cutoff();
        let q2 = q * q;
        let r4 = r2 * r2;
        let mu1_dot_rh = mu1.dot(&rh);
        let mu2_dot_rh = mu2.dot(&rh);
        let srf0 = self.short_range_f0(q);
        let srf1 = self.short_range_f1(q);
        let srf2 = self.short_range_f2(q);
        let srf3 = self.short_range_f3(q);
        let mut force_d = 3.0
            * ((5.0 * mu1_dot_rh * mu2_dot_rh - mu1.dot(mu2)) * rh
                - mu2_dot_rh * mu1
                - mu1_dot_rh * mu2)
            / r4;
        if let Some(kappa) = self.kappa() {
            let kr = kappa * r1;
            force_d *= srf0 * (1.0 + kr + kr * kr / 3.0) - q * srf1 * (1.0 + 2.0 / 3.0 * kr)
                + q2 / 3.0 * srf2;
            let force_i = mu1_dot_rh * mu2_dot_rh * rh / r4
                * (srf0 * (1.0 + kr) * kr * kr - q * srf1 * (3.0 * kr + 2.0) * kr
                    + srf2 * (1.0 + 3.0 * kr) * q2
                    - q2 * q * srf3);
            (force_d + force_i) * (-kr).exp()
        } else {
            force_d *= srf0 - q * srf1 + q * q / 3.0 * srf2;
            let force_i = mu1_dot_rh * mu2_dot_rh * rh / r4 * (srf2 * (1.0) * q2 - q2 * q * srf3);
            force_d + force_i
        }
    }

    /**
     * Interaction force between a point charge and a point quadrupole.
     *
     * Parameters:
     * - `charge`: Point charge (input charge) [UNIT: input charge]
     * - `quad`: Point quadrupole (input length^2 x input charge) [UNIT: (input length)^2 x (input charge)]
     * - `r`: Distance vector between particles (input length) [UNIT: input length]
     *
     * Returns:
     * Interaction force [UNIT: (input charge)^2 / (input length)^2]
     *
     * The force between a point charge and a point quadrupole is described by the formula:
     * F(charge, quad, r) = charge * E(quad, r)
     * where E(quad, r) is the field from the quadrupole at the location of the ion.
     */
    fn ion_quadrupole_force(&self, charge: f64, quad: Matrix3, r: Vector3) -> Vector3 {
        charge * self.quadrupole_field(&quad, &r)
    }
}
