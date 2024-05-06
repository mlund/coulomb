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

/// Field due to electric multipoles.
pub trait MultipoleField: ShortRangeFunction + Cutoff {
    /// Returns the electrostatic field from a point charge.
    ///
    /// The `charge` is a source point charge and `r` is the distance vector from the charge.
    /// The returned value has units of `input charge` / `input length squared`, i.e.
    /// _surface charge density_ also known as _charge areal density_. To get the field,
    /// divide by 4πε₀.
    ///
    /// The field is obtained with the formula:
    ///
    /// 𝐄(𝑧, 𝐫) = 𝑧𝐫 / 𝑟²・{ (1 + 𝜅𝑟)・𝑆(𝑞) - 𝑞𝑆ʹ(𝑞) }・exp{-𝜅𝑟} where 𝑞 = 𝑟 / 𝑟✂︎
    ///
    fn ion_field(&self, charge: f64, r: &Vector3) -> Vector3 {
        let r2 = r.norm_squared();
        if r2 >= self.cutoff_squared() {
            return Vector3::zeros();
        }
        let r1 = r.norm();
        let q = r1 / self.cutoff();
        let srf0 = self.short_range_f0(q);
        let srf1 = self.short_range_f1(q);
        charge * r / (r2 * r1)
            * if let Some(kappa) = self.kappa() {
                ((1.0 + kappa * r1) * srf0 - q * srf1) * (-kappa * r1).exp()
            } else {
                srf0 - q * srf1
            }
    }

    /// Returns the electrostatic field scalar from a point charge.
    ///
    /// The `charge` is a source point charge and `r` is the distance vector from the charge.
    ///
    /// 𝐸(𝑧, r) = 𝑧 / 𝑟²・{ (1 + 𝜅𝑟)・𝑆(𝑞) - 𝑞𝑆ʹ(𝑞) }・exp{-𝜅𝑟} where 𝑞 = 𝑟 / 𝑟✂︎
    ///
    fn ion_field_scalar(&self, charge: f64, r: f64) -> f64 {
        if r >= self.cutoff() {
            return 0.0;
        }
        let q = r / self.cutoff();
        let srf0 = self.short_range_f0(q);
        let srf1 = self.short_range_f1(q);
        charge / r.powi(2)
            * if let Some(kappa) = self.kappa() {
                ((1.0 + kappa * r) * srf0 - q * srf1) * (-kappa * r).exp()
            } else {
                srf0 - q * srf1
            }
    }

    /// Electrostatic field from point dipole.
    ///
    /// Parameters:
    /// - `dipole`: Point dipole (input length x input charge) [UNIT: (input length) x (input charge)]
    /// - `r`: Distance vector from point dipole (input length) [UNIT: input length]
    ///
    /// Returns:
    /// Field from dipole [UNIT: (input charge) / (input length)^2]
    ///
    /// The field from a point dipole is described by the formula:
    /// E(mu, r) = (3 * (mu.dot(r) * r / r²) - mu) / r3 *
    ///             (s(q) - q * s'(q) + q² / 3 * s''(q)) +
    ///             mu / r3 * (s(q) * 𝜅r² - 2 * 𝜅r * q * s'(q) + q² / 3 * s''(q))
    fn dipole_field(&self, dipole: &Vector3, r: &Vector3) -> Vector3 {
        let r2 = r.norm_squared();
        if r2 >= self.cutoff_squared() {
            return Vector3::zeros();
        }
        let r1 = r.norm();
        let r3_inv = (r1 * r2).recip();
        let q = r1 / self.cutoff();
        let srf0 = self.short_range_f0(q);
        let srf1 = self.short_range_f1(q);
        let srf2 = self.short_range_f2(q);
        let mut field = (3.0 * dipole.dot(r) * r / r2 - dipole) * r3_inv;

        if let Some(kappa) = self.kappa() {
            let kr = kappa * r1;
            let kr2 = kr * kr;
            field *= srf0 * (1.0 + kr + kr2 / 3.0) - q * srf1 * (1.0 + 2.0 / 3.0 * kr)
                + q * q / 3.0 * srf2;
            let field_i = dipole * r3_inv * (srf0 * kr2 - 2.0 * kr * q * srf1 + srf2 * q * q) / 3.0;
            (field + field_i) * (-kr).exp()
        } else {
            field *= srf0 - q * srf1 + q * q / 3.0 * srf2;
            let field_i = dipole * r3_inv * q * q * srf2 / 3.0;
            field + field_i
        }
    }
    /// Electrostatic field from point quadrupole.
    ///
    /// Parameters:
    /// - `quad`: Point quadrupole (input length^2 x input charge) [UNIT: (input length)^2 x (input charge)]
    /// - `r`: Distance vector from point quadrupole (input length) [UNIT: input length]
    ///
    /// Returns:
    /// Field from quadrupole [UNIT: (input charge) / (input length)^2]
    fn quadrupole_field(&self, quad: &Matrix3, r: &Vector3) -> Vector3 {
        let r2 = r.norm_squared();
        if r2 >= self.cutoff_squared() {
            return Vector3::zeros();
        }
        let r1 = r.norm();
        let r_hat = r / r1;
        let q = r1 / self.cutoff();
        let q2 = q * q;
        let r4 = r2 * r2;
        let quadrh = quad * r_hat;
        let quad_trh = quad.transpose() * r_hat;

        let s0 = self.short_range_f0(q);
        let s1 = self.short_range_f1(q);
        let s2 = self.short_range_f2(q);
        let s3 = self.short_range_f3(q);

        let f = (1.0 / r2 * r.transpose() * quad * r)[0]; // 1x1 matrix -> f64 by taking first and only element
        let mut field_d = 3.0 * ((5.0 * f - quad.trace()) * r_hat - quadrh - quad_trh) / r4;

        if let Some(kappa) = self.kappa() {
            let kr = kappa * r1;
            let kr2 = kr * kr;
            field_d *=
                s0 * (1.0 + kr + kr2 / 3.0) - q * s1 * (1.0 + 2.0 / 3.0 * kr) + q2 / 3.0 * s2;
            let field_i = f * r_hat / r4
                * (s0 * (1.0 + kr) * kr2 - q * s1 * (3.0 * kr + 2.0) * kr
                    + s2 * (1.0 + 3.0 * kr) * q2
                    - q2 * q * s3);
            0.5 * (field_d + field_i) * (-kr).exp()
        } else {
            field_d *= s0 - q * s1 + q2 / 3.0 * s2;
            let field_i = f * r_hat / r4 * (s2 * q2 - q2 * q * s3);
            0.5 * (field_d + field_i)
        }
    }
}
