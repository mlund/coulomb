// Copyright 2023 Björn Stenqvist and Mikael Lund
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

use super::{
    MultipoleEnergy, MultipoleField, MultipoleForce, MultipolePotential, ShortRangeFunction,
};
use crate::math::erfc_x;
#[cfg(test)]
use approx::assert_relative_eq;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

impl MultipolePotential for RealSpaceEwald {}
impl MultipoleField for RealSpaceEwald {}
impl MultipoleForce for RealSpaceEwald {}
impl MultipoleEnergy for RealSpaceEwald {}

/// Scheme for real-space Ewald interactions
///
/// Further information, see original article by _P.P. Ewald_, <https://doi.org/fcjts8>.
///
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RealSpaceEwald {
    /// Real space cutoff distance, 𝑟✂︎
    cutoff: f64,
    _alpha: f64,
    /// Reduced alpha, 𝜂 = 𝛼 × 𝑟✂︎ (dimensionless)
    eta: f64,
    /// Reduced kappa, 𝜻 = 𝜿 × 𝑟✂︎ (dimensionless)
    zeta: Option<f64>,
}

impl RealSpaceEwald {
    /// Square root of pi
    const SQRT_PI: f64 = 1.7724538509055159;
    /// Construct a new Ewald scheme with given cutoff, alpha, (and debye length).
    ///
    /// The Debye length and cutoff should have the same unit of length.
    pub fn new(cutoff: f64, alpha: f64, debye_length: Option<f64>) -> Self {
        Self {
            cutoff,
            _alpha: alpha,
            eta: alpha * cutoff,
            zeta: debye_length.map(|d| cutoff / d),
        }
    }
    /// Construct a salt-free Ewald scheme with given cutoff and alpha.
    pub fn new_without_salt(cutoff: f64, alpha: f64) -> Self {
        Self::new(cutoff, alpha, None)
    }
}

impl crate::Cutoff for RealSpaceEwald {
    #[inline]
    fn cutoff(&self) -> f64 {
        self.cutoff
    }
}

impl ShortRangeFunction for RealSpaceEwald {
    /// The inverse Debye length if salt is present, otherwise `None`.
    #[inline]
    fn kappa(&self) -> Option<f64> {
        self.zeta.map(|z| z / self.cutoff)
    }
    #[inline]
    fn short_range_f0(&self, q: f64) -> f64 {
        match self.zeta {
            Some(zeta) => {
                0.5 * (erfc_x(self.eta * q + zeta / (2.0 * self.eta)) * f64::exp(2.0 * zeta * q)
                    + erfc_x(self.eta * q - zeta / (2.0 * self.eta)))
            }
            None => erfc_x(self.eta * q),
        }
    }

    fn short_range_f1(&self, q: f64) -> f64 {
        match self.zeta {
            Some(zeta) => {
                let exp_c = f64::exp(-(self.eta * q - zeta / (2.0 * self.eta)).powi(2));
                let erfc_c = erfc_x(self.eta * q + zeta / (2.0 * self.eta));
                -2.0 * self.eta / Self::SQRT_PI * exp_c + zeta * erfc_c * f64::exp(2.0 * zeta * q)
            }
            None => -2.0 * self.eta / Self::SQRT_PI * f64::exp(-self.eta.powi(2) * q.powi(2)),
        }
    }

    fn short_range_f2(&self, q: f64) -> f64 {
        match self.zeta {
            Some(zeta) => {
                let exp_c = f64::exp(-(self.eta * q - zeta / (2.0 * self.eta)).powi(2));
                let erfc_c = erfc_x(self.eta * q + zeta / (2.0 * self.eta));
                4.0 * self.eta.powi(2) / Self::SQRT_PI * (self.eta * q - zeta / self.eta) * exp_c
                    + 2.0 * zeta.powi(2) * erfc_c * f64::exp(2.0 * zeta * q)
            }
            None => {
                4.0 * self.eta.powi(2) / Self::SQRT_PI
                    * (self.eta * q)
                    * f64::exp(-(self.eta * q).powi(2))
            }
        }
    }

    fn short_range_f3(&self, q: f64) -> f64 {
        match self.zeta {
            Some(zeta) => {
                let exp_c = f64::exp(-(self.eta * q - zeta / (2.0 * self.eta)).powi(2));
                let erfc_c = erfc_x(self.eta * q + zeta / (2.0 * self.eta));
                4.0 * self.eta.powi(3) / Self::SQRT_PI
                    * (1.0
                        - 2.0
                            * (self.eta * q - zeta / self.eta)
                            * (self.eta * q - zeta / (2.0 * self.eta))
                        - zeta.powi(2) / self.eta.powi(2))
                    * exp_c
                    + 4.0 * zeta.powi(3) * erfc_c * f64::exp(2.0 * zeta * q)
            }
            None => {
                4.0 * self.eta.powi(3) / Self::SQRT_PI
                    * (1.0 - 2.0 * (self.eta * q).powi(2))
                    * f64::exp(-(self.eta * q).powi(2))
            }
        }
    }
    
    fn self_energy_prefactors(&self) -> super::SelfEnergyPrefactors {
        // todo: unwrap once
        let monopole = Some(
            -self.eta / Self::SQRT_PI
                * (f64::exp(-self.zeta.unwrap_or(0.0).powi(2) / 4.0 / self.eta.powi(2))
                    - Self::SQRT_PI * self.zeta.unwrap_or(0.0) / (2.0 * self.eta)
                        * erfc_x(self.zeta.unwrap_or(0.0) / (2.0 * self.eta))),
        );
        let dipole = Some(
            -self.eta.powi(3) / Self::SQRT_PI * 2.0 / 3.0
                * (Self::SQRT_PI * self.zeta.unwrap_or(0.0).powi(3) / 4.0 / self.eta.powi(3)
                    * erfc_x(self.zeta.unwrap_or(0.0) / (2.0 * self.eta))
                    + (1.0 - self.zeta.unwrap_or(0.0).powi(2) / 2.0 / self.eta.powi(2))
                        * f64::exp(-self.zeta.unwrap_or(0.0).powi(2) / 4.0 / self.eta.powi(2))),
        );
        super::SelfEnergyPrefactors { monopole, dipole }
    }
}

#[test]
fn test_ewald() {
    // Test short-ranged function without salt
    let pot = RealSpaceEwald::new(29.0, 0.1, None);
    let eps = 1e-8;

    assert_relative_eq!(
        pot.self_energy(&vec![2.0], &vec![0.0]),
        -0.2256758334,
        epsilon = eps
    );
    assert_relative_eq!(
        pot.self_energy(&vec![0.0], &vec![f64::sqrt(2.0)]),
        -0.000752257778,
        epsilon = eps
    );

    assert_relative_eq!(pot.short_range_f0(0.5), 0.04030484067840161, epsilon = eps);
    assert_relative_eq!(pot.short_range_f1(0.5), -0.39971358519150996, epsilon = eps);
    assert_relative_eq!(pot.short_range_f2(0.5), 3.36159125, epsilon = eps);
    assert_relative_eq!(pot.short_range_f3(0.5), -21.54779992186245, epsilon = eps);

    // Test short-ranged function with a Debye screening length
    let pot = RealSpaceEwald::new(29.0, 0.1, Some(23.0));
    let eps = 1e-7;

    // CHECK(pot.self_energy({4.0, 0.0}) == Approx(-0.1493013040));
    // CHECK(pot.self_energy({0.0, 2.0}) == Approx(-0.0006704901976));
    assert_relative_eq!(
        pot.self_energy(&vec![2.0], &vec![0.0]),
        -0.14930129209178544,
        epsilon = eps
    );
    assert_relative_eq!(
        pot.self_energy(&vec![0.0], &vec![f64::sqrt(2.0)]),
        -0.0006704901976,
        epsilon = eps
    );

    assert_relative_eq!(pot.kappa().unwrap(), 1.0 / 23.0, epsilon = eps);
    assert_relative_eq!(pot.short_range_f0(0.5), 0.07306333588, epsilon = eps);
    assert_relative_eq!(pot.short_range_f1(0.5), -0.6344413331247332, epsilon = eps);
    assert_relative_eq!(pot.short_range_f2(0.5), 4.42313324197739, epsilon = eps);
    assert_relative_eq!(pot.short_range_f3(0.5), -19.859372613319028, epsilon = eps);
}
