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
use crate::debye_length::DebyeLength;
use crate::math::erfc_x;
use crate::pairwise::{SelfEnergyPrefactors, ShortRangeFunction};
#[cfg(test)]
use approx::assert_relative_eq;
#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize};

/// Scheme for real-space Ewald interactions
///
/// Further information, see original article by _P.P. Ewald_, <https://doi.org/fcjts8>.
///
#[derive(Debug, Clone, PartialEq)]
pub struct RealSpaceEwald {
    /// Real space cutoff distance, 𝑟✂︎
    cutoff: f64,
    /// Debye length
    debye_length: Option<f64>,
    /// Reduced alpha, 𝜂 = 𝛼 × 𝑟✂︎ (dimensionless)
    eta: f64,
    /// Reduced inverse screening length, 𝜻 = 𝜿 × 𝑟✂︎ (dimensionless)
    zeta: Option<f64>,
}

#[cfg(feature = "serde")]
impl Serialize for RealSpaceEwald {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("RealSpaceEwald", 3)?;
        s.serialize_field("cutoff", &self.cutoff)?;
        s.serialize_field("alpha", &self.alpha())?;
        s.serialize_field("debye_length", &self.debye_length)?;
        s.end()
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for RealSpaceEwald {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct RealSpaceEwaldData {
            cutoff: f64,
            accuracy: f64,
            debye_length: Option<f64>,
        }

        let RealSpaceEwaldData {
            cutoff,
            accuracy,
            debye_length,
        } = RealSpaceEwaldData::deserialize(deserializer)?;
        Ok(RealSpaceEwald::new(cutoff, accuracy, debye_length))
    }
}

impl core::fmt::Display for RealSpaceEwald {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Real-space Ewald: 𝑟✂ = {:.1}, 𝜂 = {:.1}",
            self.cutoff, self.eta,
        )?;
        if let Some(zeta) = self.zeta {
            write!(f, ", 𝜻 = {:.1}", zeta)?;
        }
        write!(f, " <{}>", Self::url())?;
        Ok(())
    }
}

impl RealSpaceEwald {
    /// Square root of pi
    const SQRT_PI: f64 = 1.7724538509055159;

    /// Construct a new real-space Ewald scheme from cutoff and target accuracy.
    ///
    /// Derives the splitting parameter α via the Kolafa–Perram estimate
    /// α = √(−ln ε) / r_c (see [`ewald_alpha`](crate::reciprocal::ewald_alpha)).
    ///
    /// # Arguments
    /// - `cutoff`: Real-space cutoff distance, r_c
    /// - `accuracy`: Target relative energy accuracy ε (e.g. `1e-5`), must be in (0, 1)
    /// - `debye_length`: Optional Debye screening length (same unit as cutoff)
    pub fn new(cutoff: f64, accuracy: f64, debye_length: Option<f64>) -> Self {
        let alpha = crate::reciprocal::ewald_alpha(cutoff, accuracy);
        Self::from_alpha(cutoff, alpha, debye_length)
    }

    /// Internal constructor from explicit alpha (used by serde and `real_space_scheme`).
    pub(crate) fn from_alpha(cutoff: f64, alpha: f64, debye_length: Option<f64>) -> Self {
        Self {
            cutoff,
            debye_length,
            eta: alpha * cutoff,
            zeta: debye_length.map(|d| cutoff / d),
        }
    }

    /// Returns the damping parameter α.
    pub fn alpha(&self) -> f64 {
        self.eta / self.cutoff
    }
}

impl crate::Cutoff for RealSpaceEwald {
    #[inline]
    fn cutoff(&self) -> f64 {
        self.cutoff
    }
}

impl crate::DebyeLength for RealSpaceEwald {
    #[inline]
    fn kappa(&self) -> Option<f64> {
        self.zeta.map(|z| z / self.cutoff)
    }
    fn set_debye_length(&mut self, debye_length: Option<f64>) -> crate::Result<()> {
        self.zeta = debye_length.map(|d| self.cutoff / d);
        Ok(())
    }
}

impl ShortRangeFunction for RealSpaceEwald {
    fn url() -> &'static str {
        "https://doi.org/fcjts8"
    }

    #[inline]
    fn short_range_f0(&self, q: f64) -> f64 {
        if let Some(zeta) = self.zeta {
            0.5 * (erfc_x(self.eta * q + zeta / (2.0 * self.eta)) * f64::exp(2.0 * zeta * q)
                + erfc_x(self.eta * q - zeta / (2.0 * self.eta)))
        } else {
            erfc_x(self.eta * q)
        }
    }

    fn short_range_f1(&self, q: f64) -> f64 {
        if let Some(zeta) = self.zeta {
            let exp_c = f64::exp(-(self.eta * q - zeta / (2.0 * self.eta)).powi(2));
            let erfc_c = erfc_x(self.eta * q + zeta / (2.0 * self.eta));
            -2.0 * self.eta / Self::SQRT_PI * exp_c + zeta * erfc_c * f64::exp(2.0 * zeta * q)
        } else {
            -2.0 * self.eta / Self::SQRT_PI * f64::exp(-self.eta.powi(2) * q.powi(2))
        }
    }

    fn short_range_f2(&self, q: f64) -> f64 {
        if let Some(zeta) = self.zeta {
            let exp_c = f64::exp(-(self.eta * q - zeta / (2.0 * self.eta)).powi(2));
            let erfc_c = erfc_x(self.eta * q + zeta / (2.0 * self.eta));
            4.0 * self.eta.powi(2) / Self::SQRT_PI * (self.eta * q - zeta / self.eta) * exp_c
                + 2.0 * zeta.powi(2) * erfc_c * f64::exp(2.0 * zeta * q)
        } else {
            4.0 * self.eta.powi(2) / Self::SQRT_PI
                * (self.eta * q)
                * f64::exp(-(self.eta * q).powi(2))
        }
    }

    fn short_range_f3(&self, q: f64) -> f64 {
        if let Some(zeta) = self.zeta {
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
        } else {
            4.0 * self.eta.powi(3) / Self::SQRT_PI
                * (1.0 - 2.0 * (self.eta * q).powi(2))
                * (-(self.eta * q).powi(2)).exp()
        }
    }

    fn self_energy_prefactors(&self) -> SelfEnergyPrefactors {
        let (c1, c2) = if let Some(zeta) = self.zeta {
            let c1 = -self.eta / Self::SQRT_PI
                * (f64::exp(-zeta.powi(2) / 4.0 / self.eta.powi(2))
                    - Self::SQRT_PI * zeta / (2.0 * self.eta) * erfc_x(zeta / (2.0 * self.eta)));
            let c2 = -self.eta.powi(3) / Self::SQRT_PI * 2.0 / 3.0
                * (Self::SQRT_PI * zeta.powi(3) / 4.0 / self.eta.powi(3)
                    * erfc_x(zeta / (2.0 * self.eta))
                    + (1.0 - zeta.powi(2) / 2.0 / self.eta.powi(2))
                        * f64::exp(-zeta.powi(2) / 4.0 / self.eta.powi(2)));
            (c1, c2)
        } else {
            let c1 = -self.eta / Self::SQRT_PI;
            let c2 = -self.eta.powi(3) / Self::SQRT_PI * 2.0 / 3.0;
            (c1, c2)
        };
        SelfEnergyPrefactors {
            monopole: Some(c1),
            dipole: Some(c2),
        }
    }
}

#[test]
fn test_ewald() {
    use crate::pairwise::MultipoleEnergy;

    // Test short-ranged function without salt
    let cutoff = 29.0;
    let alpha = 0.1;
    let pot = RealSpaceEwald::from_alpha(cutoff, alpha, None);
    let eps = 1e-8;

    assert_relative_eq!(
        pot.self_energy(&[2.0], &[0.0]),
        -0.2256758334,
        epsilon = eps
    );
    assert_relative_eq!(
        pot.self_energy(&[0.0], &[f64::sqrt(2.0)]),
        -0.000752257778,
        epsilon = eps
    );

    assert_relative_eq!(pot.alpha(), alpha, epsilon = eps);
    assert_relative_eq!(pot.short_range_f0(0.5), 0.04030484067840161, epsilon = eps);
    assert_relative_eq!(pot.short_range_f1(0.5), -0.39971358519150996, epsilon = eps);
    assert_relative_eq!(pot.short_range_f2(0.5), 3.36159125, epsilon = eps);
    assert_relative_eq!(pot.short_range_f3(0.5), -21.54779992186245, epsilon = eps);

    // Test short-ranged function with a Debye screening length
    let debye_length = 23.0;
    let pot = RealSpaceEwald::from_alpha(cutoff, alpha, Some(debye_length));
    let eps = 1e-7;

    assert_relative_eq!(
        pot.self_energy(&[2.0], &[0.0]),
        -0.14930129209178544,
        epsilon = eps
    );
    assert_relative_eq!(
        pot.self_energy(&[0.0], &[f64::sqrt(2.0)]),
        -0.0006704901976,
        epsilon = eps
    );

    assert_relative_eq!(pot.kappa().unwrap(), 1.0 / 23.0, epsilon = eps);
    assert_relative_eq!(pot.short_range_f0(0.5), 0.07306333588, epsilon = eps);
    assert_relative_eq!(pot.short_range_f1(0.5), -0.6344413331247332, epsilon = eps);
    assert_relative_eq!(pot.short_range_f2(0.5), 4.42313324197739, epsilon = eps);
    assert_relative_eq!(pot.short_range_f3(0.5), -19.859372613319028, epsilon = eps);

    assert_eq!(
        pot.to_string(),
        "Real-space Ewald: 𝑟✂ = 29.0, 𝜂 = 2.9, 𝜻 = 1.3 <https://doi.org/fcjts8>"
    );
}
