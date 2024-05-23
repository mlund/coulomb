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

use crate::pairwise::{SelfEnergyPrefactors, ShortRangeFunction};
use crate::{math::erf_x, math::erfc_x, Cutoff};
use core::f64::consts::FRAC_2_SQRT_PI;
#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize};

/// Truncated Gaussian Ewald scheme (real-space part).
///
/// From the abstract of <https://doi.org/dsd6>:
///
/// _We present the widespread Ewald summation method in a new light
/// by utilizing a truncated Gaussian screening charge distribution.
/// This choice entails an exact formalism, also as particle mesh Ewald,
/// which in practice is not always the case when using a Gaussian screening function.
/// The presented approach reduces the number of dependent parameters compared to a Gaussian
/// and, for an infinite reciprocal space cutoff, makes the screening charge distribution
/// width truly arbitrary. As such, this arbitrary variable becomes an ideal tool for
/// computational optimization while maintaining accuracy, which is in contrast to when a
/// Gaussian is used._
///
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct EwaldTruncated {
    /// Cutoff radius
    cutoff: f64,
    /// Alpha
    #[cfg_attr(feature = "serde", serde(alias = "α"))]
    alpha: f64,
    /// Reduced alpha = alpha * cutoff
    #[cfg_attr(feature = "serde", serde(skip))]
    eta: f64,
    /// erfc(eta)
    #[cfg_attr(feature = "serde", serde(skip))]
    erfc_eta: f64,
    /// exp(-eta^2)
    #[cfg_attr(feature = "serde", serde(skip))]
    exp_minus_eta2: f64,
    /// f0 = 1 / (1 - erfc(eta) - 2 * eta / sqrt(pi) * exp(-eta^2))
    #[cfg_attr(feature = "serde", serde(skip))]
    f0: f64,
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for EwaldTruncated {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct EwaldTruncatedData {
            cutoff: f64,
            alpha: f64,
        }

        let EwaldTruncatedData { cutoff, alpha } = EwaldTruncatedData::deserialize(deserializer)?;
        Ok(EwaldTruncated::new(cutoff, alpha))
    }
}

impl EwaldTruncated {
    /// Inverse square root of pi, 1/sqrt(pi)
    const FRAC_1_SQRT_PI: f64 = 0.5 * core::f64::consts::FRAC_2_SQRT_PI;

    pub fn new(cutoff: f64, alpha: f64) -> Self {
        let eta = alpha * cutoff;
        let f0 = (1.0 - erfc_x(eta) - eta * FRAC_2_SQRT_PI * (-eta * eta).exp()).recip();
        Self {
            cutoff,
            alpha,
            eta,
            erfc_eta: erfc_x(eta),
            exp_minus_eta2: (-eta * eta).exp(),
            f0,
        }
    }
}

impl Cutoff for EwaldTruncated {
    fn cutoff(&self) -> f64 {
        self.cutoff
    }
}

impl ShortRangeFunction for EwaldTruncated {
    fn url() -> &'static str {
        "https://doi.org/dsd6"
    }

    fn self_energy_prefactors(&self) -> SelfEnergyPrefactors {
        let c1 = -self.eta * Self::FRAC_1_SQRT_PI * (1.0 - self.exp_minus_eta2) * self.f0;
        let c2 = -2.0 * self.eta.powi(3)
            / (3.0
                * (erf_x(self.eta) / Self::FRAC_1_SQRT_PI - 2.0 * self.eta * self.exp_minus_eta2));
        SelfEnergyPrefactors {
            monopole: Some(c1),
            dipole: Some(c2),
        }
    }
    fn kappa(&self) -> Option<f64> {
        None
    }
    fn short_range_f0(&self, q: f64) -> f64 {
        (erfc_x(self.eta * q)
            - self.erfc_eta
            - (1.0 - q) * self.eta * FRAC_2_SQRT_PI * self.exp_minus_eta2)
            * self.f0
    }
    fn short_range_f1(&self, q: f64) -> f64 {
        -self.eta
            * ((-(self.eta * q).powi(2)).exp() - self.exp_minus_eta2)
            * FRAC_2_SQRT_PI
            * self.f0
    }
    fn short_range_f2(&self, q: f64) -> f64 {
        2.0 * self.eta.powi(3) * q * (-(self.eta * q).powi(2)).exp() * FRAC_2_SQRT_PI * self.f0
    }
    fn short_range_f3(&self, q: f64) -> f64 {
        -4.0 * ((self.eta * q).powi(2) - 0.5)
            * self.eta.powi(3)
            * (-(self.eta * q).powi(2)).exp()
            * FRAC_2_SQRT_PI
            * self.f0
    }
}

#[test]
fn test_truncated_ewald() {
    use crate::pairwise::MultipoleEnergy;

    use approx::assert_relative_eq;
    let cutoff = 29.0;
    let alpha = 0.1;
    let eps = 1e-9;
    let pot = EwaldTruncated::new(cutoff, alpha);
    assert_relative_eq!(pot.short_range_f0(0.5), 0.03993019621374575, epsilon = eps);
    assert_relative_eq!(pot.short_range_f1(0.5), -0.39929238172082965, epsilon = eps);
    assert_relative_eq!(pot.short_range_f2(0.5), 3.364180431728417, epsilon = eps);
    assert_relative_eq!(pot.short_range_f3(0.5), -21.56439656737916, epsilon = eps);
    assert_relative_eq!(
        pot.self_energy(&[2.0], &[0.0]),
        -0.22579937362074382,
        epsilon = eps
    );
    assert_relative_eq!(
        pot.self_energy(&[0.0], &[f64::sqrt(2.0)]),
        -0.0007528321650,
        epsilon = eps
    );
}
