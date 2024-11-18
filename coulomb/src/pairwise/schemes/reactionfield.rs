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

use crate::pairwise::*;
use crate::Cutoff;
use core::fmt::Display;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Reaction-field potential
///
/// The short-range function of the reaction-field potential is given by:
///
/// $$ S(q) = 1 + \frac{\epsilon_{out}-\epsilon_{in}}{2\epsilon_{out}+\epsilon_{in}}q^3 - \frac{3\epsilon_{out}}{2\epsilon_{out}+\epsilon_{in}}q $$
///
/// where
/// $\epsilon_{out}$ is the relative permittivity of the surrounding medium, _i.e._ "outside" the spherical cutoff.
/// $\epsilon_{in}$ is the relative permittivity of the dispersing medium, _i.e._ inside the spherical cutoff.
/// The optional last term shifts the potential to zero at the cut-off radius.
/// See <https://doi.org/dscmwg> for more information.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ReactionField {
    /// Relative permittivity outside the cut-off i.e. the surroundings.
    #[cfg_attr(feature = "serde", serde(alias = "epsrf"))]
    dielec_out: f64,
    /// Relative permittivity inside the cut-off i.e. the dispersing medium.
    #[cfg_attr(feature = "serde", serde(alias = "epsr"))]
    dielec_in: f64,
    /// Shift to zero potential at the cut-off.
    #[cfg_attr(feature = "serde", serde(alias = "shift"))]
    shift_to_zero: bool,
    /// Cut-off radius.
    cutoff: f64,
}

impl Display for ReactionField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Reaction field: εᵢ = {:.1}, εₒ = {:.1}, 𝑟✂ = {:.1}, {} <{}>",
            self.dielec_in,
            self.dielec_out,
            self.cutoff,
            if self.shift_to_zero {
                "shifted"
            } else {
                "unshifted"
            },
            Self::url()
        )?;
        Ok(())
    }
}

impl ReactionField {
    /// Create a new reaction-field potential
    ///
    /// # Arguments
    /// - `cutoff` - Spherical cut-off distance
    /// - `dielec_out` - Relative permittivity outside the cut-off i.e. the surroundings
    /// - `dielec_in` - Relative permittivity inside the cut-off i.e. the dispersing medium
    /// - `shifted` - Shift to zero potential at the cut-off
    ///
    pub const fn new(cutoff: f64, dielec_out: f64, dielec_in: f64, shift_to_zero: bool) -> Self {
        Self {
            dielec_out,
            dielec_in,
            shift_to_zero,
            cutoff,
        }
    }

    /// Create unshifted reaction-field potential
    ///
    /// # Arguments
    /// - `cutoff` - Spherical cut-off distance
    /// - `dielec_out` - Relative permittivity outside the cut-off i.e. the surroundings
    /// - `dielec_in` - Relative permittivity inside the cut-off i.e. the dispersing medium
    ///
    pub const fn new_unshifted(cutoff: f64, dielec_out: f64, dielec_in: f64) -> Self {
        Self::new(cutoff, dielec_out, dielec_in, false)
    }

    /// Create shifted reaction-field potential
    ///
    /// # Arguments
    /// - `cutoff` - Spherical cut-off distance
    /// - `dielec_out` - Relative permittivity outside the cut-off i.e. the surroundings
    /// - `dielec_in` - Relative permittivity inside the cut-off i.e. the dispersing medium
    ///
    pub const fn new_shifted(cutoff: f64, dielec_out: f64, dielec_in: f64) -> Self {
        Self::new(cutoff, dielec_out, dielec_in, true)
    }
}

impl Cutoff for ReactionField {
    fn cutoff(&self) -> f64 {
        self.cutoff
    }
    fn cutoff_squared(&self) -> f64 {
        self.cutoff * self.cutoff
    }
}

impl ShortRangeFunction for ReactionField {
    fn url() -> &'static str {
        "https://doi.org/dscmwg"
    }

    fn kappa(&self) -> Option<f64> {
        None
    }
    fn short_range_f0(&self, q: f64) -> f64 {
        let f = 1.0
            + (self.dielec_out - self.dielec_in) * q.powi(3)
                / (2.0 * self.dielec_out + self.dielec_in);
        match self.shift_to_zero {
            true => f - 3.0 * self.dielec_out * q / (2.0 * self.dielec_out + self.dielec_in),
            false => f,
        }
    }
    fn short_range_f1(&self, q: f64) -> f64 {
        let f = 3.0 * (self.dielec_out - self.dielec_in) * q.powi(2)
            / (2.0 * self.dielec_out + self.dielec_in);
        match self.shift_to_zero {
            true => f - 3.0 * self.dielec_out / (2.0 * self.dielec_out + self.dielec_in),
            false => f,
        }
    }
    fn short_range_f2(&self, q: f64) -> f64 {
        6.0 * (self.dielec_out - self.dielec_in) * q / (2.0 * self.dielec_out + self.dielec_in)
    }
    fn short_range_f3(&self, _q: f64) -> f64 {
        6.0 * (self.dielec_out - self.dielec_in) / (2.0 * self.dielec_out + self.dielec_in)
    }

    fn self_energy_prefactors(&self) -> SelfEnergyPrefactors {
        let monopole = if self.shift_to_zero {
            Some(-3.0 * self.dielec_out / (4.0 * self.dielec_out + 2.0 * self.dielec_in))
        } else {
            None
        };
        let dipole = Some(
            -(2.0 * self.dielec_out - 2.0 * self.dielec_in)
                / (2.0 * (2.0 * self.dielec_out + self.dielec_in)),
        );
        SelfEnergyPrefactors { monopole, dipole }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_reaction_field() {
        let cutoff = 29.0;
        let dielec_in = 1.0;
        let dielec_out = 80.0;

        let pot = ReactionField::new_unshifted(cutoff, dielec_out, dielec_in);

        assert_relative_eq!(pot.self_energy(&[2.0], &[0.0]), 0.0, epsilon = 1e-6);
        assert_relative_eq!(
            pot.self_energy(&[0.0], &[f64::sqrt(2.0)]),
            -0.00004023807698,
            epsilon = 1e-9
        );

        assert_relative_eq!(pot.short_range_f0(0.5), 1.061335404, epsilon = 1e-6);
        assert_relative_eq!(pot.short_range_f1(0.5), 0.3680124224, epsilon = 1e-6);
        assert_relative_eq!(pot.short_range_f2(0.5), 1.472049689, epsilon = 1e-6);
        assert_relative_eq!(pot.short_range_f3(0.5), 2.944099379, epsilon = 1e-6);
        assert_relative_eq!(pot.short_range_f0(1.0), 1.490683230, epsilon = 1e-6);
        assert_eq!(
            pot.to_string(),
            "Reaction field: εᵢ = 1.0, εₒ = 80.0, 𝑟✂ = 29.0, unshifted <https://doi.org/dscmwg>"
        );

        let pot = ReactionField::new_shifted(cutoff, dielec_out, dielec_in);
        assert_relative_eq!(pot.short_range_f0(0.5), 0.3159937888, epsilon = 1e-6);
        assert_relative_eq!(pot.short_range_f1(0.5), -1.122670807, epsilon = 1e-6);
        assert_relative_eq!(pot.short_range_f2(0.5), 1.472049689, epsilon = 1e-6);
        assert_relative_eq!(pot.short_range_f3(0.5), 2.944099379, epsilon = 1e-6);
        assert_relative_eq!(pot.short_range_f0(1.0), 0.0, epsilon = 1e-6);
        assert_eq!(
            pot.to_string(),
            "Reaction field: εᵢ = 1.0, εₒ = 80.0, 𝑟✂ = 29.0, shifted <https://doi.org/dscmwg>"
        );
    }
}
