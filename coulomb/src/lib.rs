// Copyright 2023 Mikael Lund
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

//! # Electrostatic Interactions and Electrolyte Solutions
//!
//! This library provides supoprt for working with electrostatic interactions
//! in _e.g._ molecular systems.
//! This includes:
//!
//! - Setting up a background dielectric medium.
//! - Handling of electrolyte solutions with salt of arbitrary valency and ionic strength.
//! - Calculation of pairwise interactions between ions and point multipoles using
//!   (truncated) potentials.
//!
//! ## Interactions between Multipoles
//!
//! Please see the [`pairwise`] module.
//!
//! ## Electrolyte Solutions
//!
//! This provides support for calculating properties of electrolyte solutions
//! such as the
//! [Debye length](https://en.wikipedia.org/wiki/Debye_length),
//! [ionic strength](https://en.wikipedia.org/wiki/Ionic_strength), and
//! [Bjerrum length](https://en.wikipedia.org/wiki/Bjerrum_length).
//! It also has a module for empirical models of relative permittivity as a function
//! of temperature.
//!
//! ### Examples
//!
//! The is a [`Medium`] of neat water at 298.15 K where the temperature-dependent
//! dielectric constant is found by the [`EmpiricalPermittivity::WATER`] model:
//! ~~~
//! # use approx::assert_relative_eq;
//! use coulomb::*;
//! let medium = Medium::neat_water(298.15);
//! assert_relative_eq!(medium.permittivity().unwrap(), 78.35565171480539);
//! assert!(medium.ionic_strength().is_none());
//! assert!(medium.debye_length().is_none());
//! ~~~
//!
//! We can also add [`Salt`] of arbitrary valency and concentration which
//! leads to a non-zero ionic strength and Debye length,
//! ~~~
//! # use approx::assert_relative_eq;
//! # use coulomb::{Medium, Salt, DebyeLength, IonicStrength};
//! let medium = Medium::salt_water(298.15, Salt::CalciumChloride, 0.1);
//! assert_relative_eq!(medium.ionic_strength().unwrap(), 0.3);
//! assert_relative_eq!(medium.debye_length().unwrap(), 5.548902662386284);
//! ~~~

#[cfg(test)]
extern crate approx;

/// A point in 3D space
pub type Vector3 = nalgebra::Vector3<f64>;
/// A 3x3 matrix
pub type Matrix3 = nalgebra::Matrix3<f64>;

mod math;
pub mod pairwise;
mod spline;

use anyhow::Result;
use physical_constants::{
    AVOGADRO_CONSTANT, BOLTZMANN_CONSTANT, ELEMENTARY_CHARGE, MOLAR_GAS_CONSTANT,
    VACUUM_ELECTRIC_PERMITTIVITY,
};
use std::f64::consts::PI;

mod permittivity;
pub use permittivity::{EmpiricalPermittivity, RelativePermittivity};
mod salt;
pub use salt::Salt;
mod medium;
pub use medium::Medium;

/// Trait for objects with a temperature
pub trait Temperature {
    /// Get the temperature in Kelvin
    fn temperature(&self) -> f64;
    /// Set the temperature in Kelvin.
    ///
    /// The default implementation returns an error.
    fn set_temperature(&mut self, _temperature: f64) -> Result<()> {
        Err(anyhow::anyhow!(
            "Setting the temperature is not implemented"
        ))
    }
}

/// Trait for objects that has an ionic strength
pub trait IonicStrength {
    /// Get the ionic strength in mol/l
    ///
    /// The default implementation returns `None`.
    fn ionic_strength(&self) -> Option<f64> {
        None
    }
    /// Try to set the ionic strength in mol/l
    ///
    /// The default implementation returns an error.
    fn set_ionic_strength(&mut self, _ionic_strength: f64) -> Result<()> {
        Err(anyhow::anyhow!(
            "Setting the ionic strength is not implemented"
        ))
    }
}

/// Trait for objects where a Debye length can be calculated
pub trait DebyeLength: IonicStrength + RelativePermittivity + Temperature {
    /// # Debye length in angstrom or `None` if the ionic strength is zero.
    ///
    /// May perform expensive operations so avoid use in speed critical code,
    /// such as inside tight interaction loops.
    fn debye_length(&self) -> Option<f64> {
        let temperature = self.temperature();
        let permittivity = self.permittivity(temperature).unwrap();
        self.ionic_strength()
            .map(|i| debye_length(temperature, permittivity, i))
    }
    /// Inverse Debye length in inverse angstrom or `None` if the ionic strength is zero.
    ///
    /// May perform expensive operations so avoid use in speed critical code,
    /// such as inside tight interaction loops.
    fn kappa(&self) -> Option<f64> {
        self.debye_length().map(f64::recip)
    }
}

/// Calculates the Bjerrum length, lB = e²/4πεkT commonly used in electrostatics (ångström).
///
/// More information at <https://en.wikipedia.org/wiki/Bjerrum_length>.
///
/// # Examples
/// ~~~
/// use coulomb::bjerrum_length;
/// let lB = bjerrum_length(293.0, 80.0); // angstroms
/// assert_eq!(lB, 7.1288799871283);
/// ~~~
pub fn bjerrum_length(kelvin: f64, relative_dielectric_const: f64) -> f64 {
    const ANGSTROM_PER_METER: f64 = 1e10;
    ELEMENTARY_CHARGE.powi(2) * ANGSTROM_PER_METER
        / (4.0
            * PI
            * relative_dielectric_const
            * VACUUM_ELECTRIC_PERMITTIVITY
            * BOLTZMANN_CONSTANT
            * kelvin)
}

/// Calculates the Debye length in angstrom, λD = sqrt(8πlB·I·N·V)⁻¹, where I is the ionic strength in molar units (mol/l).
///
/// # Examples
/// ~~~
/// use coulomb::debye_length;
/// let molarity = 0.03;                              // mol/l
/// let lambda = debye_length(293.0, 80.0, molarity); // angstroms
/// assert_eq!(lambda, 17.576538097378368);
/// ~~~
pub fn debye_length(kelvin: f64, relative_dielectric_const: f64, ionic_strength: f64) -> f64 {
    const LITER_PER_ANGSTROM3: f64 = 1e-27;
    (8.0 * PI
        * bjerrum_length(kelvin, relative_dielectric_const)
        * ionic_strength
        * AVOGADRO_CONSTANT
        * LITER_PER_ANGSTROM3)
        .sqrt()
        .recip()
}

/// Electrostatic prefactor, e²/4πε₀ × 10⁷ × NA (Å × kJ / mol).
///
/// Can be used to calculate e.g. the interaction energy bewteen two
/// point charges in kJ/mol:
///
/// Examples:
/// ```
/// use coulomb::ELECTRIC_PREFACTOR;
/// let z1 = 1.0;                    // unit-less charge number
/// let z2 = -1.0;                   // unit-less charge number
/// let r = 7.0;                     // separation in angstrom
/// let rel_dielectric_const = 80.0; // relative dielectric constant
/// let energy = ELECTRIC_PREFACTOR * z1 * z2 / (rel_dielectric_const * r);
/// assert_eq!(energy, -2.48099031507825); // in kJ/mol
///
pub const ELECTRIC_PREFACTOR: f64 =
    ELEMENTARY_CHARGE * ELEMENTARY_CHARGE * 1.0e10 * AVOGADRO_CONSTANT * 1e-3
        / (4.0 * PI * VACUUM_ELECTRIC_PERMITTIVITY);

/// Bjerrum length in vacuum at 298.15 K, e²/4πε₀kT (Å).
///
/// Examples:
/// ```
/// use coulomb::BJERRUM_LEN_VACUUM_298K;
/// let relative_dielectric_const = 80.0;
/// assert_eq!(BJERRUM_LEN_VACUUM_298K / relative_dielectric_const, 7.0057415269733);
/// ```
pub const BJERRUM_LEN_VACUUM_298K: f64 = ELECTRIC_PREFACTOR / (MOLAR_GAS_CONSTANT * 1e-3 * 298.15);

/// Defines information about a concept, like a short name, citation, url etc.
pub trait Info {
    /// Returns a short name for the concept. Use `_` for spaces and avoid weird characters.
    /// This is typically used as keywords in user input and output, e.g. in JSON files.
    fn short_name(&self) -> Option<&'static str> {
        None
    }
    /// Returns a long name for the concept. Spaces are allowed.
    fn long_name(&self) -> Option<&'static str> {
        None
    }

    /// Returns a citation string which should be a
    /// 1. Digital Object Identifier (DOI) in the format `doi:...` (preferred)
    /// 2. URL in the format `https://...`
    fn citation(&self) -> Option<&'static str> {
        None
    }
    /// Tries to extract a URL from the citation string
    fn url(&self) -> Option<String> {
        match self.citation() {
            Some(c) => match c.strip_prefix("doi:") {
                Some(doi) => Some(format!("https://doi.org/{}", doi)),
                _ if c.starts_with("https://") || c.starts_with("http://") => Some(c.to_string()),
                _ => None,
            },
            None => None,
        }
    }
}

/// Defines a cutoff distance
pub trait Cutoff {
    /// Squared cutoff distance
    fn cutoff_squared(&self) -> f64 {
        self.cutoff().powi(2)
    }

    /// Cutoff distance
    fn cutoff(&self) -> f64;
}
