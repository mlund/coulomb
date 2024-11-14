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
//! This library provides support for working with electrostatic interactions
//! in _e.g._ molecular systems.
//! This includes:
//!
//! - Background dielectric medium with or without implicit salt.
//! - Handling of electrolyte solutions with salt of arbitrary valency and ionic strength.
//! - Calculation of pairwise interactions between ions and point multipoles using
//!   (truncated) potentials.
//! - Ewald summation
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
//! dielectric constant is found by the [`permittivity::WATER`] model:
//! ~~~
//! # use approx::assert_relative_eq;
//! use coulomb::*;
//! let medium = Medium::neat_water(298.15);
//! assert_relative_eq!(medium.permittivity(), 78.35565171480539);
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
//!
//! The [`pairwise`] module can be used to calculate the interaction energy (and forces, field) between
//! point multipoles in the medium. Here's a simple example for the energy between two point
//! charges:
//! ~~~
//! # use approx::assert_relative_eq;
//! use coulomb::{Medium, TO_CHEMISTRY_UNIT};
//! use coulomb::pairwise::{Plain, MultipoleEnergy};
//!
//! let (z1, z2, r) = (1.0, -1.0, 7.0);      // unit-less charge numbers, separation in angstrom
//! let medium = Medium::neat_water(298.15); // pure water
//! let plain = Plain::without_cutoff();     // generic coulomb interaction scheme
//! let energy = plain.ion_ion_energy(z1, z2, r) * TO_CHEMISTRY_UNIT / medium.permittivity();
//!
//! assert_relative_eq!(energy, -2.533055636224861); // in kJ/mol
//! ~~~

#[cfg(test)]
extern crate approx;

#[cfg(feature = "uom")]
#[macro_use]
extern crate uom;
#[cfg(feature = "uom")]
mod units;

/// A point in 3D space
pub type Vector3 = nalgebra::Vector3<f64>;
/// A 3x3 matrix
pub type Matrix3 = nalgebra::Matrix3<f64>;

pub(crate) const ANGSTROM_PER_METER: f64 = 1e10;
pub(crate) const LITER_PER_ANGSTROM3: f64 = 1e-27;

mod cutoff;
pub use cutoff::Cutoff;
mod math;
mod medium;
pub mod pairwise;
pub mod permittivity;
pub mod reciprocal;
mod salt;
mod spline;
pub use medium::Medium;
pub use salt::Salt;
mod temperature;
pub use temperature::Temperature;
mod ionic_strength;
pub use ionic_strength::IonicStrength;
mod debye_length;
pub use debye_length::{bjerrum_length, debye_length, DebyeLength};

use physical_constants::{
    AVOGADRO_CONSTANT, ELEMENTARY_CHARGE, MOLAR_GAS_CONSTANT, VACUUM_ELECTRIC_PERMITTIVITY,
};
use std::f64::consts::PI;

/// Electrostatic prefactor, e²/4πε₀ × 10⁷ × NA [Å × kJ / mol].
///
/// Use to scale potential, energy, forces, fields from the [`pairwise`] module to units commonly used in chemistry;
/// `kJ`, `mol`, `Å`, and `elementary charge`.
/// Note that this uses a vacuum permittivity and the final result should be divided by the relative dielectric constant for the
/// actual medium.
/// If input length and charges are in units of angstrom and elementary charge:
///
/// - `CHEMISTRY_UNIT` × _energy_ ➔ kJ/mol
/// - `CHEMISTRY_UNIT` × _force_ ➔ kJ/mol/Å
/// - `CHEMISTRY_UNIT` × _potential_ ➔ kJ/mol×e
/// - `CHEMISTRY_UNIT` × _field_ ➔ kJ/mol/Å×e
///
/// # Examples:
/// ```
/// # use approx::assert_relative_eq;
/// use coulomb::TO_CHEMISTRY_UNIT;
/// let (z1, z2, r) = (1.0, -1.0, 7.0); // unit-less charge number, separation in angstrom
/// let rel_permittivity = 80.0;
/// let energy = TO_CHEMISTRY_UNIT / rel_permittivity * z1 * z2 / r;
/// assert_relative_eq!(energy, -2.48099031507825); // in kJ/mol
/// ```
///
pub const TO_CHEMISTRY_UNIT: f64 =
    ELEMENTARY_CHARGE * ELEMENTARY_CHARGE * ANGSTROM_PER_METER * AVOGADRO_CONSTANT * 1e-3
        / (4.0 * PI * VACUUM_ELECTRIC_PERMITTIVITY);

/// Bjerrum length in vacuum at 298.15 K, e²/4πε₀kT (Å).
///
/// Examples:
/// ```
/// use coulomb::BJERRUM_LEN_VACUUM_298K;
/// let relative_permittivity = 80.0;
/// assert_eq!(BJERRUM_LEN_VACUUM_298K / relative_permittivity, 7.0057415269733);
/// ```
pub const BJERRUM_LEN_VACUUM_298K: f64 = TO_CHEMISTRY_UNIT / (MOLAR_GAS_CONSTANT * 1e-3 * 298.15);
