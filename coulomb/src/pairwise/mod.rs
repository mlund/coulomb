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

//! Pairwise Electrostatic Interactions.
//!
//! This module contains functions for computing the electrostatic potential;
//! fields; forces; and energies from and between electric multipoles.
//! The starting point is a _short-range function_, $S(q)$,
//! of the reduced distance $q = r / r_c$,
//! where $r$ is the distance between the interacting particles and $r_c$ is the cutoff distance.
//! From this, all multipolar interactions can be derived, e.g. the monopole-monopole energy between two
//! point charges, $q_1$ and $q_2$:
//!
//! $$ u(r) \propto \frac{q_1 q_2}{r} \cdot e^{-\kappa r} \cdot S(q)$$
//!
//! where $\kappa$ is the inverse Debye screening length.
//! The generic Coulomb energy is recovered with
//! $S(q) = 1$, $r_c = \infty$, and $\kappa = 0$.
//!
//! ## Examples
//! ~~~
//! # use approx::assert_relative_eq;
//! use coulomb::pairwise::{Plain, MultipolePotential};
//! let (cutoff, debye_length) = (12.0, None);
//! let plain = Plain::new(cutoff, debye_length);
//!
//! let (charge, distance) = (1.0, 9.0);
//! assert_relative_eq!(plain.ion_potential(charge, distance), charge / distance);
//! ~~~

mod energy;
mod field;
mod force;
mod potential;
mod schemes;
pub use schemes::{
    ewald::*, ewald_truncated::EwaldTruncated, plain::Plain, poisson::*,
    reactionfield::ReactionField,
};
pub use {
    energy::MultipoleEnergy, field::MultipoleField, force::MultipoleForce,
    potential::MultipolePotential,
};
#[cfg(feature = "uom")]
mod uom;
#[cfg(feature = "uom")]
pub use {uom::MultipoleEnergySI, uom::MultipoleFieldSI, uom::MultipolePotentialSI};

impl<T: ShortRangeFunction + crate::Cutoff> MultipolePotential for T {}
impl<T: ShortRangeFunction + crate::Cutoff> MultipoleField for T {}
impl<T: MultipoleField> MultipoleForce for T {}
impl<T: MultipolePotential + MultipoleField> MultipoleEnergy for T {}

/// Short-range function for electrostatic interaction schemes.
///
/// The short-range function, $S(q)$, is a function of the reduced distance $q = r/r_c$,
/// where $r$ is the distance between the interacting particles and $r_c$
/// is a spherical cutoff distance.
/// All _schemes_ implement this trait and is a requirement for the
/// [`MultipolePotential`];
/// [`MultipoleField`];
/// [`MultipoleForce`]; and
/// [`MultipoleEnergy`] traits.
/// In connection with Ewald summation scemes, the short-range function is also known as the
/// _splitting function_.
/// There it is used to split the electrostatic interaction into a short-range part and
/// a long-range part.
pub trait ShortRangeFunction {
    /// URL to the original article describing the short-range function.
    fn url() -> &'static str
    where
        Self: Sized;

    /// Inverse Debye screening length.
    ///
    /// The default implementation returns `None`.
    fn kappa(&self) -> Option<f64> {
        None
    }

    /// Short-range function, 𝑆(𝑞)
    fn short_range_f0(&self, q: f64) -> f64;

    /// First derivative of the short-range function, 𝑑𝑆(𝑞)/𝑑𝑞.
    ///
    /// The default implementation uses a numerical central difference using
    /// `short_range_f0`. For better performance, this should be
    /// overridden with an analytical expression.
    fn short_range_f1(&self, q: f64) -> f64 {
        const EPS: f64 = 1e-6;
        if q <= EPS {
            // avoid q < 0
            (self.short_range_f0(EPS) - self.short_range_f0(0.0)) / EPS
        } else if q >= 1.0 - EPS {
            // avoid q > 1
            (self.short_range_f0(1.0) - self.short_range_f0(1.0 - EPS)) / EPS
        } else {
            (self.short_range_f0(q + EPS) - self.short_range_f0(q - EPS)) / (2.0 * EPS)
        }
    }

    /// Second derivative of the short-range function, 𝑑²𝑆(𝑞)/𝑑𝑞².
    ///
    /// The default implementation uses a numerical central difference of
    /// `short_range_f1`. For better performance, this should be
    /// overridden with an analytical expression.
    fn short_range_f2(&self, q: f64) -> f64 {
        const EPS: f64 = 1e-6;
        (self.short_range_f1(q + EPS) - self.short_range_f1(q - EPS)) / (2.0 * EPS)
    }

    /// Third derivative of the short-range function, 𝑑³𝑆(𝑞)/𝑑𝑞³.
    ///
    /// The default implementation uses a numerical central difference of
    /// `short_range_f2`. For better performance, this should be
    /// overridden with an analytical expression.
    fn short_range_f3(&self, q: f64) -> f64 {
        const EPS: f64 = 1e-6;
        (self.short_range_f2(q + EPS) - self.short_range_f2(q - EPS)) / (2.0 * EPS)
    }

    /// Prefactors for the self-energy of monopoles and dipoles.
    ///
    /// If a prefactor is `None` the self-energy is not calculated. Self-energies
    /// are normally important only when inserting or deleting particles
    /// in a system.
    /// One example is in simulations of the Grand Canonical ensemble.
    /// The default implementation returns a `SelfEnergyPrefactors` with
    /// all prefactors set to `None`.
    ///
    fn self_energy_prefactors(&self) -> SelfEnergyPrefactors {
        SelfEnergyPrefactors::default()
    }
}

/// Prefactors for calculating the self-energy of monopoles and dipoles
///
/// Some short-range functions warrent a self-energy on multipoles. This
/// is important for systems where the number of particles fluctuates, e.g.
/// in the Grand Canonical ensemble. By default the self-energy is not calculated
/// unless prefactors are set.
#[derive(Debug, Clone, Copy, Default)]
pub struct SelfEnergyPrefactors {
    /// Prefactor for the self-energy of monopoles, _c1_.
    monopole: Option<f64>,
    /// Prefactor for the self-energy of dipoles, _c2_.
    dipole: Option<f64>,
}

/// Test electric constant
#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    #[test]
    fn test_electric_constant() {
        // check that we can create a trait object
        let _trait_object =
            &crate::pairwise::Plain::default() as &dyn crate::pairwise::ShortRangeFunction;

        let bjerrum_length = 7.1; // angstrom
        let rel_dielectric_const = 80.0;
        assert_relative_eq!(
            crate::TO_CHEMISTRY_UNIT / rel_dielectric_const / bjerrum_length,
            2.4460467895137676 // In kJ/mol, roughly 1 KT at room temperature
        );
    }
}
