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

//! Reciprocal space in Ewald summation schemes.

use crate::Cutoff;
use core::f64::consts::PI;
use core::iter::{zip, IntoIterator};

/// Relative permittivity at the boundary
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum BoundaryPermittivity {
    /// Tinfoil or metallic boundary conditions, i.e. infinite permittivity (default)
    #[default]
    Tinfoil,
    /// Vacuum boundary conditions, i.e. permittivity of free space = 1
    Vacuum,
    /// Custom permittivity
    Custom(f64),
}

impl From<BoundaryPermittivity> for f64 {
    fn from(bp: BoundaryPermittivity) -> f64 {
        match bp {
            BoundaryPermittivity::Tinfoil => f64::INFINITY,
            BoundaryPermittivity::Vacuum => 1.0,
            BoundaryPermittivity::Custom(eps) => eps,
        }
    }
}

/// Reciprocal space state.
///
/// Further reading about Ewald summation:
///
/// - General, <https://doi.org/10.1063/1.481216>
/// - Update optimization <https://doi.org/10.1063/1.481216>, Eq. 24
/// - Isotropic periodic boundary conditions, <https://doi.org/10/css8>
pub trait ReciprocalState: Cutoff {
    type Vector3;

    /// Reciprocal space vectors, q
    fn k_vectors(&self) -> &'static [Self::Vector3];
    /// Reciprocal space cutoff
    fn recip_cutoff(&self) -> u32;
    /// Relative permittivity if the surrounding medium
    ///
    /// If `None`, tinfoil boundary conditions are assumed, e.g. infinite permittivity.
    fn surface_permittivity(&self) -> BoundaryPermittivity;
    /// Inverse Debye screening length
    fn kappa(&self) -> Option<f64>;
    fn alpha(&self) -> f64;
    fn box_length(&self) -> Vec<f64>;
    /// Volume of the simulation box
    fn volume(&self) -> f64;

    /// Recalculate all k-vectors
    fn recalc_k_vectors(
        &mut self,
        positions: impl IntoIterator<Item = Self::Vector3>,
        charges: impl IntoIterator<Item = f64>,
    );
}

/// Calculate the dipole moment of a system with respect to the geometric center
fn _dipole_moment(
    positions: impl IntoIterator<Item = crate::Vector3>,
    charges: impl IntoIterator<Item = f64>,
) -> crate::Vector3 {
    zip(positions, charges).map(|(p, q)| p * q).sum()
}

pub trait ReciprocalEnergy: ReciprocalState {
    fn reciprocal_energy() -> f64;

    /// Surface energy due to the dipole moment of the system, Eₛ = 2π / (2ε + 1) * μ² / V
    /// @todo Unit?
    fn surface_energy(&self, system_dipole_moment: f64) -> f64 {
        2.0 * PI / (2.0 * f64::from(self.surface_permittivity()) + 1.0) / self.volume()
            * system_dipole_moment.powi(2)
    }
}

pub trait ReciprocalForce: ReciprocalState {}

pub trait ReciprocalField: ReciprocalState {}
