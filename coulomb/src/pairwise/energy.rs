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

//! Multipole interaction energies.

use super::{MultipoleField, MultipolePotential};
use crate::{Matrix3, Vector3};

/// Interaction energy between multipoles.
pub trait MultipoleEnergy: MultipolePotential + MultipoleField {
    /// Self-energy of monopoles and dipoles
    ///
    /// The self-energy is described by:
    ///
    /// $$u_{self} = \sum_i c_1 z_j^2 / R_c + c_2 \mu_i^2 / R_c^3 + ...$$
    ///
    /// where $c_1$ and $c_2$ are constants specific for the interaction scheme.
    ///
    fn self_energy(&self, monopoles: &[f64], dipoles: &[f64]) -> f64 {
        let mut sum: f64 = 0.0;
        let prefactor = self.self_energy_prefactors();
        if let Some(c1) = prefactor.monopole {
            sum += c1 * monopoles.iter().map(|z| z * z).sum::<f64>() / self.cutoff();
        }
        if let Some(c2) = prefactor.dipole {
            sum += c2 * dipoles.iter().map(|mu| mu * mu).sum::<f64>() / self.cutoff().powi(3);
        }
        sum
    }
    /// Interaction energy between two point charges
    ///
    /// z1: Point charge, UNIT: [input charge]
    /// z2: Point charge, UNIT: [input charge]
    /// r: Charge-charge separation, UNIT: [input length]
    ///
    /// Returns the interaction energy, UNIT: [(input charge)^2 / (input length)]
    ///
    /// The interaction energy between two charges is described by:
    ///     u(z1, z2, r) = z2 * Phi(z1,r)
    /// where Phi(z1,r) is the potential from ion 1.
    fn ion_ion_energy(&self, charge1: f64, charge2: f64, r: f64) -> f64 {
        charge2 * self.ion_potential(charge1, r)
    }

    /// Interaction energy between a point charge and a point dipole
    ///
    /// - `charge`: Point charge, UNIT: [input charge]
    /// - `dipole`: Dipole moment, UNIT: [(input length) x (input charge)]
    /// - `r`: Distance-vector between dipole and charge, r = r_mu - r_z, UNIT: [input length]
    ///
    /// Returns the interaction energy, UNIT: [(input charge)^2 / (input length)]
    ///
    /// The interaction energy between an ion and a dipole is:
    ///
    /// $$u(z, \mu, r) = z * \Phi(\mu, -r)$$
    ///
    /// where $\Phi(\mu, -r)$ is the potential from the dipole at the location of the ion.
    /// This interaction can also be described by:
    ///
    /// $$u(z, \mu, r) = -\mu.dot(E(z, r))$$
    ///
    /// where $E(z, r)$ is the field from the ion at the location of the dipole.
    fn ion_dipole_energy(&self, charge: f64, dipole: &Vector3, r: &Vector3) -> f64 {
        // Both expressions below give the same answer. Keep for possible optimization in the future.
        // return -dipole_moment.dot(self.ion_field(charge, r)); // field from charge interacting with dipole
        charge * self.dipole_potential(dipole, &(-r)) // potential of dipole interacting with charge
    }

    /// Interaction energy between two point dipoles
    ///
    /// - `dipole1`: Dipole moment of particle 1, UNIT: [(input length) x (input charge)]
    /// - `dipole2`: Dipole moment of particle 2, UNIT: [(input length) x (input charge)]
    /// - `r``: Distance-vector between dipoles, r = r_mu_2 - r_mu_1, UNIT: [input length]
    ///
    /// Returns the interaction energy, UNIT: [(input charge)^2 / (input length)]
    ///
    /// The interaction energy between two dipoles is described by:
    ///     u(mu1, mu2, r) = -mu1.dot(E(mu2, r))
    /// where E(mu2, r) is the field from dipole 2 at the location of dipole 1.
    fn dipole_dipole_energy(&self, dipole1: &Vector3, dipole2: &Vector3, r: &Vector3) -> f64 {
        -dipole1.dot(&self.dipole_field(dipole2, r))
    }

    /// Interaction energy between a point charge and a point quadrupole
    ///
    /// - `charge`: Point charge, UNIT: [input charge]
    /// - `quad`: Quadrupole moment, UNIT: [(input length)^2 x (input charge)]
    /// - `r`: Distance-vector between quadrupole and charge, r = r_Q - r_z, UNIT: [input length]
    ///
    /// Returns the interaction energy, UNIT: [(input charge)^2 / (input length)]
    ///
    /// The interaction energy between an ion and a quadrupole is described by:
    ///     u(z, Q, r) = z * Phi(Q, -r)
    /// where Phi(Q, -r) is the potential from the quadrupole at the location of the ion.
    fn ion_quadrupole_energy(&self, charge: f64, quad: &Matrix3, r: &Vector3) -> f64 {
        charge * self.quadrupole_potential(quad, &(-r)) // potential of quadrupole interacting with charge
    }
}
