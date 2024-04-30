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

//! # Pairwise Electrostatic Interactions
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
//! use coulomb::pairwise::*;
//! let (cutoff, debye_length) = (12.0, None);
//! let plain = Plain::new(cutoff, debye_length);
//!
//! let (charge, distance) = (1.0, 9.0);
//! assert_relative_eq!(plain.ion_potential(charge, distance), charge / distance);
//! ~~~

mod ewald;
pub use ewald::*;
mod ewald_truncated;
pub use ewald_truncated::EwaldTruncated;

mod plain;
pub use plain::Plain;
mod poisson;
use crate::{Matrix3, Vector3};
pub use poisson::*;

mod reactionfield;
pub use reactionfield::ReactionField;

/// Short-range function for electrostatic interaction schemes
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
    const URL: &'static str;

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

/// Electric potential from point multipoles
///
/// The units of the returned potentials is [ ( input charge ) / ( input length ) ]
pub trait MultipolePotential: ShortRangeFunction + crate::Cutoff {
    #[inline]
    /// Electrostatic potential from a point charge.
    fn ion_potential(&self, charge: f64, distance: f64) -> f64 {
        if distance >= self.cutoff() {
            return 0.0;
        }
        let q = distance / self.cutoff();
        charge / distance
            * self.short_range_f0(q)
            * self.kappa().map_or(1.0, |kappa| (-kappa * distance).exp())
    }
    /// Electrostatic potential from a point dipole.
    ///
    /// Parameters:
    /// - `dipole`: Dipole moment, UNIT: [ ( input length ) x ( input charge ) ]
    /// - `r`: Distance vector from the dipole, UNIT: [ input length ]
    ///
    /// Returns:
    /// - Dipole potential, UNIT: [ ( input charge ) / ( input length ) ]
    ///
    /// The potential from a point dipole is described by the formula:
    /// Phi(mu, r) = (mu dot r) / (|r|^2) * [s(q) - q * s'(q)] * exp(-kr)
    fn dipole_potential(&self, dipole: &Vector3, r: &Vector3) -> f64 {
        let r2 = r.norm_squared();
        if r2 >= self.cutoff_squared() {
            return 0.0;
        }
        let r1 = r2.sqrt(); // |r|
        let q = r1 / self.cutoff();
        let srf0 = self.short_range_f0(q);
        let srf1 = self.short_range_f1(q);
        dipole.dot(r) / (r2 * r1)
            * if let Some(kappa) = self.kappa() {
                (srf0 * (1.0 + kappa * r1) - q * srf1) * (-kappa * r1).exp()
            } else {
                srf0 - q * srf1
            }
    }

    /// Electrostatic potential from a point quadrupole.
    fn quadrupole_potential(&self, quad: &Matrix3, r: &Vector3) -> f64 {
        let r2 = r.norm_squared();
        if r2 >= self.cutoff_squared() {
            return 0.0;
        }
        let r1 = r.norm();
        let q = r1 / self.cutoff();
        let kr = self.kappa().unwrap_or(0.0) * r1;
        let srf0 = self.short_range_f0(q);
        let srf1 = self.short_range_f1(q);
        let srf2 = self.short_range_f2(q);
        let kr2 = kr * kr;
        let a =
            srf0 * (1.0 + kr + kr2 / 3.0) - q * srf1 * (1.0 + 2.0 / 3.0 * kr) + q * q / 3.0 * srf2;
        let b = (srf0 * kr2 - 2.0 * kr * q * srf1 + srf2 * q * q) / 3.0;
        0.5 * ((3.0 / r2 * (r.transpose() * quad * r)[0] - quad.trace()) * a + quad.trace() * b)
            / (r1 * r2)
            * (-kr).exp()

        // C++:
        // const double r1 = std::sqrt(r2);
        // const double q = r1 * inverse_cutoff;
        // const double q2 = q * q;
        // const double kr = inverse_debye_length * r1;
        // const double kr2 = kr * kr;
        // const double srf0 = static_cast<const T *>(this)->short_range_function(q);
        // const double srf1 = static_cast<const T *>(this)->short_range_function_derivative(q);
        // const double srf2 = static_cast<const T *>(this)->short_range_function_second_derivative(q);

        // const double a = (srf0 * (1.0 + kr + kr2 / 3.0) - q * srf1 * (1.0 + 2.0 / 3.0 * kr) + q2 / 3.0 * srf2);
        // const double b = (srf0 * kr2 - 2.0 * kr * q * srf1 + srf2 * q2) / 3.0;
        // return 0.5 * ((3.0 / r2 * r.transpose() * quad * r - quad.trace()) * a + quad.trace() * b) / r2 / r1 *
        //        std::exp(-inverse_debye_length * r1);
    }
}

/// # Field due to electric multipoles
pub trait MultipoleField: ShortRangeFunction + crate::Cutoff {
    /// Electrostatic field from point charge.
    ///
    /// Parameters:
    /// - `charge`: Point charge (input charge) [UNIT: input charge]
    /// - `r`: Distance vector from point charge (input length) [UNIT: input length]
    ///
    /// Returns:
    /// Field from charge [UNIT: (input charge) / (input length)^2]
    ///
    /// The field from a charge is described by the formula:
    /// E(z, r) = z * r / |r|^2 * (s(q) - q * s'(q))
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
        let r_norm = r.norm();
        let r_hat = r / r_norm;
        let q = r_norm / self.cutoff();
        let q2 = q * q;
        let kr = self.kappa().unwrap_or(0.0) * r_norm;
        let kr2 = kr * kr;
        let r4 = r2 * r2;
        let quadrh = quad * r_hat;
        let quad_trh = quad.transpose() * r_hat;

        let quadfactor = (1.0 / r2 * r.transpose() * quad * r)[0]; // 1x1 matrix -> f64 by taking first and only element
        let s0 = self.short_range_f0(q);
        let s1 = self.short_range_f1(q);
        let s2 = self.short_range_f2(q);
        let s3 = self.short_range_f3(q);

        let field_d = 3.0 * ((5.0 * quadfactor - quad.trace()) * r_hat - quadrh - quad_trh) / r4
            * (s0 * (1.0 + kr + kr2 / 3.0) - q * s1 * (1.0 + 2.0 / 3.0 * kr) + q2 / 3.0 * s2);
        let field_i = quadfactor * r_hat / r4
            * (s0 * (1.0 + kr) * kr2 - q * s1 * (3.0 * kr + 2.0) * kr + s2 * (1.0 + 3.0 * kr) * q2
                - q2 * q * s3);
        return 0.5 * (field_d + field_i) * (-kr).exp();

        // let field_d = 3.0 * ((5.0 * quadfactor - quad.trace()) * r_hat - quadrh - quad_trh) / r4
        //     * (s0 * (1.0 + kr + kr2 / 3.0) - q * s1 * (1.0 + 2.0 / 3.0 * kr) + q2 / 3.0 * s2);
        // let field_i = quadfactor * r_hat / r4
        //     * (s0 * (1.0 + kr) * kr2 - q * s1 * (3.0 * kr + 2.0) * kr + s2 * (1.0 + 3.0 * kr) * q2
        //         - q2 * q * s3);
        // 0.5 * (field_d + field_i) * (-kr).exp()
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

/// # Interaction energy between multipoles
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
    /// r: Distance-vector between dipoles, r = r_mu_2 - r_mu_1, UNIT: [input length]
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

/// # Force between multipoles
pub trait MultipoleForce: MultipoleField {
    /// Force between two point charges.
    ///
    /// Parameters:
    /// - `charge1`: Point charge (input charge) [UNIT: input charge]
    /// - `charge2`: Point charge (input charge) [UNIT: input charge]
    /// - `r`: Distance vector between charges (input length) [UNIT: input length]
    ///
    /// Returns:
    /// Interaction force (input charge^2 / input length^2) [UNIT: (input charge)^2 / (input length)^2]
    ///
    /// The force between two point charges is described by the formula:
    /// F(z1, z2, r) = z2 * E(z1, r)
    ///
    /// where:
    /// - `charge1`: Point charge
    /// - `charge2`: Point charge
    /// - `r`: Distance vector between charges
    /// - `E(zA, r)`: Field from ion A at the location of ion B
    fn ion_ion_force(&self, charge1: f64, charge2: f64, r: &Vector3) -> Vector3 {
        charge2 * self.ion_field(charge1, r)
    }
    /// Interaction force between a point charge and a point dipole.
    ///
    /// Parameters:
    /// - `charge`: Charge (input charge) [UNIT: input charge]
    /// - `dipole`: Dipole moment (input length x input charge) [UNIT: (input length) x (input charge)]
    /// - `r`: Distance vector between dipole and charge (input length) [UNIT: input length]
    ///
    /// Returns:
    /// Interaction force (input charge^2 / input length^2) [UNIT: (input charge)^2 / (input length)^2]
    ///
    /// The force between an ion and a dipole is described by the formula:
    /// F(charge, mu, r) = charge * E(mu, r)
    ///
    /// where:
    /// - `charge`: Charge
    /// - `mu`: Dipole moment
    /// - `r`: Distance vector between dipole and charge
    /// - `E(mu, r)`: Field from the dipole at the location of the ion
    fn ion_dipole_force(&self, charge: f64, dipole: &Vector3, r: &Vector3) -> Vector3 {
        charge * self.dipole_field(dipole, r)
    }

    /// Interaction force between two point dipoles.
    ///
    /// Parameters:
    /// - `mu1`: Dipole moment of particle 1 (input length x input charge) [UNIT: (input length) x (input charge)]
    /// - `mu2`: Dipole moment of particle 2 (input length x input charge) [UNIT: (input length) x (input charge)]
    /// - `r`: Distance vector between dipoles (input length) [UNIT: input length]
    ///
    /// Returns:
    /// Interaction force (input charge^2 / input length^2) [UNIT: (input charge)^2 / (input length)^2]
    ///
    /// The force between two dipoles is described by the formula:
    /// F(mu1, mu2, r) = FD(mu1, mu2, r) * (s(q) - q * s'(q) + (q^2 / 3) * s''(q))
    ///                  + FI(mu1, mu2, r) * (s''(q) - q * s'''(q)) * q^2 * exp(-kr)
    fn dipole_dipole_force(&self, mu1: &Vector3, mu2: &Vector3, r: &Vector3) -> Vector3 {
        let r2 = r.norm_squared();
        if r2 >= self.cutoff_squared() {
            return Vector3::zeros();
        }
        let r1 = r.norm();
        let rh = r / r1;
        let q = r1 / self.cutoff();
        let q2 = q * q;
        let r4 = r2 * r2;
        let mu1_dot_rh = mu1.dot(&rh);
        let mu2_dot_rh = mu2.dot(&rh);
        let srf0 = self.short_range_f0(q);
        let srf1 = self.short_range_f1(q);
        let srf2 = self.short_range_f2(q);
        let srf3 = self.short_range_f3(q);
        let mut force_d = 3.0
            * ((5.0 * mu1_dot_rh * mu2_dot_rh - mu1.dot(mu2)) * rh
                - mu2_dot_rh * mu1
                - mu1_dot_rh * mu2)
            / r4;
        if let Some(kappa) = self.kappa() {
            let kr = kappa * r1;
            force_d *= srf0 * (1.0 + kr + kr * kr / 3.0) - q * srf1 * (1.0 + 2.0 / 3.0 * kr)
                + q2 / 3.0 * srf2;
            let force_i = mu1_dot_rh * mu2_dot_rh * rh / r4
                * (srf0 * (1.0 + kr) * kr * kr - q * srf1 * (3.0 * kr + 2.0) * kr
                    + srf2 * (1.0 + 3.0 * kr) * q2
                    - q2 * q * srf3);
            (force_d + force_i) * (-kr).exp()
        } else {
            force_d *= srf0 - q * srf1 + q * q / 3.0 * srf2;
            let force_i = mu1_dot_rh * mu2_dot_rh * rh / r4 * (srf2 * (1.0) * q2 - q2 * q * srf3);
            force_d + force_i
        }
    }

    /**
     * Interaction force between a point charge and a point quadrupole.
     *
     * Parameters:
     * - `charge`: Point charge (input charge) [UNIT: input charge]
     * - `quad`: Point quadrupole (input length^2 x input charge) [UNIT: (input length)^2 x (input charge)]
     * - `r`: Distance vector between particles (input length) [UNIT: input length]
     *
     * Returns:
     * Interaction force [UNIT: (input charge)^2 / (input length)^2]
     *
     * The force between a point charge and a point quadrupole is described by the formula:
     * F(charge, quad, r) = charge * E(quad, r)
     * where E(quad, r) is the field from the quadrupole at the location of the ion.
     */
    fn ion_quadrupole_force(&self, charge: f64, quad: Matrix3, r: Vector3) -> Vector3 {
        charge * self.quadrupole_field(&quad, &r)
    }
}

/// Test electric constant
#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    #[test]
    fn test_electric_constant() {
        let bjerrum_length = 7.1; // angstrom
        let rel_dielectric_const = 80.0;
        assert_relative_eq!(
            crate::TO_CHEMISTRY_UNIT / rel_dielectric_const / bjerrum_length,
            2.4460467895137676 // In kJ/mol, roughly 1 KT at room temperature
        );
    }
}
