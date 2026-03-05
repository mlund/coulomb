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

//! Reciprocal-space Ewald summation for Coulomb and Yukawa potentials.
//!
//! This module provides [`EwaldReciprocal`], which handles k-space energies,
//! forces, and efficient Monte Carlo partial updates for both unscreened
//! Coulomb (κ=0) and screened Yukawa (κ>0) interactions.
//!
//! # MC workflow
//! ```
//! # use coulomb::reciprocal::*;
//! # let box_length = [10.0, 10.0, 10.0].into();
//! # let mut ewald = EwaldReciprocal::new(box_length, 0.8, 0.0, 5, BoundaryPermittivity::Tinfoil);
//! # let positions = vec![[-0.5, 0.0, 0.0].into(), [0.5, 0.0, 0.0].into()];
//! # let charges = vec![1.0, -1.0];
//! ewald.update_all(&positions, &charges);
//! # let (charge, old, new) = (1.0, [-0.5, 0.0, 0.0].into(), [-0.3, 0.0, 0.0].into());
//! let de = ewald.energy_change(charge, old, new);
//! // if accept { ewald.update_particle(charge, old, new); }
//! ```
//!
//! # MD workflow
//! ```
//! # use coulomb::reciprocal::*;
//! # let box_length = [10.0, 10.0, 10.0].into();
//! # let mut ewald = EwaldReciprocal::new(box_length, 0.8, 0.0, 5, BoundaryPermittivity::Tinfoil);
//! # let positions = vec![[-0.5, 0.0, 0.0].into(), [0.5, 0.0, 0.0].into()];
//! # let charges = vec![1.0, -1.0];
//! ewald.update_all(&positions, &charges);
//! let forces = ewald.forces(&positions, &charges);
//! ```

mod kvectors;
mod parameters;

use crate::{NalgebraVector3, Vector3};
use core::f64::consts::PI;
use kvectors::KVectors;

/// Relative permittivity at the boundary.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum BoundaryPermittivity {
    /// Tinfoil or metallic boundary conditions, i.e. infinite permittivity (default).
    #[default]
    Tinfoil,
    /// Vacuum boundary conditions, i.e. permittivity of free space = 1.
    Vacuum,
    /// Custom permittivity.
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

/// Reciprocal-space Ewald summation for Coulomb and Yukawa potentials.
///
/// Handles k-space energies, forces, and efficient MC partial updates.
/// For Coulomb, set κ=0. For Yukawa, set κ = 1/λ_D.
///
/// Internally uses half-shell k-vectors (only one of each ±k pair) with
/// doubled A(k) coefficients for efficiency.
pub struct EwaldReciprocal {
    kvecs: KVectors,
    volume: f64,
    /// Retained for `self_energy()` which needs the raw splitting parameter.
    alpha: f64,
    /// Retained for `self_energy()` which needs the raw screening parameter.
    kappa: f64,
    boundary: BoundaryPermittivity,
    /// Complex structure factors: Re(S(k)) for each k-vector
    sk_re: Vec<f64>,
    /// Complex structure factors: Im(S(k)) for each k-vector
    sk_im: Vec<f64>,
}

impl EwaldReciprocal {
    /// Construct with explicit Ewald parameters.
    ///
    /// - `box_length`: Side lengths of the orthorhombic simulation box (must be positive)
    /// - `alpha`: Ewald splitting parameter (must be positive)
    /// - `kappa`: Inverse Debye screening length (0 for Coulomb, must be non-negative)
    /// - `num_k_vectors`: Spherical cutoff in integer k-space (must be ≥ 1)
    /// - `boundary`: Boundary permittivity condition
    ///
    /// # Panics
    /// Panics in debug mode if `alpha <= 0`, `kappa < 0`, `num_k_vectors == 0`,
    /// or any box length is non-positive.
    pub fn new(
        box_length: Vector3,
        alpha: f64,
        kappa: f64,
        num_k_vectors: u32,
        boundary: BoundaryPermittivity,
    ) -> Self {
        debug_assert!(alpha > 0.0, "alpha must be positive");
        debug_assert!(kappa >= 0.0, "kappa must be non-negative");
        debug_assert!(num_k_vectors >= 1, "num_k_vectors must be at least 1");
        let bl: NalgebraVector3 = box_length.into();
        debug_assert!(
            bl.x > 0.0 && bl.y > 0.0 && bl.z > 0.0,
            "box lengths must be positive"
        );
        let volume = bl.x * bl.y * bl.z;
        let kvecs = KVectors::new(bl, num_k_vectors, alpha, kappa);
        let n_k = kvecs.len();
        Self {
            kvecs,
            volume,
            alpha,
            kappa,
            boundary,
            sk_re: vec![0.0; n_k],
            sk_im: vec![0.0; n_k],
        }
    }

    /// Construct with automatic parameter selection for target accuracy.
    ///
    /// Returns the `EwaldReciprocal` along with the recommended real-space cutoff.
    pub fn with_accuracy(
        box_length: Vector3,
        kappa: f64,
        accuracy: f64,
        num_particles: usize,
        boundary: BoundaryPermittivity,
    ) -> (Self, f64) {
        let bl: NalgebraVector3 = box_length.into();
        let params = parameters::optimize_ewald_parameters(&bl, kappa, accuracy, num_particles);
        let ewald = Self::new(box_length, params.alpha, kappa, params.k_max, boundary);
        (ewald, params.real_cutoff)
    }

    /// Recompute all structure factors from scratch. O(N × N_k).
    pub fn update_all(&mut self, positions: &[Vector3], charges: &[f64]) {
        self.sk_re.fill(0.0);
        self.sk_im.fill(0.0);
        for (&charge, pos) in charges.iter().zip(positions.iter()) {
            let r: NalgebraVector3 = (*pos).into();
            for ((k, sk_re), sk_im) in self
                .kvecs
                .vectors
                .iter()
                .zip(self.sk_re.iter_mut())
                .zip(self.sk_im.iter_mut())
            {
                let (sin_kr, cos_kr) = k.dot(&r).sin_cos();
                *sk_re += charge * cos_kr;
                *sk_im += charge * sin_kr;
            }
        }
    }

    /// Commit a single-particle move. O(N_k).
    pub fn update_particle(&mut self, charge: f64, old_pos: Vector3, new_pos: Vector3) {
        let r_old: NalgebraVector3 = old_pos.into();
        let r_new: NalgebraVector3 = new_pos.into();
        for ((k, sk_re), sk_im) in self
            .kvecs
            .vectors
            .iter()
            .zip(self.sk_re.iter_mut())
            .zip(self.sk_im.iter_mut())
        {
            let (sin_old, cos_old) = k.dot(&r_old).sin_cos();
            let (sin_new, cos_new) = k.dot(&r_new).sin_cos();
            *sk_re += charge * (cos_new - cos_old);
            *sk_im += charge * (sin_new - sin_old);
        }
    }

    /// Commit a cluster move. O(N_cluster × N_k).
    pub fn update_cluster(
        &mut self,
        charges: &[f64],
        old_positions: &[Vector3],
        new_positions: &[Vector3],
    ) {
        for ((&charge, old_pos), new_pos) in
            charges.iter().zip(old_positions).zip(new_positions)
        {
            self.update_particle(charge, *old_pos, *new_pos);
        }
    }

    /// Reciprocal-space energy. O(N_k).
    ///
    /// U_k = (2π/V) Σ_k A(k) |S(k)|²
    pub fn energy(&self) -> f64 {
        let prefactor = 2.0 * PI / self.volume;
        let sum: f64 = self
            .kvecs
            .aks
            .iter()
            .zip(self.sk_re.iter())
            .zip(self.sk_im.iter())
            .map(|((&ak, &re), &im)| ak * (re * re + im * im))
            .sum();
        prefactor * sum
    }

    /// Energy change for a trial particle move, without mutating state. O(N_k).
    ///
    /// ΔU = (2π/V) Σ_k A(k) [2 Re(S(k)·ΔS*(k)) + |ΔS(k)|²]
    pub fn energy_change(&self, charge: f64, old_pos: Vector3, new_pos: Vector3) -> f64 {
        let r_old: NalgebraVector3 = old_pos.into();
        let r_new: NalgebraVector3 = new_pos.into();
        let prefactor = 2.0 * PI / self.volume;
        let mut sum = 0.0;
        for ((&ak, (&s_re, &s_im)), k) in self
            .kvecs
            .aks
            .iter()
            .zip(self.sk_re.iter().zip(self.sk_im.iter()))
            .zip(self.kvecs.vectors.iter())
        {
            let (sin_old, cos_old) = k.dot(&r_old).sin_cos();
            let (sin_new, cos_new) = k.dot(&r_new).sin_cos();
            let ds_re = charge * (cos_new - cos_old);
            let ds_im = charge * (sin_new - sin_old);
            let cross = 2.0 * (s_re * ds_re + s_im * ds_im);
            let ds_sq = ds_re * ds_re + ds_im * ds_im;
            sum += ak * (cross + ds_sq);
        }
        prefactor * sum
    }

    /// Force on one particle. O(N_k).
    ///
    /// F_i = (4πq_i/V) Σ_k A(k) (sin(k·r)·S_re - cos(k·r)·S_im) · k
    pub fn force(&self, position: Vector3, charge: f64) -> Vector3 {
        let r: NalgebraVector3 = position.into();
        let prefactor = 4.0 * PI * charge / self.volume;
        let mut f = NalgebraVector3::zeros();
        for ((&ak, (&s_re, &s_im)), k) in self
            .kvecs
            .aks
            .iter()
            .zip(self.sk_re.iter().zip(self.sk_im.iter()))
            .zip(self.kvecs.vectors.iter())
        {
            let (sin_kr, cos_kr) = k.dot(&r).sin_cos();
            let term = sin_kr * s_re - cos_kr * s_im;
            f += ak * term * k;
        }
        (prefactor * f).into()
    }

    /// Forces on all particles. O(N × N_k).
    pub fn forces(&self, positions: &[Vector3], charges: &[f64]) -> Vec<Vector3> {
        positions
            .iter()
            .zip(charges.iter())
            .map(|(pos, &charge)| self.force(*pos, charge))
            .collect()
    }

    /// Self-energy correction (position-independent). O(N).
    ///
    /// For Coulomb (κ=0): U_self = -(α/√π) Σ q²
    /// For Yukawa (κ>0): U_self = -(α/√π) exp(-κ²/(4α²)) Σ q² + (κ/2) erfc(κ/(2α)) Σ q²
    pub fn self_energy(&self, charges: &[f64]) -> f64 {
        let sum_q2: f64 = charges.iter().map(|q| q * q).sum();
        let alpha = self.alpha;
        let sqrt_pi = PI.sqrt();

        if self.kappa > 0.0 {
            let kappa = self.kappa;
            let exp_term = (-kappa * kappa / (4.0 * alpha * alpha)).exp();
            let erfc_term = crate::math::erfc_x(kappa / (2.0 * alpha));
            (-alpha / sqrt_pi * exp_term + kappa / 2.0 * erfc_term) * sum_q2
        } else {
            -alpha / sqrt_pi * sum_q2
        }
    }

    /// Surface energy for non-tinfoil boundaries. O(N).
    ///
    /// U_surface = 2π / ((2ε_s + 1) V) |Σ q_i r_i|²
    ///
    /// Returns 0 for tinfoil boundary conditions.
    pub fn surface_energy(&self, positions: &[Vector3], charges: &[f64]) -> f64 {
        let eps_s: f64 = self.boundary.into();
        if eps_s.is_infinite() {
            return 0.0;
        }
        let dipole = dipole_moment(positions, charges);
        2.0 * PI / ((2.0 * eps_s + 1.0) * self.volume) * dipole.norm_squared()
    }

    /// Surface force on a particle. O(1) per particle after dipole computation.
    ///
    /// F_surface_i = -4π q_i / ((2ε_s + 1) V) × Σ q_j r_j
    ///
    /// Note: if calling for all particles, prefer [`surface_forces`](Self::surface_forces)
    /// which computes the dipole moment once.
    pub fn surface_force(
        &self,
        charge: f64,
        positions: &[Vector3],
        charges: &[f64],
    ) -> Vector3 {
        let eps_s: f64 = self.boundary.into();
        if eps_s.is_infinite() {
            return Vector3 { x: 0.0, y: 0.0, z: 0.0 };
        }
        let dipole = dipole_moment(positions, charges);
        let prefactor = -4.0 * PI * charge / ((2.0 * eps_s + 1.0) * self.volume);
        (prefactor * dipole).into()
    }

    /// Surface forces on all particles. O(N).
    ///
    /// Computes the dipole moment once and applies it to all particles.
    pub fn surface_forces(&self, positions: &[Vector3], charges: &[f64]) -> Vec<Vector3> {
        let eps_s: f64 = self.boundary.into();
        if eps_s.is_infinite() {
            return vec![Vector3 { x: 0.0, y: 0.0, z: 0.0 }; charges.len()];
        }
        let dipole = dipole_moment(positions, charges);
        let base = -4.0 * PI / ((2.0 * eps_s + 1.0) * self.volume);
        charges
            .iter()
            .map(|&q| (base * q * dipole).into())
            .collect()
    }

    /// Number of k-vectors (for diagnostics).
    pub fn num_k_vectors(&self) -> usize {
        self.kvecs.len()
    }

    /// The splitting parameter α (for diagnostics/pairing with RealSpaceEwald).
    pub fn alpha(&self) -> f64 {
        self.alpha
    }
}

/// Calculate the dipole moment: Σ q_i r_i
fn dipole_moment(positions: &[Vector3], charges: &[f64]) -> NalgebraVector3 {
    charges
        .iter()
        .zip(positions.iter())
        .map(|(&q, p)| {
            let r: NalgebraVector3 = (*p).into();
            q * r
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Reference values from C++ coulombgalore/test/unittests.cpp
    // Setup: box=10³, cutoff=5, n_max=11, α=0.8, κ=0
    // Two charges [+1, -1] at [(-0.5,0,0), (0.5,0,0)]

    fn setup_two_charges() -> (EwaldReciprocal, Vec<Vector3>, Vec<f64>) {
        let box_length: Vector3 = [10.0, 10.0, 10.0].into();
        let alpha = 0.8;
        let mut ewald =
            EwaldReciprocal::new(box_length, alpha, 0.0, 11, BoundaryPermittivity::Tinfoil);
        let positions = vec![[-0.5, 0.0, 0.0].into(), [0.5, 0.0, 0.0].into()];
        let charges = vec![1.0, -1.0];
        ewald.update_all(&positions, &charges);
        (ewald, positions, charges)
    }

    #[test]
    fn test_reciprocal_energy_two_charges() {
        let (ewald, _, _) = setup_two_charges();
        assert_relative_eq!(ewald.energy(), 0.1584768302, epsilon = 1e-6);
    }

    #[test]
    fn test_reciprocal_force_two_charges() {
        let (ewald, positions, charges) = setup_two_charges();
        let force = ewald.force(positions[0], charges[0]);
        let f: NalgebraVector3 = force.into();
        assert_relative_eq!(f.x, 0.2617988467, epsilon = 1e-6);
        assert_relative_eq!(f.y, 0.0, epsilon = 1e-6);
        assert_relative_eq!(f.z, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_self_energy_coulomb() {
        let box_length: Vector3 = [10.0, 10.0, 10.0].into();
        let ewald =
            EwaldReciprocal::new(box_length, 0.8, 0.0, 11, BoundaryPermittivity::Tinfoil);
        assert_relative_eq!(
            ewald.self_energy(&[1.0, -1.0]),
            -0.9027033337,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_surface_energy_tinfoil() {
        let (ewald, positions, charges) = setup_two_charges();
        assert_relative_eq!(ewald.surface_energy(&positions, &charges), 0.0);
    }

    #[test]
    fn test_surface_energy_vacuum() {
        let box_length: Vector3 = [10.0, 10.0, 10.0].into();
        let mut ewald =
            EwaldReciprocal::new(box_length, 0.8, 0.0, 11, BoundaryPermittivity::Vacuum);
        let positions: Vec<Vector3> = vec![[-0.5, 0.0, 0.0].into(), [0.5, 0.0, 0.0].into()];
        let charges = vec![1.0, -1.0];
        ewald.update_all(&positions, &charges);
        assert_relative_eq!(
            ewald.surface_energy(&positions, &charges),
            0.0020943951,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_forces_sum_to_zero() {
        let (ewald, positions, charges) = setup_two_charges();
        let forces = ewald.forces(&positions, &charges);
        let sum: NalgebraVector3 = forces.iter().map(|f| NalgebraVector3::from(*f)).sum();
        assert_relative_eq!(sum.norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_force_matches_numerical_gradient() {
        let box_length: Vector3 = [10.0, 10.0, 10.0].into();
        let mut ewald =
            EwaldReciprocal::new(box_length, 0.8, 0.0, 11, BoundaryPermittivity::Tinfoil);
        let positions: Vec<Vector3> = vec![[-0.5, 0.0, 0.0].into(), [0.5, 0.0, 0.0].into()];
        let charges = vec![1.0, -1.0];
        ewald.update_all(&positions, &charges);

        let force = ewald.force(positions[0], charges[0]);
        let f: NalgebraVector3 = force.into();

        let delta = 1e-5;
        let pos_plus: Vec<Vector3> =
            vec![[-0.5 + delta, 0.0, 0.0].into(), [0.5, 0.0, 0.0].into()];
        let pos_minus: Vec<Vector3> =
            vec![[-0.5 - delta, 0.0, 0.0].into(), [0.5, 0.0, 0.0].into()];
        ewald.update_all(&pos_plus, &charges);
        let e_plus = ewald.energy();
        ewald.update_all(&pos_minus, &charges);
        let e_minus = ewald.energy();
        let numerical_fx = -(e_plus - e_minus) / (2.0 * delta);
        assert_relative_eq!(f.x, numerical_fx, epsilon = 1e-4);
    }

    #[test]
    fn test_partial_update_equals_full() {
        let box_length: Vector3 = [10.0, 10.0, 10.0].into();
        let mut ewald =
            EwaldReciprocal::new(box_length, 0.8, 0.0, 5, BoundaryPermittivity::Tinfoil);
        let positions: Vec<Vector3> = vec![
            [1.0, 2.0, 3.0].into(),
            [-1.0, 0.5, -0.5].into(),
            [2.0, -1.0, 1.5].into(),
            [-2.0, 1.0, -1.0].into(),
        ];
        let charges = vec![1.0, -1.0, 0.5, -0.5];

        ewald.update_all(&positions, &charges);

        let new_pos: Vector3 = [1.5, 2.5, 3.5].into();
        ewald.update_particle(charges[0], positions[0], new_pos);
        let e_partial = ewald.energy();

        let mut new_positions = positions.clone();
        new_positions[0] = new_pos;
        ewald.update_all(&new_positions, &charges);
        let e_full = ewald.energy();

        assert_relative_eq!(e_partial, e_full, epsilon = 1e-12);
    }

    #[test]
    fn test_energy_change_particle() {
        let box_length: Vector3 = [10.0, 10.0, 10.0].into();
        let mut ewald =
            EwaldReciprocal::new(box_length, 0.8, 0.0, 5, BoundaryPermittivity::Tinfoil);
        let positions: Vec<Vector3> = vec![
            [1.0, 2.0, 3.0].into(),
            [-1.0, 0.5, -0.5].into(),
            [2.0, -1.0, 1.5].into(),
            [-2.0, 1.0, -1.0].into(),
        ];
        let charges = vec![1.0, -1.0, 0.5, -0.5];

        ewald.update_all(&positions, &charges);
        let e_before = ewald.energy();

        let new_pos: Vector3 = [1.5, 2.5, 3.5].into();
        let de = ewald.energy_change(charges[0], positions[0], new_pos);

        ewald.update_particle(charges[0], positions[0], new_pos);
        let e_after = ewald.energy();

        assert_relative_eq!(de, e_after - e_before, epsilon = 1e-12);
    }

    #[test]
    fn test_cluster_update_equals_full() {
        let box_length: Vector3 = [10.0, 10.0, 10.0].into();
        let mut ewald =
            EwaldReciprocal::new(box_length, 0.8, 0.0, 5, BoundaryPermittivity::Tinfoil);
        let positions: Vec<Vector3> = vec![
            [1.0, 2.0, 3.0].into(),
            [-1.0, 0.5, -0.5].into(),
            [2.0, -1.0, 1.5].into(),
            [-2.0, 1.0, -1.0].into(),
        ];
        let charges = vec![1.0, -1.0, 0.5, -0.5];

        ewald.update_all(&positions, &charges);

        let old_pos = [positions[0], positions[1]];
        let new_pos: Vec<Vector3> = vec![[1.5, 2.5, 3.5].into(), [-0.5, 1.0, 0.0].into()];
        let cluster_charges = [charges[0], charges[1]];
        ewald.update_cluster(&cluster_charges, &old_pos, &new_pos);
        let e_cluster = ewald.energy();

        let mut new_positions = positions.clone();
        new_positions[0] = new_pos[0];
        new_positions[1] = new_pos[1];
        ewald.update_all(&new_positions, &charges);
        let e_full = ewald.energy();

        assert_relative_eq!(e_cluster, e_full, epsilon = 1e-12);
    }

    #[test]
    fn test_yukawa_reduces_to_coulomb() {
        let box_length: Vector3 = [10.0, 10.0, 10.0].into();
        let positions: Vec<Vector3> = vec![[-0.5, 0.0, 0.0].into(), [0.5, 0.0, 0.0].into()];
        let charges = vec![1.0, -1.0];

        let mut ewald_c =
            EwaldReciprocal::new(box_length, 0.8, 0.0, 11, BoundaryPermittivity::Tinfoil);
        ewald_c.update_all(&positions, &charges);

        let mut ewald_y =
            EwaldReciprocal::new(box_length, 0.8, 1e-10, 11, BoundaryPermittivity::Tinfoil);
        ewald_y.update_all(&positions, &charges);

        assert_relative_eq!(ewald_c.energy(), ewald_y.energy(), epsilon = 1e-6);
    }

    #[test]
    fn test_yukawa_energy_decays_with_kappa() {
        let box_length: Vector3 = [10.0, 10.0, 10.0].into();
        let positions: Vec<Vector3> = vec![[-0.5, 0.0, 0.0].into(), [0.5, 0.0, 0.0].into()];
        let charges = vec![1.0, -1.0];

        let mut energies = Vec::new();
        for &kappa in &[0.0, 0.1, 0.5, 1.0] {
            let mut ewald =
                EwaldReciprocal::new(box_length, 0.8, kappa, 11, BoundaryPermittivity::Tinfoil);
            ewald.update_all(&positions, &charges);
            energies.push(ewald.energy().abs());
        }
        for i in 1..energies.len() {
            assert!(energies[i] < energies[i - 1]);
        }
    }

    #[test]
    fn test_self_energy_yukawa() {
        let box_length: Vector3 = [10.0, 10.0, 10.0].into();
        let alpha = 0.8;
        let kappa = 0.5;
        let ewald =
            EwaldReciprocal::new(box_length, alpha, kappa, 11, BoundaryPermittivity::Tinfoil);
        let charges = [1.0, -1.0];
        let self_e = ewald.self_energy(&charges);
        let sum_q2: f64 = charges.iter().map(|q| q * q).sum();
        let expected = (-alpha / PI.sqrt() * (-kappa * kappa / (4.0 * alpha * alpha)).exp()
            + kappa / 2.0 * crate::math::erfc_x(kappa / (2.0 * alpha)))
            * sum_q2;
        assert_relative_eq!(self_e, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_with_accuracy() {
        let box_length: Vector3 = [10.0, 10.0, 10.0].into();
        let (ewald, rc) = EwaldReciprocal::with_accuracy(
            box_length,
            0.0,
            1e-5,
            100,
            BoundaryPermittivity::Tinfoil,
        );
        assert!(ewald.alpha() > 0.0);
        assert!(rc > 0.0);
        assert!(rc < 5.0);
        assert!(ewald.num_k_vectors() > 0);
    }

    #[test]
    fn test_total_energy_two_charges() {
        let (ewald, positions, charges) = setup_two_charges();
        let recip = ewald.energy();
        let self_e = ewald.self_energy(&charges);
        let surface = ewald.surface_energy(&positions, &charges);

        assert_relative_eq!(recip, 0.1584768302, epsilon = 1e-6);
        assert_relative_eq!(self_e, -0.9027033337, epsilon = 1e-6);
        assert_relative_eq!(surface, 0.0);
        let partial = recip + self_e + surface;
        assert_relative_eq!(partial, -0.7442265035, epsilon = 1e-5);
    }

    #[test]
    fn test_surface_forces_batch() {
        let box_length: Vector3 = [10.0, 10.0, 10.0].into();
        let ewald =
            EwaldReciprocal::new(box_length, 0.8, 0.0, 11, BoundaryPermittivity::Vacuum);
        let positions: Vec<Vector3> = vec![[-0.5, 0.0, 0.0].into(), [0.5, 0.0, 0.0].into()];
        let charges = vec![1.0, -1.0];

        let batch = ewald.surface_forces(&positions, &charges);
        for (i, &charge) in charges.iter().enumerate() {
            let single = ewald.surface_force(charge, &positions, &charges);
            let b: NalgebraVector3 = batch[i].into();
            let s: NalgebraVector3 = single.into();
            assert_relative_eq!(b, s, epsilon = 1e-15);
        }
    }

    #[test]
    fn test_madelung_nacl() {
        use crate::pairwise::{MultipoleEnergy, RealSpaceEwald};

        // NaCl Madelung constant via full Ewald summation (real + reciprocal + self).
        // 2×2×2 supercell: 64 ions in a cubic box with nearest-neighbor distance a = 1.
        let a = 1.0;
        let l_cell = 2.0 * a;
        let n_rep = 2_usize;
        let l_box = l_cell * n_rep as f64;

        // NaCl basis in conventional cell (lattice constant = 2a)
        let na_basis: [[f64; 3]; 4] = [
            [0.0, 0.0, 0.0],
            [a, a, 0.0],
            [a, 0.0, a],
            [0.0, a, a],
        ];
        let cl_basis: [[f64; 3]; 4] = [
            [a, 0.0, 0.0],
            [0.0, a, 0.0],
            [0.0, 0.0, a],
            [a, a, a],
        ];

        let mut positions: Vec<Vector3> = Vec::with_capacity(64);
        let mut charges: Vec<f64> = Vec::with_capacity(64);
        for ix in 0..n_rep {
            for iy in 0..n_rep {
                for iz in 0..n_rep {
                    let ox = ix as f64 * l_cell;
                    let oy = iy as f64 * l_cell;
                    let oz = iz as f64 * l_cell;
                    for b in &na_basis {
                        positions.push([b[0] + ox, b[1] + oy, b[2] + oz].into());
                        charges.push(1.0);
                    }
                    for b in &cl_basis {
                        positions.push([b[0] + ox, b[1] + oy, b[2] + oz].into());
                        charges.push(-1.0);
                    }
                }
            }
        }
        assert_eq!(positions.len(), 64);

        let alpha = 2.5;
        let r_c = l_box / 2.0 - 0.01;
        let k_max = 10;
        let box_length: Vector3 = [l_box, l_box, l_box].into();

        // Reciprocal space
        let mut ewald =
            EwaldReciprocal::new(box_length, alpha, 0.0, k_max, BoundaryPermittivity::Tinfoil);
        ewald.update_all(&positions, &charges);
        let e_recip = ewald.energy();
        let e_self = ewald.self_energy(&charges);

        // Real space: minimum-image pairwise sum
        let real_ewald = RealSpaceEwald::new_without_salt(r_c, alpha);
        let n = positions.len();
        let mut e_real = 0.0;
        for i in 0..n {
            let pi: NalgebraVector3 = positions[i].into();
            for j in (i + 1)..n {
                let pj: NalgebraVector3 = positions[j].into();
                let mut dr = pi - pj;
                dr.x -= l_box * (dr.x / l_box).round();
                dr.y -= l_box * (dr.y / l_box).round();
                dr.z -= l_box * (dr.z / l_box).round();
                let r = dr.norm();
                if r < r_c {
                    e_real += real_ewald.ion_ion_energy(charges[i], charges[j], r);
                }
            }
        }

        let e_total = e_real + e_recip + e_self;
        let n_pairs = n / 2;
        let madelung = -e_total * a / n_pairs as f64;

        // NaCl Madelung constant: 1.7475645946...
        assert_relative_eq!(madelung, 1.7475645946, epsilon = 1e-4);
    }

    #[test]
    fn test_surface_forces_tinfoil_zero() {
        let box_length: Vector3 = [10.0, 10.0, 10.0].into();
        let ewald =
            EwaldReciprocal::new(box_length, 0.8, 0.0, 11, BoundaryPermittivity::Tinfoil);
        let positions: Vec<Vector3> = vec![[-0.5, 0.0, 0.0].into(), [0.5, 0.0, 0.0].into()];
        let charges = vec![1.0, -1.0];
        let forces = ewald.surface_forces(&positions, &charges);
        for f in &forces {
            let fv: NalgebraVector3 = (*f).into();
            assert_relative_eq!(fv.norm(), 0.0);
        }
    }
}
