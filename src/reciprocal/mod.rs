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
//! The API uses SoA (Structure-of-Arrays) layout: separate `&[f64]` slices
//! for x, y, z coordinates and charges. This matches common MD/MC engine
//! storage and enables future SIMD and GPU backends.
//!
//! # MC workflow
//! ```
//! # use coulomb::reciprocal::*;
//! # let mut ewald = EwaldReciprocal::new([10.0; 3], 0.8, 0.0, 5, BoundaryPermittivity::Tinfoil);
//! # let (x, y, z) = (vec![-0.5, 0.5], vec![0.0, 0.0], vec![0.0, 0.0]);
//! # let charges = vec![1.0, -1.0];
//! ewald.update_all(&x, &y, &z, &charges);
//! # let (old, new) = ([-0.5, 0.0, 0.0], [-0.3, 0.0, 0.0]);
//! let de = ewald.energy_change(1.0, old, new);
//! // if accept { ewald.update_particle(1.0, old, new); }
//! ```
//!
//! # MD workflow
//! ```
//! # use coulomb::reciprocal::*;
//! # let mut ewald = EwaldReciprocal::new([10.0; 3], 0.8, 0.0, 5, BoundaryPermittivity::Tinfoil);
//! # let (x, y, z) = (vec![-0.5, 0.5], vec![0.0, 0.0], vec![0.0, 0.0]);
//! # let charges = vec![1.0, -1.0];
//! ewald.update_all(&x, &y, &z, &charges);
//! let (mut fx, mut fy, mut fz) = (vec![0.0; 2], vec![0.0; 2], vec![0.0; 2]);
//! ewald.add_forces(&x, &y, &z, &charges, &mut fx, &mut fy, &mut fz);
//! ```

mod kvectors;
mod parameters;

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
/// Uses SoA (Structure-of-Arrays) layout throughout: positions are passed as
/// separate `x`, `y`, `z` slices, and forces are accumulated into caller-provided
/// buffers. This avoids allocation, matches common particle storage layouts, and
/// prepares for SIMD/GPU backends.
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
    /// Structure factors: Re(S(k)) for each k-vector
    sk_re: Vec<f64>,
    /// Structure factors: Im(S(k)) for each k-vector
    sk_im: Vec<f64>,
}

impl EwaldReciprocal {
    /// Construct with explicit Ewald parameters.
    ///
    /// - `box_length`: Side lengths `[Lx, Ly, Lz]` of the orthorhombic simulation box
    /// - `alpha`: Ewald splitting parameter (must be positive)
    /// - `kappa`: Inverse Debye screening length (0 for Coulomb, must be non-negative)
    /// - `num_k_vectors`: Spherical cutoff in integer k-space (must be ≥ 1)
    /// - `boundary`: Boundary permittivity condition
    ///
    /// # Panics
    /// Panics in debug mode if parameters are non-physical.
    pub fn new(
        box_length: [f64; 3],
        alpha: f64,
        kappa: f64,
        num_k_vectors: u32,
        boundary: BoundaryPermittivity,
    ) -> Self {
        debug_assert!(alpha > 0.0, "alpha must be positive");
        debug_assert!(kappa >= 0.0, "kappa must be non-negative");
        debug_assert!(num_k_vectors >= 1, "num_k_vectors must be at least 1");
        debug_assert!(
            box_length[0] > 0.0 && box_length[1] > 0.0 && box_length[2] > 0.0,
            "box lengths must be positive"
        );
        let volume = box_length[0] * box_length[1] * box_length[2];
        let kvecs = KVectors::new(box_length, num_k_vectors, alpha, kappa);
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
        box_length: [f64; 3],
        kappa: f64,
        accuracy: f64,
        num_particles: usize,
        boundary: BoundaryPermittivity,
    ) -> (Self, f64) {
        let params =
            parameters::optimize_ewald_parameters(&box_length, kappa, accuracy, num_particles);
        let ewald = Self::new(box_length, params.alpha, kappa, params.k_max, boundary);
        (ewald, params.real_cutoff)
    }

    /// Recompute all structure factors from scratch. O(N × N_k).
    pub fn update_all(&mut self, x: &[f64], y: &[f64], z: &[f64], charges: &[f64]) {
        self.sk_re.fill(0.0);
        self.sk_im.fill(0.0);
        for ((&charge, &px), (&py, &pz)) in
            charges.iter().zip(x.iter()).zip(y.iter().zip(z.iter()))
        {
            for ((((&kx, &ky), &kz), sk_re), sk_im) in self
                .kvecs
                .kx
                .iter()
                .zip(self.kvecs.ky.iter())
                .zip(self.kvecs.kz.iter())
                .zip(self.sk_re.iter_mut())
                .zip(self.sk_im.iter_mut())
            {
                let (sin_kr, cos_kr) = (kx * px + ky * py + kz * pz).sin_cos();
                *sk_re += charge * cos_kr;
                *sk_im += charge * sin_kr;
            }
        }
    }

    /// Commit a single-particle move. O(N_k).
    pub fn update_particle(&mut self, charge: f64, old: [f64; 3], new: [f64; 3]) {
        for ((((&kx, &ky), &kz), sk_re), sk_im) in self
            .kvecs
            .kx
            .iter()
            .zip(self.kvecs.ky.iter())
            .zip(self.kvecs.kz.iter())
            .zip(self.sk_re.iter_mut())
            .zip(self.sk_im.iter_mut())
        {
            let (sin_old, cos_old) =
                (kx * old[0] + ky * old[1] + kz * old[2]).sin_cos();
            let (sin_new, cos_new) =
                (kx * new[0] + ky * new[1] + kz * new[2]).sin_cos();
            *sk_re += charge * (cos_new - cos_old);
            *sk_im += charge * (sin_new - sin_old);
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
    pub fn energy_change(&self, charge: f64, old: [f64; 3], new: [f64; 3]) -> f64 {
        let prefactor = 2.0 * PI / self.volume;
        let mut sum = 0.0;
        for (((((&kx, &ky), &kz), &ak), &s_re), &s_im) in self
            .kvecs
            .kx
            .iter()
            .zip(self.kvecs.ky.iter())
            .zip(self.kvecs.kz.iter())
            .zip(self.kvecs.aks.iter())
            .zip(self.sk_re.iter())
            .zip(self.sk_im.iter())
        {
            let (sin_old, cos_old) =
                (kx * old[0] + ky * old[1] + kz * old[2]).sin_cos();
            let (sin_new, cos_new) =
                (kx * new[0] + ky * new[1] + kz * new[2]).sin_cos();
            let ds_re = charge * (cos_new - cos_old);
            let ds_im = charge * (sin_new - sin_old);
            let cross = 2.0 * (s_re * ds_re + s_im * ds_im);
            let ds_sq = ds_re * ds_re + ds_im * ds_im;
            sum += ak * (cross + ds_sq);
        }
        prefactor * sum
    }

    /// Force on one particle. O(N_k). Returns `[fx, fy, fz]`.
    ///
    /// F_i = (4πq_i/V) Σ_k A(k) (sin(k·r)·S_re - cos(k·r)·S_im) · k
    pub fn force(&self, pos: [f64; 3], charge: f64) -> [f64; 3] {
        let prefactor = 4.0 * PI * charge / self.volume;
        let (mut fx, mut fy, mut fz) = (0.0, 0.0, 0.0);
        for (((((&kx, &ky), &kz), &ak), &s_re), &s_im) in self
            .kvecs
            .kx
            .iter()
            .zip(self.kvecs.ky.iter())
            .zip(self.kvecs.kz.iter())
            .zip(self.kvecs.aks.iter())
            .zip(self.sk_re.iter())
            .zip(self.sk_im.iter())
        {
            let (sin_kr, cos_kr) = (kx * pos[0] + ky * pos[1] + kz * pos[2]).sin_cos();
            let term = ak * (sin_kr * s_re - cos_kr * s_im);
            fx += term * kx;
            fy += term * ky;
            fz += term * kz;
        }
        [prefactor * fx, prefactor * fy, prefactor * fz]
    }

    /// Accumulate reciprocal-space forces on all particles. O(N × N_k).
    ///
    /// Adds (+=) force contributions into the provided `fx`, `fy`, `fz` buffers.
    /// Caller is responsible for zeroing the buffers if starting fresh.
    #[allow(clippy::too_many_arguments)]
    pub fn add_forces(
        &self,
        x: &[f64],
        y: &[f64],
        z: &[f64],
        charges: &[f64],
        fx: &mut [f64],
        fy: &mut [f64],
        fz: &mut [f64],
    ) {
        for (((&charge, &px), (&py, &pz)), (out_fx, (out_fy, out_fz))) in charges
            .iter()
            .zip(x.iter())
            .zip(y.iter().zip(z.iter()))
            .zip(fx.iter_mut().zip(fy.iter_mut().zip(fz.iter_mut())))
        {
            let [dfx, dfy, dfz] = self.force([px, py, pz], charge);
            *out_fx += dfx;
            *out_fy += dfy;
            *out_fz += dfz;
        }
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
    pub fn surface_energy(
        &self,
        x: &[f64],
        y: &[f64],
        z: &[f64],
        charges: &[f64],
    ) -> f64 {
        let eps_s: f64 = self.boundary.into();
        if eps_s.is_infinite() {
            return 0.0;
        }
        let [dx, dy, dz] = dipole_moment(x, y, z, charges);
        2.0 * PI / ((2.0 * eps_s + 1.0) * self.volume) * (dx * dx + dy * dy + dz * dz)
    }

    /// Accumulate surface forces on all particles. O(N).
    ///
    /// Adds (+=) force contributions into the provided `fx`, `fy`, `fz` buffers.
    /// Does nothing for tinfoil boundary conditions.
    #[allow(clippy::too_many_arguments)]
    pub fn add_surface_forces(
        &self,
        x: &[f64],
        y: &[f64],
        z: &[f64],
        charges: &[f64],
        fx: &mut [f64],
        fy: &mut [f64],
        fz: &mut [f64],
    ) {
        let eps_s: f64 = self.boundary.into();
        if eps_s.is_infinite() {
            return;
        }
        let [dx, dy, dz] = dipole_moment(x, y, z, charges);
        let base = -4.0 * PI / ((2.0 * eps_s + 1.0) * self.volume);
        for ((&q, out_fx), (out_fy, out_fz)) in charges
            .iter()
            .zip(fx.iter_mut())
            .zip(fy.iter_mut().zip(fz.iter_mut()))
        {
            let c = base * q;
            *out_fx += c * dx;
            *out_fy += c * dy;
            *out_fz += c * dz;
        }
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

/// Calculate the dipole moment: Σ q_i r_i, returned as [dx, dy, dz].
fn dipole_moment(x: &[f64], y: &[f64], z: &[f64], charges: &[f64]) -> [f64; 3] {
    let (mut dx, mut dy, mut dz) = (0.0, 0.0, 0.0);
    for ((&q, &px), (&py, &pz)) in charges.iter().zip(x.iter()).zip(y.iter().zip(z.iter())) {
        dx += q * px;
        dy += q * py;
        dz += q * pz;
    }
    [dx, dy, dz]
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Reference values from C++ coulombgalore/test/unittests.cpp
    // Setup: box=10³, n_max=11, α=0.8, κ=0
    // Two charges [+1, -1] at [(-0.5,0,0), (0.5,0,0)]

    fn setup_two_charges() -> (EwaldReciprocal, [f64; 2], [f64; 2], [f64; 2], [f64; 2]) {
        let mut ewald =
            EwaldReciprocal::new([10.0; 3], 0.8, 0.0, 11, BoundaryPermittivity::Tinfoil);
        let x = [-0.5, 0.5];
        let y = [0.0, 0.0];
        let z = [0.0, 0.0];
        let charges = [1.0, -1.0];
        ewald.update_all(&x, &y, &z, &charges);
        (ewald, x, y, z, charges)
    }

    #[test]
    fn test_reciprocal_energy_two_charges() {
        let (ewald, _, _, _, _) = setup_two_charges();
        assert_relative_eq!(ewald.energy(), 0.1584768302, epsilon = 1e-6);
    }

    #[test]
    fn test_reciprocal_force_two_charges() {
        let (ewald, x, y, z, charges) = setup_two_charges();
        let [fx, fy, fz] = ewald.force([x[0], y[0], z[0]], charges[0]);
        assert_relative_eq!(fx, 0.2617988467, epsilon = 1e-6);
        assert_relative_eq!(fy, 0.0, epsilon = 1e-6);
        assert_relative_eq!(fz, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_self_energy_coulomb() {
        let ewald =
            EwaldReciprocal::new([10.0; 3], 0.8, 0.0, 11, BoundaryPermittivity::Tinfoil);
        assert_relative_eq!(
            ewald.self_energy(&[1.0, -1.0]),
            -0.9027033337,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_surface_energy_tinfoil() {
        let (ewald, x, y, z, charges) = setup_two_charges();
        assert_relative_eq!(ewald.surface_energy(&x, &y, &z, &charges), 0.0);
    }

    #[test]
    fn test_surface_energy_vacuum() {
        let mut ewald =
            EwaldReciprocal::new([10.0; 3], 0.8, 0.0, 11, BoundaryPermittivity::Vacuum);
        let (x, y, z) = ([-0.5, 0.5], [0.0, 0.0], [0.0, 0.0]);
        let charges = [1.0, -1.0];
        ewald.update_all(&x, &y, &z, &charges);
        assert_relative_eq!(
            ewald.surface_energy(&x, &y, &z, &charges),
            0.0020943951,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_forces_sum_to_zero() {
        let (ewald, x, y, z, charges) = setup_two_charges();
        let mut fx = [0.0; 2];
        let mut fy = [0.0; 2];
        let mut fz = [0.0; 2];
        ewald.add_forces(&x, &y, &z, &charges, &mut fx, &mut fy, &mut fz);
        assert_relative_eq!(fx.iter().sum::<f64>(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(fy.iter().sum::<f64>(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(fz.iter().sum::<f64>(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_add_forces_accumulates() {
        let (ewald, x, y, z, charges) = setup_two_charges();
        // Start with non-zero buffers to verify += behavior
        let mut fx = [100.0; 2];
        let mut fy = [200.0; 2];
        let mut fz = [300.0; 2];
        ewald.add_forces(&x, &y, &z, &charges, &mut fx, &mut fy, &mut fz);
        let [f0x, _, _] = ewald.force([x[0], y[0], z[0]], charges[0]);
        assert_relative_eq!(fx[0], 100.0 + f0x, epsilon = 1e-12);
    }

    #[test]
    fn test_force_matches_numerical_gradient() {
        let mut ewald =
            EwaldReciprocal::new([10.0; 3], 0.8, 0.0, 11, BoundaryPermittivity::Tinfoil);
        let (x, y, z) = ([-0.5, 0.5], [0.0, 0.0], [0.0, 0.0]);
        let charges = [1.0, -1.0];
        ewald.update_all(&x, &y, &z, &charges);

        let [fx, _, _] = ewald.force([x[0], y[0], z[0]], charges[0]);

        let delta = 1e-5;
        ewald.update_all(&[-0.5 + delta, 0.5], &y, &z, &charges);
        let e_plus = ewald.energy();
        ewald.update_all(&[-0.5 - delta, 0.5], &y, &z, &charges);
        let e_minus = ewald.energy();
        let numerical_fx = -(e_plus - e_minus) / (2.0 * delta);
        assert_relative_eq!(fx, numerical_fx, epsilon = 1e-4);
    }

    #[test]
    fn test_partial_update_equals_full() {
        let mut ewald =
            EwaldReciprocal::new([10.0; 3], 0.8, 0.0, 5, BoundaryPermittivity::Tinfoil);
        let (x, y, z) = ([1.0, -1.0, 2.0, -2.0], [2.0, 0.5, -1.0, 1.0], [3.0, -0.5, 1.5, -1.0]);
        let charges = [1.0, -1.0, 0.5, -0.5];

        ewald.update_all(&x, &y, &z, &charges);

        let old = [x[0], y[0], z[0]];
        let new = [1.5, 2.5, 3.5];
        ewald.update_particle(charges[0], old, new);
        let e_partial = ewald.energy();

        let x2 = [1.5, -1.0, 2.0, -2.0];
        let y2 = [2.5, 0.5, -1.0, 1.0];
        let z2 = [3.5, -0.5, 1.5, -1.0];
        ewald.update_all(&x2, &y2, &z2, &charges);
        let e_full = ewald.energy();

        assert_relative_eq!(e_partial, e_full, epsilon = 1e-12);
    }

    #[test]
    fn test_energy_change_particle() {
        let mut ewald =
            EwaldReciprocal::new([10.0; 3], 0.8, 0.0, 5, BoundaryPermittivity::Tinfoil);
        let (x, y, z) = ([1.0, -1.0, 2.0, -2.0], [2.0, 0.5, -1.0, 1.0], [3.0, -0.5, 1.5, -1.0]);
        let charges = [1.0, -1.0, 0.5, -0.5];

        ewald.update_all(&x, &y, &z, &charges);
        let e_before = ewald.energy();

        let old = [x[0], y[0], z[0]];
        let new = [1.5, 2.5, 3.5];
        let de = ewald.energy_change(charges[0], old, new);

        ewald.update_particle(charges[0], old, new);
        let e_after = ewald.energy();

        assert_relative_eq!(de, e_after - e_before, epsilon = 1e-12);
    }

    #[test]
    fn test_yukawa_reduces_to_coulomb() {
        let (x, y, z) = ([-0.5, 0.5], [0.0, 0.0], [0.0, 0.0]);
        let charges = [1.0, -1.0];

        let mut ewald_c =
            EwaldReciprocal::new([10.0; 3], 0.8, 0.0, 11, BoundaryPermittivity::Tinfoil);
        ewald_c.update_all(&x, &y, &z, &charges);

        let mut ewald_y =
            EwaldReciprocal::new([10.0; 3], 0.8, 1e-10, 11, BoundaryPermittivity::Tinfoil);
        ewald_y.update_all(&x, &y, &z, &charges);

        assert_relative_eq!(ewald_c.energy(), ewald_y.energy(), epsilon = 1e-6);
    }

    #[test]
    fn test_yukawa_energy_decays_with_kappa() {
        let (x, y, z) = ([-0.5, 0.5], [0.0, 0.0], [0.0, 0.0]);
        let charges = [1.0, -1.0];

        let mut energies = Vec::new();
        for &kappa in &[0.0, 0.1, 0.5, 1.0] {
            let mut ewald =
                EwaldReciprocal::new([10.0; 3], 0.8, kappa, 11, BoundaryPermittivity::Tinfoil);
            ewald.update_all(&x, &y, &z, &charges);
            energies.push(ewald.energy().abs());
        }
        for i in 1..energies.len() {
            assert!(energies[i] < energies[i - 1]);
        }
    }

    #[test]
    fn test_self_energy_yukawa() {
        let alpha = 0.8;
        let kappa = 0.5;
        let ewald =
            EwaldReciprocal::new([10.0; 3], alpha, kappa, 11, BoundaryPermittivity::Tinfoil);
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
        let (ewald, rc) = EwaldReciprocal::with_accuracy(
            [10.0; 3],
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
        let (ewald, x, y, z, charges) = setup_two_charges();
        let recip = ewald.energy();
        let self_e = ewald.self_energy(&charges);
        let surface = ewald.surface_energy(&x, &y, &z, &charges);

        assert_relative_eq!(recip, 0.1584768302, epsilon = 1e-6);
        assert_relative_eq!(self_e, -0.9027033337, epsilon = 1e-6);
        assert_relative_eq!(surface, 0.0);
        let partial = recip + self_e + surface;
        assert_relative_eq!(partial, -0.7442265035, epsilon = 1e-5);
    }

    #[test]
    fn test_add_surface_forces() {
        let ewald =
            EwaldReciprocal::new([10.0; 3], 0.8, 0.0, 11, BoundaryPermittivity::Vacuum);
        let (x, y, z) = ([-0.5, 0.5], [0.0, 0.0], [0.0, 0.0]);
        let charges = [1.0, -1.0];
        let mut fx = [0.0; 2];
        let mut fy = [0.0; 2];
        let mut fz = [0.0; 2];
        ewald.add_surface_forces(&x, &y, &z, &charges, &mut fx, &mut fy, &mut fz);
        // Dipole = (-1, 0, 0), so force should be along x
        assert!(fx[0].abs() > 0.0);
        assert_relative_eq!(fy[0], 0.0);
        assert_relative_eq!(fz[0], 0.0);
    }

    #[test]
    fn test_add_surface_forces_tinfoil_noop() {
        let ewald =
            EwaldReciprocal::new([10.0; 3], 0.8, 0.0, 11, BoundaryPermittivity::Tinfoil);
        let (x, y, z) = ([-0.5, 0.5], [0.0, 0.0], [0.0, 0.0]);
        let charges = [1.0, -1.0];
        let mut fx = [99.0; 2];
        let mut fy = [99.0; 2];
        let mut fz = [99.0; 2];
        ewald.add_surface_forces(&x, &y, &z, &charges, &mut fx, &mut fy, &mut fz);
        // Tinfoil: no surface forces, buffers should be unchanged
        assert_eq!(fx, [99.0; 2]);
        assert_eq!(fy, [99.0; 2]);
        assert_eq!(fz, [99.0; 2]);
    }

    #[test]
    fn test_madelung_nacl() {
        use crate::pairwise::{MultipoleEnergy, RealSpaceEwald};

        // NaCl Madelung constant via full Ewald summation (real + reciprocal + self).
        // 2×2×2 supercell: 64 ions, nearest-neighbor distance a = 1.
        let a = 1.0;
        let l_cell = 2.0 * a;
        let n_rep = 2_usize;
        let l_box = l_cell * n_rep as f64;

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

        let mut px = Vec::with_capacity(64);
        let mut py = Vec::with_capacity(64);
        let mut pz = Vec::with_capacity(64);
        let mut charges = Vec::with_capacity(64);
        for ix in 0..n_rep {
            for iy in 0..n_rep {
                for iz in 0..n_rep {
                    let ox = ix as f64 * l_cell;
                    let oy = iy as f64 * l_cell;
                    let oz = iz as f64 * l_cell;
                    for b in &na_basis {
                        px.push(b[0] + ox);
                        py.push(b[1] + oy);
                        pz.push(b[2] + oz);
                        charges.push(1.0);
                    }
                    for b in &cl_basis {
                        px.push(b[0] + ox);
                        py.push(b[1] + oy);
                        pz.push(b[2] + oz);
                        charges.push(-1.0);
                    }
                }
            }
        }
        assert_eq!(px.len(), 64);

        let alpha = 2.5;
        let r_c = l_box / 2.0 - 0.01;
        let k_max = 10;

        // Reciprocal space
        let mut ewald = EwaldReciprocal::new(
            [l_box; 3],
            alpha,
            0.0,
            k_max,
            BoundaryPermittivity::Tinfoil,
        );
        ewald.update_all(&px, &py, &pz, &charges);
        let e_recip = ewald.energy();
        let e_self = ewald.self_energy(&charges);

        // Real space: minimum-image pairwise sum
        let real_ewald = RealSpaceEwald::new_without_salt(r_c, alpha);
        let n = px.len();
        let mut e_real = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let mut dx = px[i] - px[j];
                let mut dy = py[i] - py[j];
                let mut dz = pz[i] - pz[j];
                dx -= l_box * (dx / l_box).round();
                dy -= l_box * (dy / l_box).round();
                dz -= l_box * (dz / l_box).round();
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                if r < r_c {
                    e_real += real_ewald.ion_ion_energy(charges[i], charges[j], r);
                }
            }
        }

        let e_total = e_real + e_recip + e_self;
        let n_pairs = n / 2;
        let madelung = -e_total * a / n_pairs as f64;

        // NaCl Madelung constant: 1.7475645946 (Wikipedia)
        assert_relative_eq!(madelung, 1.7475645946, epsilon = 1e-4);
    }
}
