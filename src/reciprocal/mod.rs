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
//! Coulomb ($\kappa=0$) and screened Yukawa ($\kappa>0$) interactions.
//!
//! # Theory
//!
//! The Ewald method splits the pair potential into real-space and reciprocal-space
//! contributions. For a system of $N$ charges $q_i$ in an orthorhombic box with
//! side lengths $L_x, L_y, L_z$ and volume $V = L_x L_y L_z$, the total
//! electrostatic energy is:
//!
//! $$U = U_\text{real} + U_\text{recip} + U_\text{self} + U_\text{surface}$$
//!
//! This module handles $U_\text{recip}$, $U_\text{self}$, and $U_\text{surface}$.
//!
//! ## Reciprocal-space energy
//!
//! $$U_\text{recip} = \frac{2\pi}{V} \sum_{\mathbf{k} \neq 0} A(\mathbf{k}) \, |S(\mathbf{k})|^2$$
//!
//! where the structure factor is $S(\mathbf{k}) = \sum_j q_j e^{i\mathbf{k} \cdot \mathbf{r}_j}$
//! and the k-space coefficient is:
//!
//! $$A(\mathbf{k}) = \frac{\exp\!\bigl(-(k^2 + \kappa^2) / 4\alpha^2\bigr)}{k^2 + \kappa^2}$$
//!
//! For Coulomb ($\kappa = 0$), this reduces to $A(\mathbf{k}) = \exp(-k^2/4\alpha^2)/k^2$.
//!
//! ## Self-energy correction
//!
//! The Ewald real-space sum inadvertently includes the interaction of each charge
//! with its own Gaussian image. This must be subtracted:
//!
//! $$U_\text{self} = -\frac{\alpha}{\sqrt\pi} e^{-\kappa^2/4\alpha^2} \sum_i q_i^2
//!   + \frac{\kappa}{2} \operatorname{erfc}\!\left(\frac{\kappa}{2\alpha}\right) \sum_i q_i^2$$
//!
//! For Coulomb, only the first term survives: $U_\text{self} = -(\alpha/\sqrt\pi) \sum_i q_i^2$.
//!
//! ## Surface term
//!
//! For non-tinfoil (finite $\varepsilon_s$) boundary conditions:
//!
//! $$U_\text{surface} = \frac{2\pi}{(2\varepsilon_s + 1) V} \left|\sum_i q_i \mathbf{r}_i\right|^2$$
//!
//! This term vanishes for metallic (tinfoil, $\varepsilon_s = \infty$) boundaries.
//!
//! ## IPBC (Isotropic Periodic Boundary Conditions)
//!
//! IPBC ([Stenqvist & Lund, 2018](https://doi.org/10/css8)) decomposes
//! $\cos(\mathbf{k} \cdot \mathbf{r}_{ij})$ into products of per-axis
//! $\cos/\sin$ terms. This requires only first-octant k-vectors ($n_x, n_y, n_z \geq 0$)
//! and stores 8 structure-factor components $Q_{\alpha\beta\gamma}$ per k-vector,
//! with appropriate octant symmetry factors baked into $A(\mathbf{k})$.
//!
//! # Parameter optimization
//!
//! Automatic parameter selection uses Kolafa–Perram error estimates:
//!
//! - **Real-space cutoff**: $r_c = L_\text{min} / 2$ (minimum image convention limit)
//! - **Splitting parameter**: $\alpha = \sqrt{-\ln \varepsilon} / r_c$, balancing
//!   real-space error $\sim \exp(-\alpha^2 r_c^2)$
//! - **Reciprocal cutoff**: $n_\text{max} = \lceil \alpha L_\text{max} \sqrt{-\ln \varepsilon} / \pi \rceil$
//!
//! For Yukawa ($\kappa > 0$), [Salin & Caillol (2000)](https://doi.org/10.1063/1.1326477)
//! showed that the optimal splitting parameter is independent of $\kappa$ (Eq. 3.3),
//! so the same $\alpha$ and $n_\text{max}$ work for both Coulomb and Yukawa.
//!
//! # Mixed precision
//!
//! K-vectors and $A(\mathbf{k})$ coefficients are stored as `f32`, halving memory
//! bandwidth and enabling wider SIMD. Trigonometric evaluations use `f32` arithmetic,
//! while structure factor accumulators and energy sums use `f64` for numerical safety.
//! On x86_64 with the `simd` feature, a polynomial `sin_cos` avoids per-lane libm calls.
//!
//! # MC workflow
//! ```
//! # use coulomb::reciprocal::*;
//! let (mut ewald, r_cutoff) = EwaldReciprocal::new([10.0; 3], 1e-5, None);
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
//! let (mut ewald, _) = EwaldReciprocal::new([10.0; 3], 1e-5, None);
//! # let (x, y, z) = (vec![-0.5, 0.5], vec![0.0, 0.0], vec![0.0, 0.0]);
//! # let charges = vec![1.0, -1.0];
//! ewald.update_all(&x, &y, &z, &charges);
//! let (mut fx, mut fy, mut fz) = (vec![0.0; 2], vec![0.0; 2], vec![0.0; 2]);
//! ewald.add_forces(&x, &y, &z, &charges, &mut fx, &mut fy, &mut fz);
//! ```

mod backend;
mod kvectors;
mod parameters;

use core::f64::consts::PI;
use kvectors::KVectors;

use crate::permittivity::ConstantPermittivity;

/// Policy for reciprocal-space Ewald summation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EwaldPolicy {
    /// Standard periodic boundary conditions (default).
    #[default]
    PBC,
    /// Isotropic periodic boundary conditions ([Stenqvist & Lund, 2018](https://doi.org/10/css8)).
    IPBC,
}

/// Reciprocal-space Ewald summation for Coulomb and Yukawa potentials.
///
/// Computes $U_\text{recip}$, $U_\text{self}$, and $U_\text{surface}$
/// for both unscreened Coulomb ($\kappa = 0$) and screened Yukawa ($\kappa = 1/\lambda_D$).
///
/// Positions are passed as separate `x`, `y`, `z` slices (SoA layout), and forces
/// are accumulated into caller-provided buffers. The input type is generic over
/// `T: Into<f64> + Copy`, accepting both `f32` and `f64` slices at zero cost.
///
/// Supports both standard PBC (half-shell k-vectors) and IPBC (first-octant
/// k-vectors, [Stenqvist & Lund, 2018](https://doi.org/10/css8)) via [`EwaldPolicy`].
pub struct EwaldReciprocal {
    kvecs: KVectors,
    /// Stored for `with_policy()` rebuild and `surface_energy()`.
    box_length: [f64; 3],
    volume: f64,
    /// Retained for `self_energy()` which needs the raw splitting parameter.
    alpha: f64,
    /// Retained for `self_energy()` which needs the raw screening parameter.
    kappa: f64,
    /// Integer k-space cutoff, stored for `with_policy()` rebuild.
    n_max: u32,
    boundary: ConstantPermittivity,
    policy: EwaldPolicy,
    /// PBC structure factors: Re(S(k)) and Im(S(k)) for each k-vector.
    /// Empty when policy is IPBC (empty Vec costs nothing — just a dangling pointer).
    sk_re: Vec<f64>,
    sk_im: Vec<f64>,
    /// IPBC structure factors: all 8 cos/sin combinations per k-vector [CCC..SSS].
    /// All 8 are needed because cos(k·r_ij) expands into a sum over products of
    /// per-axis cos/sin terms, and only the full set factorizes into Σ Q²_αβγ form
    /// suitable for O(N·N_k) evaluation. Empty when policy is PBC.
    sk_ipbc: Vec<[f64; 8]>,
}

impl EwaldReciprocal {
    /// Construct with automatic parameter selection for target accuracy.
    ///
    /// Returns `(ewald, real_cutoff)` where `real_cutoff` is the recommended
    /// real-space pair cutoff (Å) to pair with this reciprocal-space instance.
    /// Parameters are chosen via Kolafa–Perram estimates:
    /// $r_c = L_\text{min}/2$,
    /// $\alpha = \sqrt{-\ln\varepsilon}/r_c$,
    /// $n_\text{max} = \lceil \alpha L_\text{max} \sqrt{-\ln\varepsilon}/\pi \rceil$.
    ///
    /// - `box_length`: Side lengths `[Lx, Ly, Lz]` of the orthorhombic simulation box (Å)
    /// - `accuracy`: Target relative energy accuracy $\varepsilon$ (e.g. `1e-5`)
    /// - `kappa`: Inverse Debye screening length (`None` or `Some(0.0)` for Coulomb, `Some(κ)` for Yukawa).
    ///   The optimal $\alpha$ is independent of $\kappa$
    ///   ([Salin & Caillol, 2000](https://doi.org/10.1063/1.1326477)).
    ///
    /// Uses tinfoil (metallic) boundary conditions and standard PBC by default.
    /// Customize with [`Self::with_policy`] and [`Self::set_boundary`].
    ///
    /// # Example
    /// ```
    /// use coulomb::reciprocal::EwaldReciprocal;
    /// let (mut ewald, r_cutoff) = EwaldReciprocal::new([10.0; 3], 1e-5, None);
    /// assert!(r_cutoff > 0.0);
    /// assert!(ewald.num_k_vectors() > 0);
    /// ```
    pub fn new(box_length: [f64; 3], accuracy: f64, kappa: Option<f64>) -> (Self, f64) {
        let params = parameters::optimize_ewald_parameters(&box_length, accuracy);
        let kappa = kappa.unwrap_or(0.0);
        let ewald = Self::build(
            box_length,
            params.alpha,
            kappa,
            params.n_max,
            EwaldPolicy::PBC,
        );
        (ewald, params.real_cutoff)
    }

    /// Construct with explicit Ewald parameters.
    ///
    /// Use this when you know the exact $\alpha$ and $n_\text{max}$ values you want
    /// (e.g. for reproducing literature results or benchmarking).
    ///
    /// - `box_length`: Side lengths $[L_x, L_y, L_z]$ of the orthorhombic simulation box (Å)
    /// - `alpha`: Ewald splitting parameter $\alpha$ (Å⁻¹)
    /// - `kappa`: Inverse Debye screening length $\kappa$ (0 for Coulomb, >0 for Yukawa)
    /// - `n_max`: Spherical cutoff in integer k-space $n_\text{max}$;
    ///   k-vectors satisfy $n_x^2 + n_y^2 + n_z^2 \leq n_\text{max}^2$
    ///   and map to $\mathbf{k} = 2\pi(n_x/L_x, n_y/L_y, n_z/L_z)$.
    ///
    /// # Example
    /// ```
    /// use coulomb::reciprocal::EwaldReciprocal;
    /// let ewald = EwaldReciprocal::with_parameters([10.0; 3], 0.8, 0.0, 11);
    /// assert_eq!(ewald.num_k_vectors(), 2787);
    /// ```
    pub fn with_parameters(box_length: [f64; 3], alpha: f64, kappa: f64, n_max: u32) -> Self {
        Self::build(box_length, alpha, kappa, n_max, EwaldPolicy::PBC)
    }

    /// Switch to a different Ewald policy (PBC or IPBC).
    ///
    /// Rebuilds k-vectors and structure factor storage. Call before `update_all`.
    pub fn with_policy(self, policy: EwaldPolicy) -> Self {
        if policy == self.policy {
            return self;
        }
        Self::build(self.box_length, self.alpha, self.kappa, self.n_max, policy)
    }

    /// Set the boundary permittivity.
    ///
    /// Only affects [`Self::surface_energy`] and [`Self::add_surface_forces`].
    /// Default is tinfoil ($\varepsilon_s = \infty$), which makes the surface term vanish.
    pub fn set_boundary(&mut self, boundary: ConstantPermittivity) {
        self.boundary = boundary;
    }

    /// Internal constructor shared by all public constructors.
    fn build(
        box_length: [f64; 3],
        alpha: f64,
        kappa: f64,
        n_max: u32,
        policy: EwaldPolicy,
    ) -> Self {
        // Parameter validation is handled by KVectors::new() via debug_assert.
        let volume = box_length[0] * box_length[1] * box_length[2];
        let kvecs = KVectors::new(box_length, n_max, alpha, kappa, policy);
        let n_k = kvecs.len();
        let (sk_re, sk_im, sk_ipbc) = match policy {
            EwaldPolicy::PBC => (vec![0.0; n_k], vec![0.0; n_k], Vec::new()),
            EwaldPolicy::IPBC => (Vec::new(), Vec::new(), vec![[0.0; 8]; n_k]),
        };
        Self {
            kvecs,
            box_length,
            volume,
            alpha,
            kappa,
            n_max,
            boundary: crate::permittivity::METAL,
            policy,
            sk_re,
            sk_im,
            sk_ipbc,
        }
    }

    /// Recompute all structure factors from scratch. $O(N \times N_k)$.
    pub fn update_all<T: Into<f64> + Copy>(
        &mut self,
        x: &[T],
        y: &[T],
        z: &[T],
        charges: &[T],
    ) {
        match self.policy {
            EwaldPolicy::PBC => {
                self.sk_re.fill(0.0);
                self.sk_im.fill(0.0);
                for ((&charge, &px), (&py, &pz)) in
                    charges.iter().zip(x.iter()).zip(y.iter().zip(z.iter()))
                {
                    update_all_pbc(
                        &self.kvecs,
                        px.into(),
                        py.into(),
                        pz.into(),
                        charge.into(),
                        &mut self.sk_re,
                        &mut self.sk_im,
                    );
                }
            }
            EwaldPolicy::IPBC => {
                self.sk_ipbc.fill([0.0; 8]);
                for ((&charge, &px), (&py, &pz)) in
                    charges.iter().zip(x.iter()).zip(y.iter().zip(z.iter()))
                {
                    backend::scalar::update_all_ipbc(
                        &self.kvecs,
                        px.into(),
                        py.into(),
                        pz.into(),
                        charge.into(),
                        &mut self.sk_ipbc,
                    );
                }
            }
        }
    }

    /// Commit a single-particle move. $O(N_k)$.
    pub fn update_particle<T: Into<f64> + Copy>(
        &mut self,
        charge: T,
        old: [T; 3],
        new: [T; 3],
    ) {
        let charge = charge.into();
        let old = [old[0].into(), old[1].into(), old[2].into()];
        let new = [new[0].into(), new[1].into(), new[2].into()];
        match self.policy {
            EwaldPolicy::PBC => {
                update_particle_pbc(
                    &self.kvecs,
                    charge,
                    old,
                    new,
                    &mut self.sk_re,
                    &mut self.sk_im,
                );
            }
            EwaldPolicy::IPBC => {
                backend::scalar::update_particle_ipbc(
                    &self.kvecs,
                    charge,
                    old,
                    new,
                    &mut self.sk_ipbc,
                );
            }
        }
    }

    /// Reciprocal-space energy. $O(N_k)$.
    ///
    /// $$U_\text{recip} = \frac{2\pi}{V} \sum_{\mathbf{k}} A(\mathbf{k}) \, |S(\mathbf{k})|^2$$
    pub fn energy(&self) -> f64 {
        let prefactor = 2.0 * PI / self.volume;
        let sum: f64 = match self.policy {
            EwaldPolicy::PBC => energy_pbc(&self.kvecs, &self.sk_re, &self.sk_im),
            EwaldPolicy::IPBC => backend::scalar::energy_ipbc(&self.kvecs, &self.sk_ipbc),
        };
        prefactor * sum
    }

    /// Energy change for a trial particle move, without mutating state. $O(N_k)$.
    ///
    /// Uses $|S+\Delta S|^2 - |S|^2 = 2S \cdot \Delta S + |\Delta S|^2$ to avoid
    /// a full recomputation of structure factors, making MC trial moves $O(N_k)$ not $O(N \cdot N_k)$.
    ///
    /// $$\Delta U = \frac{2\pi}{V} \sum_{\mathbf{k}} A(\mathbf{k}) \bigl[2\,S(\mathbf{k}) \cdot \Delta S(\mathbf{k}) + |\Delta S(\mathbf{k})|^2\bigr]$$
    pub fn energy_change<T: Into<f64> + Copy>(
        &self,
        charge: T,
        old: [T; 3],
        new: [T; 3],
    ) -> f64 {
        let charge = charge.into();
        let old = [old[0].into(), old[1].into(), old[2].into()];
        let new = [new[0].into(), new[1].into(), new[2].into()];
        let prefactor = 2.0 * PI / self.volume;
        let sum = match self.policy {
            EwaldPolicy::PBC => {
                energy_change_pbc(&self.kvecs, &self.sk_re, &self.sk_im, charge, old, new)
            }
            EwaldPolicy::IPBC => {
                backend::scalar::energy_change_ipbc(&self.kvecs, &self.sk_ipbc, charge, old, new)
            }
        };
        prefactor * sum
    }

    /// Force on one particle. $O(N_k)$. Returns `[fx, fy, fz]`.
    ///
    /// PBC: $\mathbf{F}_i = \frac{4\pi q_i}{V} \sum_{\mathbf{k}} A(\mathbf{k}) \bigl[\sin(\mathbf{k} \cdot \mathbf{r}_i) S_\text{re} - \cos(\mathbf{k} \cdot \mathbf{r}_i) S_\text{im}\bigr] \mathbf{k}$
    pub fn force<T: Into<f64> + Copy>(&self, pos: [T; 3], charge: T) -> [f64; 3] {
        let charge = charge.into();
        let pos = [pos[0].into(), pos[1].into(), pos[2].into()];
        let prefactor = 4.0 * PI * charge / self.volume;
        let [fx, fy, fz] = match self.policy {
            EwaldPolicy::PBC => force_pbc(&self.kvecs, pos, &self.sk_re, &self.sk_im),
            EwaldPolicy::IPBC => backend::scalar::force_ipbc(&self.kvecs, pos, &self.sk_ipbc),
        };
        [prefactor * fx, prefactor * fy, prefactor * fz]
    }

    /// Accumulate reciprocal-space forces on all particles. $O(N \times N_k)$.
    ///
    /// Adds (+=) force contributions into the provided `fx`, `fy`, `fz` buffers.
    /// Caller is responsible for zeroing the buffers if starting fresh.
    #[allow(clippy::too_many_arguments)]
    pub fn add_forces<T: Into<f64> + Copy>(
        &self,
        x: &[T],
        y: &[T],
        z: &[T],
        charges: &[T],
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

    /// Self-energy correction (position-independent). $O(N)$.
    ///
    /// $$U_\text{self} = \left[-\frac{\alpha}{\sqrt\pi} e^{-\kappa^2/4\alpha^2}
    ///   + \frac{\kappa}{2} \operatorname{erfc}\!\left(\frac{\kappa}{2\alpha}\right)\right] \sum_i q_i^2$$
    ///
    /// For Coulomb ($\kappa = 0$): $U_\text{self} = -(\alpha/\sqrt\pi) \sum_i q_i^2$.
    pub fn self_energy<T: Into<f64> + Copy>(&self, charges: &[T]) -> f64 {
        let sum_q2: f64 = charges.iter().map(|&q| { let q: f64 = q.into(); q * q }).sum();
        let alpha = self.alpha;
        let sqrt_pi = PI.sqrt();

        // Yukawa self-energy has an extra erfc term from the screened potential;
        // for Coulomb (κ=0) this reduces to the standard -α/√π form.
        if self.kappa > 0.0 {
            let kappa = self.kappa;
            let exp_term = (-kappa * kappa / (4.0 * alpha * alpha)).exp();
            let erfc_term = crate::math::erfc_x(kappa / (2.0 * alpha));
            (-alpha / sqrt_pi * exp_term + kappa / 2.0 * erfc_term) * sum_q2
        } else {
            -alpha / sqrt_pi * sum_q2
        }
    }

    /// Surface energy for non-tinfoil boundaries. $O(N)$.
    ///
    /// $$U_\text{surface} = \frac{2\pi}{(2\varepsilon_s + 1)\,V} \left|\sum_i q_i \mathbf{r}_i\right|^2$$
    ///
    /// Returns 0 for tinfoil ($\varepsilon_s = \infty$) boundary conditions.
    pub fn surface_energy<T: Into<f64> + Copy>(
        &self,
        x: &[T],
        y: &[T],
        z: &[T],
        charges: &[T],
    ) -> f64 {
        // Tinfoil (ε_s = ∞) makes the surface term vanish; finite ε_s corrects
        // for the depolarization field at the boundary of the Ewald sphere.
        let eps_s: f64 = self.boundary.into();
        if eps_s.is_infinite() {
            return 0.0;
        }
        let [dx, dy, dz] = dipole_moment(x, y, z, charges);
        2.0 * PI / ((2.0 * eps_s + 1.0) * self.volume) * (dx * dx + dy * dy + dz * dz)
    }

    /// Accumulate surface forces on all particles. $O(N)$.
    ///
    /// Adds (+=) force contributions into the provided `fx`, `fy`, `fz` buffers.
    /// Does nothing for tinfoil boundary conditions.
    #[allow(clippy::too_many_arguments)]
    pub fn add_surface_forces<T: Into<f64> + Copy>(
        &self,
        x: &[T],
        y: &[T],
        z: &[T],
        charges: &[T],
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
            let c: f64 = base * q.into();
            *out_fx += c * dx;
            *out_fy += c * dy;
            *out_fz += c * dz;
        }
    }

    /// Number of k-vectors (for diagnostics).
    pub fn num_k_vectors(&self) -> usize {
        self.kvecs.len()
    }

    /// The splitting parameter $\alpha$ (for diagnostics/pairing with real-space Ewald).
    pub fn alpha(&self) -> f64 {
        self.alpha
    }
}

// ── Backend dispatch: SIMD on x86_64, scalar elsewhere ──────────────────────

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn update_all_pbc(
    kvecs: &KVectors,
    px: f64,
    py: f64,
    pz: f64,
    charge: f64,
    sk_re: &mut [f64],
    sk_im: &mut [f64],
) {
    backend::simd::update_all_pbc(kvecs, px, py, pz, charge, sk_re, sk_im);
}

#[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
fn update_all_pbc(
    kvecs: &KVectors,
    px: f64,
    py: f64,
    pz: f64,
    charge: f64,
    sk_re: &mut [f64],
    sk_im: &mut [f64],
) {
    backend::scalar::update_all_pbc(kvecs, px, py, pz, charge, sk_re, sk_im);
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn energy_pbc(kvecs: &KVectors, sk_re: &[f64], sk_im: &[f64]) -> f64 {
    backend::simd::energy_pbc(kvecs, sk_re, sk_im)
}

#[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
fn energy_pbc(kvecs: &KVectors, sk_re: &[f64], sk_im: &[f64]) -> f64 {
    backend::scalar::energy_pbc(kvecs, sk_re, sk_im)
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn energy_change_pbc(
    kvecs: &KVectors,
    sk_re: &[f64],
    sk_im: &[f64],
    charge: f64,
    old: [f64; 3],
    new: [f64; 3],
) -> f64 {
    backend::simd::energy_change_pbc(kvecs, sk_re, sk_im, charge, old, new)
}

#[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
fn energy_change_pbc(
    kvecs: &KVectors,
    sk_re: &[f64],
    sk_im: &[f64],
    charge: f64,
    old: [f64; 3],
    new: [f64; 3],
) -> f64 {
    backend::scalar::energy_change_pbc(kvecs, sk_re, sk_im, charge, old, new)
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn update_particle_pbc(
    kvecs: &KVectors,
    charge: f64,
    old: [f64; 3],
    new: [f64; 3],
    sk_re: &mut [f64],
    sk_im: &mut [f64],
) {
    backend::simd::update_particle_pbc(kvecs, charge, old, new, sk_re, sk_im);
}

#[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
fn update_particle_pbc(
    kvecs: &KVectors,
    charge: f64,
    old: [f64; 3],
    new: [f64; 3],
    sk_re: &mut [f64],
    sk_im: &mut [f64],
) {
    backend::scalar::update_particle_pbc(kvecs, charge, old, new, sk_re, sk_im);
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn force_pbc(kvecs: &KVectors, pos: [f64; 3], sk_re: &[f64], sk_im: &[f64]) -> [f64; 3] {
    backend::simd::force_pbc(kvecs, pos, sk_re, sk_im)
}

#[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
fn force_pbc(kvecs: &KVectors, pos: [f64; 3], sk_re: &[f64], sk_im: &[f64]) -> [f64; 3] {
    backend::scalar::force_pbc(kvecs, pos, sk_re, sk_im)
}

/// Calculate the dipole moment: Σ q_i r_i, returned as [dx, dy, dz].
fn dipole_moment<T: Into<f64> + Copy>(x: &[T], y: &[T], z: &[T], charges: &[T]) -> [f64; 3] {
    charges
        .iter()
        .zip(x.iter())
        .zip(y.iter().zip(z.iter()))
        .fold([0.0; 3], |[dx, dy, dz], ((&q, &px), (&py, &pz))| {
            let (q, px, py, pz) = (q.into(), px.into(), py.into(), pz.into());
            [dx + q * px, dy + q * py, dz + q * pz]
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Reference values from C++ coulombgalore/test/unittests.cpp
    // Setup: box=10³, n_max=11, α=0.8, κ=0
    // Two charges [+1, -1] at [(-0.5,0,0), (0.5,0,0)]

    fn setup_two_charges() -> (EwaldReciprocal, [f64; 2], [f64; 2], [f64; 2], [f64; 2]) {
        let mut ewald = EwaldReciprocal::with_parameters([10.0; 3], 0.8, 0.0, 11);
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
        let ewald = EwaldReciprocal::with_parameters([10.0; 3], 0.8, 0.0, 11);
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
        let mut ewald = EwaldReciprocal::with_parameters([10.0; 3], 0.8, 0.0, 11);
        ewald.set_boundary(crate::permittivity::VACUUM);
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
        let mut ewald = EwaldReciprocal::with_parameters([10.0; 3], 0.8, 0.0, 11);
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
        assert_relative_eq!(fx, numerical_fx, epsilon = 1e-3);
    }

    #[test]
    fn test_partial_update_equals_full() {
        let mut ewald = EwaldReciprocal::with_parameters([10.0; 3], 0.8, 0.0, 5);
        let (x, y, z) = (
            [1.0, -1.0, 2.0, -2.0],
            [2.0, 0.5, -1.0, 1.0],
            [3.0, -0.5, 1.5, -1.0],
        );
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

        assert_relative_eq!(e_partial, e_full, epsilon = 1e-6);
    }

    #[test]
    fn test_energy_change_particle() {
        let mut ewald = EwaldReciprocal::with_parameters([10.0; 3], 0.8, 0.0, 5);
        let (x, y, z) = (
            [1.0, -1.0, 2.0, -2.0],
            [2.0, 0.5, -1.0, 1.0],
            [3.0, -0.5, 1.5, -1.0],
        );
        let charges = [1.0, -1.0, 0.5, -0.5];

        ewald.update_all(&x, &y, &z, &charges);
        let e_before = ewald.energy();

        let old = [x[0], y[0], z[0]];
        let new = [1.5, 2.5, 3.5];
        let de = ewald.energy_change(charges[0], old, new);

        ewald.update_particle(charges[0], old, new);
        let e_after = ewald.energy();

        assert_relative_eq!(de, e_after - e_before, epsilon = 1e-6);
    }

    #[test]
    fn test_yukawa_reduces_to_coulomb() {
        let (x, y, z) = ([-0.5, 0.5], [0.0, 0.0], [0.0, 0.0]);
        let charges = [1.0, -1.0];

        let mut ewald_c = EwaldReciprocal::with_parameters([10.0; 3], 0.8, 0.0, 11);
        ewald_c.update_all(&x, &y, &z, &charges);

        let mut ewald_y = EwaldReciprocal::with_parameters([10.0; 3], 0.8, 1e-10, 11);
        ewald_y.update_all(&x, &y, &z, &charges);

        assert_relative_eq!(ewald_c.energy(), ewald_y.energy(), epsilon = 1e-6);
    }

    #[test]
    fn test_yukawa_energy_decays_with_kappa() {
        let (x, y, z) = ([-0.5, 0.5], [0.0, 0.0], [0.0, 0.0]);
        let charges = [1.0, -1.0];

        let mut energies = Vec::new();
        for &kappa in &[0.0, 0.1, 0.5, 1.0] {
            let mut ewald = EwaldReciprocal::with_parameters([10.0; 3], 0.8, kappa, 11);
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
        let ewald = EwaldReciprocal::with_parameters([10.0; 3], alpha, kappa, 11);
        let charges = [1.0, -1.0];
        let self_e = ewald.self_energy(&charges);
        let sum_q2: f64 = charges.iter().map(|q| q * q).sum();
        let expected = (-alpha / PI.sqrt() * (-kappa * kappa / (4.0 * alpha * alpha)).exp()
            + kappa / 2.0 * crate::math::erfc_x(kappa / (2.0 * alpha)))
            * sum_q2;
        assert_relative_eq!(self_e, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_automatic_parameters() {
        let (ewald, rc) = EwaldReciprocal::new([10.0; 3], 1e-5, None);
        assert!(ewald.alpha() > 0.0);
        assert_relative_eq!(rc, 5.0); // L_min / 2
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
        let mut ewald = EwaldReciprocal::with_parameters([10.0; 3], 0.8, 0.0, 11);
        ewald.set_boundary(crate::permittivity::VACUUM);
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
        let ewald = EwaldReciprocal::with_parameters([10.0; 3], 0.8, 0.0, 11);
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

        let na_basis: [[f64; 3]; 4] = [[0.0, 0.0, 0.0], [a, a, 0.0], [a, 0.0, a], [0.0, a, a]];
        let cl_basis: [[f64; 3]; 4] = [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a], [a, a, a]];

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
        let mut ewald = EwaldReciprocal::with_parameters([l_box; 3], alpha, 0.0, k_max);
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

    // --- IPBC tests (reference: faunus/src/energy.cpp, https://doi.org/10/css8) ---

    #[test]
    fn test_ipbc_kvector_count() {
        let ewald = EwaldReciprocal::with_parameters([10.0; 3], 0.894427, 0.0, 11)
            .with_policy(EwaldPolicy::IPBC);
        assert_eq!(ewald.num_k_vectors(), 846);
    }

    #[test]
    fn test_ipbc_fewer_kvectors_than_pbc() {
        let pbc = EwaldReciprocal::with_parameters([10.0; 3], 0.8, 0.0, 11);
        let ipbc = EwaldReciprocal::with_parameters([10.0; 3], 0.8, 0.0, 11)
            .with_policy(EwaldPolicy::IPBC);
        assert!(ipbc.num_k_vectors() < pbc.num_k_vectors());
    }

    #[test]
    fn test_ipbc_reciprocal_energy_matches_pbc_for_axis_config() {
        // For charges along one axis, IPBC 8-component Σ Q² equals PBC |S|².
        // With octant factors matching half-shell factors, energies should agree.
        let alpha = 0.894427;
        let (x, y, z) = ([-0.5, 0.5], [0.0, 0.0], [0.0, 0.0]);
        let charges = [1.0, -1.0];

        let mut pbc = EwaldReciprocal::with_parameters([10.0; 3], alpha, 0.0, 11);
        pbc.update_all(&x, &y, &z, &charges);

        let mut ipbc = EwaldReciprocal::with_parameters([10.0; 3], alpha, 0.0, 11)
            .with_policy(EwaldPolicy::IPBC);
        ipbc.update_all(&x, &y, &z, &charges);

        assert_relative_eq!(ipbc.energy(), pbc.energy(), epsilon = 1e-6);
    }

    #[test]
    fn test_ipbc_self_energy_matches_pbc() {
        // Self energy is policy-independent
        let charges = [1.0, -1.0];
        let pbc = EwaldReciprocal::with_parameters([10.0; 3], 0.894427, 0.0, 11);
        let ipbc = EwaldReciprocal::with_parameters([10.0; 3], 0.894427, 0.0, 11)
            .with_policy(EwaldPolicy::IPBC);
        assert_relative_eq!(
            pbc.self_energy(&charges),
            ipbc.self_energy(&charges),
            epsilon = 1e-12,
        );
        // faunus reference value
        assert_relative_eq!(
            ipbc.self_energy(&charges),
            -1.0092530088080642,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_ipbc_surface_energy() {
        // Surface energy is also policy-independent
        let mut ewald = EwaldReciprocal::with_parameters([10.0; 3], 0.894427, 0.0, 11)
            .with_policy(EwaldPolicy::IPBC);
        ewald.set_boundary(crate::permittivity::VACUUM);
        let (x, y, z) = ([-0.5, 0.5], [0.0, 0.0], [0.0, 0.0]);
        let charges = [1.0, -1.0];
        ewald.update_all(&x, &y, &z, &charges);
        assert_relative_eq!(
            ewald.surface_energy(&x, &y, &z, &charges),
            0.0020943951023931952,
            epsilon = 1e-6,
        );
    }

    #[test]
    fn test_ipbc_partial_update_equals_full() {
        let mut ewald =
            EwaldReciprocal::with_parameters([10.0; 3], 0.8, 0.0, 5).with_policy(EwaldPolicy::IPBC);
        let (x, y, z) = (
            [1.0, -1.0, 2.0, -2.0],
            [2.0, 0.5, -1.0, 1.0],
            [3.0, -0.5, 1.5, -1.0],
        );
        let charges = [1.0, -1.0, 0.5, -0.5];

        ewald.update_all(&x, &y, &z, &charges);

        let old = [x[0], y[0], z[0]];
        let new_pos = [1.5, 2.5, 3.5];
        ewald.update_particle(charges[0], old, new_pos);
        let e_partial = ewald.energy();

        let x2 = [1.5, -1.0, 2.0, -2.0];
        let y2 = [2.5, 0.5, -1.0, 1.0];
        let z2 = [3.5, -0.5, 1.5, -1.0];
        ewald.update_all(&x2, &y2, &z2, &charges);
        let e_full = ewald.energy();

        assert_relative_eq!(e_partial, e_full, epsilon = 1e-6);
    }

    #[test]
    fn test_ipbc_energy_change_consistency() {
        let mut ewald =
            EwaldReciprocal::with_parameters([10.0; 3], 0.8, 0.0, 5).with_policy(EwaldPolicy::IPBC);
        let (x, y, z) = (
            [1.0, -1.0, 2.0, -2.0],
            [2.0, 0.5, -1.0, 1.0],
            [3.0, -0.5, 1.5, -1.0],
        );
        let charges = [1.0, -1.0, 0.5, -0.5];

        ewald.update_all(&x, &y, &z, &charges);
        let e_before = ewald.energy();

        let old = [x[0], y[0], z[0]];
        let new_pos = [1.5, 2.5, 3.5];
        let de = ewald.energy_change(charges[0], old, new_pos);

        ewald.update_particle(charges[0], old, new_pos);
        let e_after = ewald.energy();

        assert_relative_eq!(de, e_after - e_before, epsilon = 1e-6);
    }

    #[test]
    fn test_ipbc_force_numerical_gradient() {
        let mut ewald = EwaldReciprocal::with_parameters([10.0; 3], 0.8, 0.0, 11)
            .with_policy(EwaldPolicy::IPBC);
        let (x, y, z) = ([-0.5, 0.5], [0.1, -0.1], [0.2, -0.2]);
        let charges = [1.0, -1.0];
        ewald.update_all(&x, &y, &z, &charges);

        let [fx, fy, fz] = ewald.force([x[0], y[0], z[0]], charges[0]);

        let delta = 1e-5;
        // Numerical gradient for x
        ewald.update_all(&[-0.5 + delta, 0.5], &y, &z, &charges);
        let e_plus = ewald.energy();
        ewald.update_all(&[-0.5 - delta, 0.5], &y, &z, &charges);
        let e_minus = ewald.energy();
        assert_relative_eq!(fx, -(e_plus - e_minus) / (2.0 * delta), epsilon = 1e-3);

        // Numerical gradient for y
        ewald.update_all(&x, &[0.1 + delta, -0.1], &z, &charges);
        let e_plus = ewald.energy();
        ewald.update_all(&x, &[0.1 - delta, -0.1], &z, &charges);
        let e_minus = ewald.energy();
        assert_relative_eq!(fy, -(e_plus - e_minus) / (2.0 * delta), epsilon = 1e-3);

        // Numerical gradient for z
        ewald.update_all(&x, &y, &[0.2 + delta, -0.2], &charges);
        let e_plus = ewald.energy();
        ewald.update_all(&x, &y, &[0.2 - delta, -0.2], &charges);
        let e_minus = ewald.energy();
        assert_relative_eq!(fz, -(e_plus - e_minus) / (2.0 * delta), epsilon = 1e-3);
    }

    #[test]
    fn test_f32_input() {
        let mut ewald = EwaldReciprocal::with_parameters([10.0; 3], 0.8, 0.0, 5);
        let x: [f32; 2] = [-0.5, 0.5];
        let y: [f32; 2] = [0.0, 0.0];
        let z: [f32; 2] = [0.0, 0.0];
        let charges: [f32; 2] = [1.0, -1.0];
        ewald.update_all(&x, &y, &z, &charges);
        let e = ewald.energy();
        assert!(e.abs() > 0.0);

        let de = ewald.energy_change(1.0_f32, [-0.5_f32, 0.0, 0.0], [-0.3_f32, 0.0, 0.0]);
        assert!(de.abs() > 0.0);

        let [fx, _, _] = ewald.force([-0.5_f32, 0.0, 0.0], 1.0_f32);
        assert!(fx.abs() > 0.0);

        let self_e = ewald.self_energy(&charges);
        assert!(self_e < 0.0);
    }

    #[test]
    fn test_ipbc_forces_sum_to_zero() {
        let mut ewald = EwaldReciprocal::with_parameters([10.0; 3], 0.8, 0.0, 11)
            .with_policy(EwaldPolicy::IPBC);
        let (x, y, z) = ([-0.5, 0.5], [0.1, -0.1], [0.2, -0.2]);
        let charges = [1.0, -1.0];
        ewald.update_all(&x, &y, &z, &charges);
        let mut fx = [0.0; 2];
        let mut fy = [0.0; 2];
        let mut fz = [0.0; 2];
        ewald.add_forces(&x, &y, &z, &charges, &mut fx, &mut fy, &mut fz);
        assert_relative_eq!(fx.iter().sum::<f64>(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(fy.iter().sum::<f64>(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(fz.iter().sum::<f64>(), 0.0, epsilon = 1e-10);
    }
}
