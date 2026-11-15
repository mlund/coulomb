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
//! Provides [`EwaldReciprocal`] for k-space energies, forces, and efficient
//! Monte Carlo partial updates. Supports both unscreened Coulomb ($\kappa=0$)
//! and screened Yukawa ($\kappa>0$), with PBC and IPBC
//! ([Stenqvist & Lund, 2018](https://doi.org/10/css8)) policies.
//!
//! # Usage
//!
//! ```
//! # use coulomb::reciprocal::*;
//! let mut ewald = EwaldReciprocal::new([10.0; 3], 5.0, 1e-5, None);
//! # let (x, y, z) = (vec![-0.5, 0.5], vec![0.0, 0.0], vec![0.0, 0.0]);
//! # let charges = vec![1.0, -1.0];
//! ewald.update_all(&x, &y, &z, &charges, None, false);
//!
//! // MC: O(N_k) trial move
//! # let (old, new) = ([-0.5, 0.0, 0.0], [-0.3, 0.0, 0.0]);
//! let de = ewald.energy_change(1.0, old, new);
//! // if accept { ewald.update_particle(1.0, old, new); }
//!
//! // MD: accumulate forces
//! let (mut fx, mut fy, mut fz) = (vec![0.0; 2], vec![0.0; 2], vec![0.0; 2]);
//! ewald.add_forces(&x, &y, &z, &charges, &mut fx, &mut fy, &mut fz);
//! ```
//!
//! Parameters α and $n_\text{max}$ are derived automatically from the cutoff and
//! accuracy target.
//! For Yukawa, passing `optimize = true` to [`EwaldReciprocal::update_all`] can
//! further reduce k-vectors by jointly scanning (α, $n_\text{max}$) against
//! actual particle data (PBC only; no-op for Coulomb and IPBC).

mod backend;
mod kvectors;
pub(crate) mod parameters;

use core::f64::consts::PI;
use kvectors::KVectors;

/// √π — precomputed to avoid repeated runtime `PI.sqrt()` calls.
const SQRT_PI: f64 = 1.7724538509055159;

use crate::permittivity::ConstantPermittivity;

/// Policy for reciprocal-space Ewald summation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EwaldPolicy {
    /// Standard periodic boundary conditions (default).
    #[default]
    PBC,
    /// Isotropic periodic boundary conditions ([Stenqvist & Lund, 2018](https://doi.org/10/css8)).
    ///
    /// Uses cos-product structure factors and first-octant k-vectors.
    /// Note: the IPBC energy is **not α-invariant**, so `optimize = true` in
    /// [`EwaldReciprocal::update_all`] is a no-op for this policy.
    IPBC,
}

/// How α and n_max follow a box or policy change.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Splitting {
    /// Re-derived from `accuracy`, as at construction.
    Analytic,
    /// Optimized or user-set α: the real-space part keeps it, so α and the k-space cutoff
    /// stay fixed and only n_max follows the box.
    Fixed { k_cut: f64 },
}

/// Reciprocal-space Ewald summation for Coulomb and Yukawa potentials.
///
/// Handles $U_\text{recip}$, $U_\text{self}$, and $U_\text{surface}$ with
/// SoA position layout and generic `T: Into<f64> + Copy` input (f32/f64).
/// Supports PBC and IPBC via [`EwaldPolicy`].
#[derive(Clone)]
pub struct EwaldReciprocal {
    kvecs: KVectors,
    box_length: [f64; 3],
    two_pi_over_v: f64,
    four_pi_over_v: f64,
    alpha: f64,
    kappa: f64,
    n_max: u32,
    splitting: Splitting,
    real_cutoff: f64,
    accuracy: f64,
    boundary: ConstantPermittivity,
    policy: EwaldPolicy,
    sk_re: Vec<f64>,
    sk_im: Vec<f64>,
    sk_ipbc: Vec<f64>,
}

impl EwaldReciprocal {
    /// Construct with automatic parameter selection for target accuracy.
    ///
    /// Pass `debye_length: None` for Coulomb, `Some(λ_D)` for Yukawa.
    /// Defaults to tinfoil boundaries and PBC; customize with
    /// [`Self::set_policy`] and [`Self::set_boundary`].
    pub fn new(
        box_length: [f64; 3],
        real_cutoff: f64,
        accuracy: f64,
        debye_length: Option<f64>,
    ) -> Self {
        Self::build(
            box_length,
            real_cutoff,
            accuracy,
            debye_length.map_or(0.0, |d| 1.0 / d),
            EwaldPolicy::PBC,
        )
    }

    /// Construct with a fixed splitting parameter α; $n_\text{max}$ follows from `accuracy`.
    ///
    /// For when α comes from elsewhere, e.g. a fitted model; the real-space part must use
    /// the same α, see [`Self::real_space_scheme`].
    pub fn with_alpha(
        box_length: [f64; 3],
        real_cutoff: f64,
        accuracy: f64,
        debye_length: Option<f64>,
        alpha: f64,
    ) -> Self {
        let kappa = debye_length.map_or(0.0, |d| 1.0 / d);
        let n_max = parameters::ewald_n_max(&box_length, alpha, accuracy, kappa);
        let mut ewald = Self::build_explicit(
            box_length,
            real_cutoff,
            accuracy,
            kappa,
            alpha,
            n_max,
            EwaldPolicy::PBC,
        );
        ewald.splitting = Splitting::Fixed {
            k_cut: parameters::ewald_k_cut(alpha, accuracy, kappa),
        };
        ewald
    }

    /// Replace `self` by `next`, keeping the boundary (build_explicit defaults to tinfoil).
    fn adopt(&mut self, mut next: Self, splitting: Splitting) {
        next.boundary = self.boundary;
        next.splitting = splitting;
        *self = next;
    }

    /// Rebuild k-vectors for a new box or policy, following [`Splitting`].
    fn rebuild(&mut self, box_length: [f64; 3], policy: EwaldPolicy) {
        let next = match self.splitting {
            Splitting::Analytic => Self::build(
                box_length,
                self.real_cutoff,
                self.accuracy,
                self.kappa,
                policy,
            ),
            Splitting::Fixed { k_cut } => Self::build_explicit(
                box_length,
                self.real_cutoff,
                self.accuracy,
                self.kappa,
                self.alpha,
                // Tolerance: a virtual-volume perturbation must not add a shell
                parameters::n_max_from_k_cut(k_cut, &box_length, 1e-6),
                policy,
            ),
        };
        self.adopt(next, self.splitting);
    }

    /// Jointly optimize α and n_max to minimize the number of k-vectors.
    ///
    /// Called internally by `update_all` when `optimize = true`, κ > 0, and PBC.
    /// Leaves structure factors populated after optimization.
    fn optimize<T>(&mut self, x: &[T], y: &[T], z: &[T], charges: &[T])
    where
        T: Into<f64> + Copy,
    {
        let alpha_max = parameters::ewald_alpha(self.real_cutoff, self.accuracy, self.kappa);
        let n_max_analytic =
            parameters::ewald_n_max(&self.box_length, alpha_max, self.accuracy, self.kappa);

        // Yukawa screening strong enough to skip reciprocal sum entirely
        if n_max_analytic == 0 {
            *self = Self::build_explicit(
                self.box_length,
                self.real_cutoff,
                self.accuracy,
                self.kappa,
                alpha_max,
                0,
                self.policy,
            );
            return;
        }

        let charges_f64: Vec<f64> = charges.iter().map(|&q| q.into()).collect();
        let sum_q2: f64 = charges_f64.iter().map(|q| q * q).sum();
        let threshold = self.accuracy * sum_q2;

        // Reference energy at analytic (α, n_max) — guaranteed accurate
        let mut ref_ewald = Self::build_explicit(
            self.box_length,
            self.real_cutoff,
            self.accuracy,
            self.kappa,
            alpha_max,
            n_max_analytic,
            self.policy,
        );
        ref_ewald.update_all(x, y, z, charges, None, false);
        let e_recip_ref = ref_ewald.energy();
        let e_self_ref = ref_ewald.self_energy(&charges_f64);
        let x_f64: Vec<f64> = x.iter().map(|&v| v.into()).collect();
        let y_f64: Vec<f64> = y.iter().map(|&v| v.into()).collect();
        let z_f64: Vec<f64> = z.iter().map(|&v| v.into()).collect();
        let e_real_ref = real_space_energy(
            &ref_ewald.real_space_scheme(),
            &x_f64,
            &y_f64,
            &z_f64,
            &charges_f64,
            &self.box_length,
        );
        let e_total_ref = e_real_ref + e_recip_ref + e_self_ref;

        let mut best = ref_ewald;
        let mut best_kvecs = best.num_k_vectors();

        // Scan η = α·r_c downward from analytic maximum.
        // Step size Δη = 0.2 gives ~5–10 candidates for typical η_max ≈ 2–3.
        // Floor η_min = 0.8 is a hard safety net; in practice the sf0 ≤ 0.1 guard
        // (erfc(0.8) ≈ 0.26) breaks the loop well before this floor is reached.
        let eta_max = alpha_max * self.real_cutoff;
        let eta_step = 0.2_f64;
        let eta_min = 0.8_f64;
        let debye_length = self.debye_length();

        // Precompute charge sums for inline self-energy (avoids allocating per α)
        let q_tot: f64 = charges_f64.iter().sum();
        let two_pi_over_v = self.two_pi_over_v;
        let kappa = self.kappa;
        let mut eta = eta_max - eta_step;
        while eta >= eta_min {
            let alpha = eta / self.real_cutoff;

            // Guard: sf0(1) ≤ 0.1 via existing ShortRangeFunction implementation
            use crate::pairwise::ShortRangeFunction;
            let real_scheme =
                crate::pairwise::RealSpaceEwald::from_alpha(self.real_cutoff, alpha, debye_length);
            if real_scheme.short_range_f0(1.0) > 0.1 {
                break; // real-space truncation too severe
            }

            // Early exit: if even n_max=1 can't beat best, skip this α
            let n_max_upper =
                parameters::ewald_n_max(&self.box_length, alpha, self.accuracy, self.kappa);
            let kvecs_at_nmax1 = KVectors::new(self.box_length, 1, alpha, kappa, self.policy).len();
            if kvecs_at_nmax1 >= best_kvecs {
                eta -= eta_step;
                continue;
            }

            // Compute real-space energy at this α (O(N²), called once per α)
            let e_real = real_space_energy(
                &real_scheme,
                &x_f64,
                &y_f64,
                &z_f64,
                &charges_f64,
                &self.box_length,
            );

            // Self-energy + net-charge correction at this α.
            // Inlined from self_energy() to avoid allocating k-vectors just for a scalar.
            // Must match the formulas in self_energy() exactly — see its doc comments.
            let e_self = if kappa > 0.0 {
                // Yukawa: self-energy + finite k=0 term
                let exp_term = (-kappa * kappa / (4.0 * alpha * alpha)).exp();
                let erfc_term = crate::math::erfc_x(kappa / (2.0 * alpha));
                let e_s = (-alpha / SQRT_PI).mul_add(exp_term, kappa / 2.0 * erfc_term) * sum_q2;
                let e_k0 = two_pi_over_v * exp_term / (kappa * kappa) * q_tot * q_tot;
                e_s + e_k0
            } else {
                // Coulomb: self-energy + neutralizing background
                let e_s = -alpha / SQRT_PI * sum_q2;
                let volume = 2.0 * PI / two_pi_over_v;
                let e_bg = -PI / (2.0 * volume * alpha * alpha) * q_tot * q_tot;
                e_s + e_bg
            };

            // Sweep n_max from 1 upward, find smallest where total energy converges
            let mut n_max = 1_u32;
            while n_max <= n_max_upper {
                let mut tmp = Self::build_explicit(
                    self.box_length,
                    self.real_cutoff,
                    self.accuracy,
                    self.kappa,
                    alpha,
                    n_max,
                    self.policy,
                );
                tmp.update_all(x, y, z, charges, None, false);
                let e_recip = tmp.energy();
                let e_total = e_real + e_recip + e_self;

                if (e_total - e_total_ref).abs() < threshold && tmp.num_k_vectors() < best_kvecs {
                    best_kvecs = tmp.num_k_vectors();
                    best = tmp;
                    break;
                }
                // Linear steps for small n_max (where each step adds many k-vectors
                // relative to total), then double to avoid O(n_max) build_explicit calls.
                n_max = if n_max < 4 {
                    n_max + 1
                } else {
                    (n_max * 2).min(n_max_upper + 1)
                };
            }

            eta -= eta_step;
        }

        let k_cut = 2.0 * PI * f64::from(best.n_max) / parameters::longest_side(&best.box_length);
        self.adopt(best, Splitting::Fixed { k_cut });
    }

    /// Switch to a different Ewald policy (PBC or IPBC).
    ///
    /// Rebuilds k-vectors and structure factor storage. Call before `update_all`.
    pub fn set_policy(&mut self, policy: EwaldPolicy) {
        if policy == self.policy {
            return;
        }
        self.rebuild(self.box_length, policy);
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
        real_cutoff: f64,
        accuracy: f64,
        kappa: f64,
        policy: EwaldPolicy,
    ) -> Self {
        let alpha = parameters::ewald_alpha(real_cutoff, accuracy, kappa);
        let n_max = parameters::ewald_n_max(&box_length, alpha, accuracy, kappa);
        Self::build_explicit(
            box_length,
            real_cutoff,
            accuracy,
            kappa,
            alpha,
            n_max,
            policy,
        )
    }

    /// Constructor with explicit α and n_max (bypasses automatic parameter selection).
    pub(crate) fn build_explicit(
        box_length: [f64; 3],
        real_cutoff: f64,
        accuracy: f64,
        kappa: f64,
        alpha: f64,
        n_max: u32,
        policy: EwaldPolicy,
    ) -> Self {
        // Parameter validation is handled by KVectors::new() via debug_assert.
        let volume = box_length[0] * box_length[1] * box_length[2];
        let two_pi_over_v = 2.0 * PI / volume;
        let four_pi_over_v = 4.0 * PI / volume;
        let kvecs = KVectors::new(box_length, n_max, alpha, kappa, policy);
        let n_k = kvecs.len();
        let (sk_re, sk_im, sk_ipbc) = match policy {
            EwaldPolicy::PBC => (vec![0.0; n_k], vec![0.0; n_k], Vec::new()),
            EwaldPolicy::IPBC => (Vec::new(), Vec::new(), vec![0.0; n_k]),
        };
        Self {
            kvecs,
            box_length,
            two_pi_over_v,
            four_pi_over_v,
            alpha,
            kappa,
            n_max,
            splitting: Splitting::Analytic,
            real_cutoff,
            accuracy,
            boundary: crate::permittivity::METAL,
            policy,
            sk_re,
            sk_im,
            sk_ipbc,
        }
    }

    /// Recompute all structure factors from scratch. $O(N \times N_k)$.
    ///
    /// If `box_length` is `Some`, rebuilds the k-vectors first: default parameters are
    /// re-derived from `accuracy`, while an optimized or user-set α is kept along with its
    /// k-space cutoff.
    /// If `optimize` is true and κ > 0 (Yukawa), jointly optimizes α and $n_\text{max}$
    /// to minimize the number of k-vectors while preserving energy accuracy.
    /// No-op for pure Coulomb (κ = 0) or IPBC policy.
    pub fn update_all<T: Into<f64> + Copy>(
        &mut self,
        x: &[T],
        y: &[T],
        z: &[T],
        charges: &[T],
        box_length: Option<[f64; 3]>,
        optimize: bool,
    ) {
        if let Some(box_length) = box_length {
            self.rebuild(box_length, self.policy);
        }
        if optimize && self.kappa > 0.0 && self.policy == EwaldPolicy::PBC {
            self.optimize(x, y, z, charges);
            return;
        }
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
                self.sk_ipbc.fill(0.0);
                for ((&charge, &px), (&py, &pz)) in
                    charges.iter().zip(x.iter()).zip(y.iter().zip(z.iter()))
                {
                    update_all_ipbc(
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
    pub fn update_particle<T: Into<f64> + Copy>(&mut self, charge: T, old: [T; 3], new: [T; 3]) {
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
                update_particle_ipbc(&self.kvecs, charge, old, new, &mut self.sk_ipbc);
            }
        }
    }

    /// Reciprocal-space energy. $O(N_k)$.
    ///
    /// $$U_\text{recip} = \frac{2\pi}{V} \sum_{\mathbf{k}} A(\mathbf{k}) \, |S(\mathbf{k})|^2$$
    pub fn energy(&self) -> f64 {
        let sum: f64 = match self.policy {
            EwaldPolicy::PBC => energy_pbc(&self.kvecs, &self.sk_re, &self.sk_im),
            EwaldPolicy::IPBC => energy_ipbc(&self.kvecs, &self.sk_ipbc),
        };
        self.two_pi_over_v * sum
    }

    /// Energy change for a trial particle move, without mutating state. $O(N_k)$.
    pub fn energy_change<T: Into<f64> + Copy>(&self, charge: T, old: [T; 3], new: [T; 3]) -> f64 {
        let charge = charge.into();
        let old = [old[0].into(), old[1].into(), old[2].into()];
        let new = [new[0].into(), new[1].into(), new[2].into()];
        let sum = match self.policy {
            EwaldPolicy::PBC => {
                energy_change_pbc(&self.kvecs, &self.sk_re, &self.sk_im, charge, old, new)
            }
            EwaldPolicy::IPBC => energy_change_ipbc(&self.kvecs, &self.sk_ipbc, charge, old, new),
        };
        self.two_pi_over_v * sum
    }

    /// Force on one particle. $O(N_k)$. Returns `[fx, fy, fz]`.
    pub fn force<T: Into<f64> + Copy>(&self, pos: [T; 3], charge: T) -> [f64; 3] {
        let charge = charge.into();
        let pos = [pos[0].into(), pos[1].into(), pos[2].into()];
        let prefactor = self.four_pi_over_v * charge;
        let [fx, fy, fz] = match self.policy {
            EwaldPolicy::PBC => force_pbc(&self.kvecs, pos, &self.sk_re, &self.sk_im),
            EwaldPolicy::IPBC => force_ipbc(&self.kvecs, pos, &self.sk_ipbc),
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

    /// Self-energy, $k=0$, and neutralizing background corrections. $O(N)$.
    ///
    /// Corrects for spurious self-interaction in the real-space sum and, for
    /// non-neutral systems, the excluded $k=0$ term (Yukawa) or neutralizing
    /// background (Coulomb). All terms vanish for charge-neutral systems.
    pub fn self_energy<T: Into<f64> + Copy>(&self, charges: &[T]) -> f64 {
        let mut sum_q2 = 0.0_f64;
        let mut q_tot = 0.0_f64;
        for &q in charges {
            let q: f64 = q.into();
            sum_q2 += q * q;
            q_tot += q;
        }
        let alpha = self.alpha;

        if self.kappa > 0.0 {
            let kappa = self.kappa;
            let exp_term = (-kappa * kappa / (4.0 * alpha * alpha)).exp();
            let erfc_term = crate::math::erfc_x(kappa / (2.0 * alpha));
            let e_self = (-alpha / SQRT_PI).mul_add(exp_term, kappa / 2.0 * erfc_term) * sum_q2;
            let e_k0 = self.two_pi_over_v * exp_term / (kappa * kappa) * q_tot * q_tot;
            e_self + e_k0
        } else {
            let e_self = -alpha / SQRT_PI * sum_q2;
            // Neutralizing background correction for non-neutral Coulomb systems.
            // U_bg = -π Q² / (2 V α²), see e.g. Frenkel & Smit, Eq. 12.1.25.
            let volume = 2.0 * PI / self.two_pi_over_v;
            let e_bg = -PI / (2.0 * volume * alpha * alpha) * q_tot * q_tot;
            e_self + e_bg
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
        let eps_s: f64 = self.boundary.into();
        if eps_s.is_infinite() {
            return 0.0;
        }
        let [dx, dy, dz] = dipole_moment(x, y, z, charges);
        self.two_pi_over_v / 2.0f64.mul_add(eps_s, 1.0) * dz.mul_add(dz, dx.mul_add(dx, dy * dy))
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
        let base = -self.four_pi_over_v / 2.0f64.mul_add(eps_s, 1.0);
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
    pub const fn alpha(&self) -> f64 {
        self.alpha
    }

    /// The integer k-space cutoff $n_\text{max}$.
    pub const fn n_max(&self) -> u32 {
        self.n_max
    }

    /// The real-space pair cutoff paired with this reciprocal instance.
    pub const fn real_cutoff(&self) -> f64 {
        self.real_cutoff
    }

    /// Debye screening length, or `None` for Coulomb.
    pub fn debye_length(&self) -> Option<f64> {
        if self.kappa > 0.0 {
            Some(1.0 / self.kappa)
        } else {
            None
        }
    }

    /// Create the matching real-space Ewald pair potential.
    ///
    /// The returned [`RealSpaceEwald`](crate::pairwise::RealSpaceEwald) uses the same
    /// splitting parameter $\alpha$, screening $\kappa$, and real-space cutoff
    /// as this reciprocal instance.
    pub fn real_space_scheme(&self) -> crate::pairwise::RealSpaceEwald {
        crate::pairwise::RealSpaceEwald::from_alpha(
            self.real_cutoff,
            self.alpha,
            self.debye_length(),
        )
    }
}

// ── Backend dispatch: SIMD on x86_64, scalar elsewhere ──────────────────────

/// Generate a cfg-gated dispatch function that forwards to `backend::simd::$name`
/// on x86_64 with the `simd` feature, and `backend::scalar::$name` otherwise.
macro_rules! dispatch {
    ($name:ident ( $($arg:ident : $ty:ty),* $(,)? ) -> $ret:ty) => {
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        fn $name($($arg: $ty),*) -> $ret {
            backend::simd::$name($($arg),*)
        }
        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
        fn $name($($arg: $ty),*) -> $ret {
            backend::scalar::$name($($arg),*)
        }
    };
}

// PBC dispatch
dispatch!(update_all_pbc(kvecs: &KVectors, px: f64, py: f64, pz: f64, charge: f64, sk_re: &mut [f64], sk_im: &mut [f64]) -> ());
dispatch!(energy_pbc(kvecs: &KVectors, sk_re: &[f64], sk_im: &[f64]) -> f64);
dispatch!(energy_change_pbc(kvecs: &KVectors, sk_re: &[f64], sk_im: &[f64], charge: f64, old: [f64; 3], new: [f64; 3]) -> f64);
dispatch!(update_particle_pbc(kvecs: &KVectors, charge: f64, old: [f64; 3], new: [f64; 3], sk_re: &mut [f64], sk_im: &mut [f64]) -> ());
dispatch!(force_pbc(kvecs: &KVectors, pos: [f64; 3], sk_re: &[f64], sk_im: &[f64]) -> [f64; 3]);

// IPBC dispatch
dispatch!(update_all_ipbc(kvecs: &KVectors, px: f64, py: f64, pz: f64, charge: f64, sk_ipbc: &mut [f64]) -> ());
dispatch!(energy_ipbc(kvecs: &KVectors, sk_ipbc: &[f64]) -> f64);
dispatch!(energy_change_ipbc(kvecs: &KVectors, sk_ipbc: &[f64], charge: f64, old: [f64; 3], new: [f64; 3]) -> f64);
dispatch!(update_particle_ipbc(kvecs: &KVectors, charge: f64, old: [f64; 3], new: [f64; 3], sk_ipbc: &mut [f64]) -> ());
dispatch!(force_ipbc(kvecs: &KVectors, pos: [f64; 3], sk_ipbc: &[f64]) -> [f64; 3]);

/// Minimum-image all-to-all real-space energy (O(N²)).
fn real_space_energy(
    scheme: &crate::pairwise::RealSpaceEwald,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    charges: &[f64],
    box_length: &[f64; 3],
) -> f64 {
    use crate::cutoff::Cutoff;
    use crate::pairwise::MultipoleEnergy;
    let n = x.len();
    let cutoff = scheme.cutoff();
    let mut energy = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let mut dx = x[i] - x[j];
            let mut dy = y[i] - y[j];
            let mut dz = z[i] - z[j];
            dx -= box_length[0] * (dx / box_length[0]).round();
            dy -= box_length[1] * (dy / box_length[1]).round();
            dz -= box_length[2] * (dz / box_length[2]).round();
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r < cutoff {
                energy += scheme.ion_ion_energy(charges[i], charges[j], r);
            }
        }
    }
    energy
}

/// Calculate the dipole moment: Σ q_i r_i, returned as [dx, dy, dz].
fn dipole_moment<T: Into<f64> + Copy>(x: &[T], y: &[T], z: &[T], charges: &[T]) -> [f64; 3] {
    charges
        .iter()
        .zip(x.iter())
        .zip(y.iter().zip(z.iter()))
        .fold([0.0; 3], |[dx, dy, dz], ((&q, &px), (&py, &pz))| {
            let (q, px, py, pz) = (q.into(), px.into(), py.into(), pz.into());
            [q.mul_add(px, dx), q.mul_add(py, dy), q.mul_add(pz, dz)]
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
        let mut ewald = EwaldReciprocal::new([10.0; 3], 5.0, 1e-5, None);
        let x = [-0.5, 0.5];
        let y = [0.0, 0.0];
        let z = [0.0, 0.0];
        let charges = [1.0, -1.0];
        ewald.update_all(&x, &y, &z, &charges, None, false);
        (ewald, x, y, z, charges)
    }

    #[test]
    fn test_reciprocal_energy_two_charges() {
        let (ewald, _, _, _, _) = setup_two_charges();
        // Opposite charges: reciprocal energy should be positive (unfavorable k-space contribution)
        assert!(ewald.energy() > 0.0);
    }

    #[test]
    fn test_reciprocal_force_two_charges() {
        let (ewald, x, y, z, charges) = setup_two_charges();
        let [fx, fy, fz] = ewald.force([x[0], y[0], z[0]], charges[0]);
        // Force along x-axis (particles separated along x), zero in y/z
        assert!(fx > 0.0);
        assert_relative_eq!(fy, 0.0, epsilon = 1e-6);
        assert_relative_eq!(fz, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_self_energy_coulomb() {
        let ewald = EwaldReciprocal::new([10.0; 3], 5.0, 1e-5, None);
        let alpha = ewald.alpha();
        let expected = -alpha / SQRT_PI * 2.0; // sum(q^2) = 2
        assert_relative_eq!(ewald.self_energy(&[1.0, -1.0]), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_surface_energy_tinfoil() {
        let (ewald, x, y, z, charges) = setup_two_charges();
        assert_relative_eq!(ewald.surface_energy(&x, &y, &z, &charges), 0.0);
    }

    #[test]
    fn test_surface_energy_vacuum() {
        let mut ewald = EwaldReciprocal::new([10.0; 3], 5.0, 1e-5, None);
        ewald.set_boundary(crate::permittivity::VACUUM);
        let (x, y, z) = ([-0.5, 0.5], [0.0, 0.0], [0.0, 0.0]);
        let charges = [1.0, -1.0];
        ewald.update_all(&x, &y, &z, &charges, None, false);
        assert_relative_eq!(
            ewald.surface_energy(&x, &y, &z, &charges),
            0.0020943951,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_forces_sum_to_zero() {
        for policy in [EwaldPolicy::PBC, EwaldPolicy::IPBC] {
            let mut ewald = EwaldReciprocal::new([10.0; 3], 5.0, 1e-5, None);
            ewald.set_policy(policy);
            let (x, y, z) = ([-0.5, 0.5], [0.1, -0.1], [0.2, -0.2]);
            let charges = [1.0, -1.0];
            ewald.update_all(&x, &y, &z, &charges, None, false);
            let mut fx = [0.0; 2];
            let mut fy = [0.0; 2];
            let mut fz = [0.0; 2];
            ewald.add_forces(&x, &y, &z, &charges, &mut fx, &mut fy, &mut fz);
            assert_relative_eq!(fx.iter().sum::<f64>(), 0.0, epsilon = 1e-10);
            assert_relative_eq!(fy.iter().sum::<f64>(), 0.0, epsilon = 1e-10);
            assert_relative_eq!(fz.iter().sum::<f64>(), 0.0, epsilon = 1e-10);
        }
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
        for policy in [EwaldPolicy::PBC, EwaldPolicy::IPBC] {
            let mut ewald = EwaldReciprocal::new([10.0; 3], 5.0, 1e-5, None);
            ewald.set_policy(policy);
            let (x, y, z) = ([-0.5, 0.5], [0.1, -0.1], [0.2, -0.2]);
            let charges = [1.0, -1.0];
            ewald.update_all(&x, &y, &z, &charges, None, false);
            let [fx, fy, fz] = ewald.force([x[0], y[0], z[0]], charges[0]);
            let delta = 1e-5;

            ewald.update_all(&[-0.5 + delta, 0.5], &y, &z, &charges, None, false);
            let ep = ewald.energy();
            ewald.update_all(&[-0.5 - delta, 0.5], &y, &z, &charges, None, false);
            let em = ewald.energy();
            assert_relative_eq!(fx, -(ep - em) / (2.0 * delta), epsilon = 1e-3);

            ewald.update_all(&x, &[0.1 + delta, -0.1], &z, &charges, None, false);
            let ep = ewald.energy();
            ewald.update_all(&x, &[0.1 - delta, -0.1], &z, &charges, None, false);
            let em = ewald.energy();
            assert_relative_eq!(fy, -(ep - em) / (2.0 * delta), epsilon = 1e-3);

            ewald.update_all(&x, &y, &[0.2 + delta, -0.2], &charges, None, false);
            let ep = ewald.energy();
            ewald.update_all(&x, &y, &[0.2 - delta, -0.2], &charges, None, false);
            let em = ewald.energy();
            assert_relative_eq!(fz, -(ep - em) / (2.0 * delta), epsilon = 1e-3);
        }
    }

    #[test]
    fn test_partial_update_equals_full() {
        for policy in [EwaldPolicy::PBC, EwaldPolicy::IPBC] {
            let mut ewald = EwaldReciprocal::new([10.0; 3], 5.0, 1e-5, None);
            ewald.set_policy(policy);
            let (x, y, z) = (
                [1.0, -1.0, 2.0, -2.0],
                [2.0, 0.5, -1.0, 1.0],
                [3.0, -0.5, 1.5, -1.0],
            );
            let charges = [1.0, -1.0, 0.5, -0.5];
            ewald.update_all(&x, &y, &z, &charges, None, false);

            let old = [x[0], y[0], z[0]];
            let new_pos = [1.5, 2.5, 3.5];
            ewald.update_particle(charges[0], old, new_pos);
            let e_partial = ewald.energy();

            let x2 = [1.5, -1.0, 2.0, -2.0];
            let y2 = [2.5, 0.5, -1.0, 1.0];
            let z2 = [3.5, -0.5, 1.5, -1.0];
            ewald.update_all(&x2, &y2, &z2, &charges, None, false);
            let e_full = ewald.energy();

            assert_relative_eq!(e_partial, e_full, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_energy_change_particle() {
        for policy in [EwaldPolicy::PBC, EwaldPolicy::IPBC] {
            let mut ewald = EwaldReciprocal::new([10.0; 3], 5.0, 1e-5, None);
            ewald.set_policy(policy);
            let (x, y, z) = (
                [1.0, -1.0, 2.0, -2.0],
                [2.0, 0.5, -1.0, 1.0],
                [3.0, -0.5, 1.5, -1.0],
            );
            let charges = [1.0, -1.0, 0.5, -0.5];
            ewald.update_all(&x, &y, &z, &charges, None, false);
            let e_before = ewald.energy();

            let old = [x[0], y[0], z[0]];
            let new_pos = [1.5, 2.5, 3.5];
            let de = ewald.energy_change(charges[0], old, new_pos);
            ewald.update_particle(charges[0], old, new_pos);
            let e_after = ewald.energy();

            assert_relative_eq!(de, e_after - e_before, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_yukawa_reduces_to_coulomb() {
        let (x, y, z) = ([-0.5, 0.5], [0.0, 0.0], [0.0, 0.0]);
        let charges = [1.0, -1.0];

        let mut ewald_c = EwaldReciprocal::new([10.0; 3], 5.0, 1e-5, None);
        ewald_c.update_all(&x, &y, &z, &charges, None, false);

        // Very large Debye length ≈ Coulomb
        let mut ewald_y = EwaldReciprocal::new([10.0; 3], 5.0, 1e-5, Some(1e10));
        ewald_y.update_all(&x, &y, &z, &charges, None, false);

        assert_relative_eq!(ewald_c.energy(), ewald_y.energy(), epsilon = 1e-6);
    }

    #[test]
    fn test_yukawa_energy_decays_with_screening() {
        let (x, y, z) = ([-0.5, 0.5], [0.0, 0.0], [0.0, 0.0]);
        let charges = [1.0, -1.0];

        // Decreasing Debye length = increasing screening = decreasing energy
        let mut energies = Vec::new();
        for &debye in &[1e10, 10.0, 2.0, 1.0] {
            let mut ewald = EwaldReciprocal::new([10.0; 3], 5.0, 1e-5, Some(debye));
            ewald.update_all(&x, &y, &z, &charges, None, false);
            energies.push(ewald.energy().abs());
        }
        for i in 1..energies.len() {
            assert!(energies[i] < energies[i - 1]);
        }
    }

    #[test]
    fn test_self_energy_yukawa() {
        let debye_length = 2.0; // kappa = 0.5
        let kappa = 1.0 / debye_length;
        let ewald = EwaldReciprocal::new([10.0; 3], 5.0, 1e-5, Some(debye_length));
        let alpha = ewald.alpha();
        let charges = [1.0, -1.0];
        let self_e = ewald.self_energy(&charges);
        let sum_q2: f64 = charges.iter().map(|q| q * q).sum();
        let expected = (-alpha / SQRT_PI * (-kappa * kappa / (4.0 * alpha * alpha)).exp()
            + kappa / 2.0 * crate::math::erfc_x(kappa / (2.0 * alpha)))
            * sum_q2;
        assert_relative_eq!(self_e, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_automatic_parameters() {
        let ewald = EwaldReciprocal::new([10.0; 3], 5.0, 1e-5, None);
        assert!(ewald.alpha() > 0.0);
        assert_relative_eq!(ewald.real_cutoff(), 5.0); // L_min / 2
        assert!(ewald.num_k_vectors() > 0);
    }

    #[test]
    fn test_total_energy_two_charges() {
        let (ewald, x, y, z, charges) = setup_two_charges();
        let recip = ewald.energy();
        let self_e = ewald.self_energy(&charges);
        let surface = ewald.surface_energy(&x, &y, &z, &charges);

        assert!(recip > 0.0);
        assert!(self_e < 0.0);
        assert_relative_eq!(surface, 0.0);
        // The reciprocal + self sum is alpha-dependent but should be finite and negative
        let partial = recip + self_e + surface;
        assert!(partial < 0.0);
    }

    #[test]
    fn test_add_surface_forces() {
        let mut ewald = EwaldReciprocal::new([10.0; 3], 5.0, 1e-5, None);
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
        let ewald = EwaldReciprocal::new([10.0; 3], 5.0, 1e-5, None);
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

        let r_c = l_box / 2.0 - 0.01;

        // Reciprocal space
        let mut ewald = EwaldReciprocal::new([l_box; 3], r_c, 1e-8, None);
        ewald.update_all(&px, &py, &pz, &charges, None, false);
        let e_recip = ewald.energy();
        let e_self = ewald.self_energy(&charges);

        // Real space: minimum-image pairwise sum
        let e_real = real_space_energy(
            &ewald.real_space_scheme(),
            &px,
            &py,
            &pz,
            &charges,
            &[l_box; 3],
        );

        let e_total = e_real + e_recip + e_self;
        let n_pairs = px.len() / 2;
        let madelung = -e_total * a / n_pairs as f64;

        // NaCl Madelung constant: 1.7475645946 (Wikipedia)
        assert_relative_eq!(madelung, 1.7475645946, epsilon = 1e-4);
    }

    // --- IPBC tests (reference: faunus/src/energy.cpp, https://doi.org/10/css8) ---

    #[test]
    fn test_ipbc_fewer_kvectors_than_pbc() {
        let pbc = EwaldReciprocal::new([10.0; 3], 5.0, 1e-5, None);
        let mut ipbc = EwaldReciprocal::new([10.0; 3], 5.0, 1e-5, None);
        ipbc.set_policy(EwaldPolicy::IPBC);
        assert!(ipbc.num_k_vectors() > 0);
        assert!(ipbc.num_k_vectors() < pbc.num_k_vectors());
    }

    #[test]
    fn test_ipbc_cos_symmetry_antisymmetric_charges() {
        // For charges ±1 at (±a, 0, 0), the cos-product structure factor
        // Q_k = cos(kx·(-a))·1·1 + (-1)·cos(kx·a)·1·1 = 0 because cos is even.
        // So the IPBC reciprocal energy should be exactly zero.
        let mut ipbc = EwaldReciprocal::new([10.0; 3], 5.0, 1e-5, None);
        ipbc.set_policy(EwaldPolicy::IPBC);
        ipbc.update_all(
            &[-0.5, 0.5],
            &[0.0, 0.0],
            &[0.0, 0.0],
            &[1.0, -1.0],
            None,
            false,
        );
        assert_relative_eq!(ipbc.energy(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_ipbc_self_energy_matches_pbc() {
        let charges = [1.0, -1.0];
        let pbc = EwaldReciprocal::new([10.0; 3], 5.0, 1e-5, None);
        let mut ipbc = EwaldReciprocal::new([10.0; 3], 5.0, 1e-5, None);
        ipbc.set_policy(EwaldPolicy::IPBC);
        assert_relative_eq!(
            pbc.self_energy(&charges),
            ipbc.self_energy(&charges),
            epsilon = 1e-12,
        );
    }

    #[test]
    fn test_f32_input() {
        let mut ewald = EwaldReciprocal::new([10.0; 3], 5.0, 1e-5, None);
        let x: [f32; 2] = [-0.5, 0.5];
        let y: [f32; 2] = [0.0, 0.0];
        let z: [f32; 2] = [0.0, 0.0];
        let charges: [f32; 2] = [1.0, -1.0];
        ewald.update_all(&x, &y, &z, &charges, None, false);
        let e = ewald.energy();
        assert!(e.abs() > 0.0);

        let de = ewald.energy_change(1.0_f32, [-0.5_f32, 0.0, 0.0], [-0.3_f32, 0.0, 0.0]);
        assert!(de.abs() > 0.0);

        let [fx, _, _] = ewald.force([-0.5_f32, 0.0, 0.0], 1.0_f32);
        assert!(fx.abs() > 0.0);

        let self_e = ewald.self_energy(&charges);
        assert!(self_e < 0.0);
    }

    // ── Optimizer tests ─────────────────────────────────────────────────────

    /// CPPM P18 particle sites (subset: first 8 for fast tests).
    const CPPM_SITES: &[([f64; 3], f64)] = &[
        ([10.51, 17.01, 0.0], 1.0),
        ([-17.01, 0.0, -10.51], -1.0),
        ([6.18, 16.18, 10.0], 1.0),
        ([0.0, 20.0, 0.0], 1.0),
        ([6.18, 16.18, -10.0], 1.0),
        ([16.18, 10.0, 6.18], 1.0),
        ([-10.0, -6.18, 16.18], -1.0),
        ([16.18, 10.0, -6.18], 1.0),
    ];

    const R_PARTICLE: f64 = 20.0;

    /// Build a simple cubic lattice of CPPM particles (no rotations for portability).
    fn build_cppm_lattice(l_box: f64, n_grid: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let spacing = l_box / n_grid as f64;
        let n_total = n_grid.pow(3) * CPPM_SITES.len();
        let (mut px, mut py, mut pz, mut charges) = (
            Vec::with_capacity(n_total),
            Vec::with_capacity(n_total),
            Vec::with_capacity(n_total),
            Vec::with_capacity(n_total),
        );
        for ix in 0..n_grid {
            for iy in 0..n_grid {
                for iz in 0..n_grid {
                    let cx = (ix as f64 + 0.5) * spacing;
                    let cy = (iy as f64 + 0.5) * spacing;
                    let cz = (iz as f64 + 0.5) * spacing;
                    for &([sx, sy, sz], q) in CPPM_SITES {
                        let mut x = cx + sx;
                        let mut y = cy + sy;
                        let mut z = cz + sz;
                        x -= l_box * (x / l_box).floor();
                        y -= l_box * (y / l_box).floor();
                        z -= l_box * (z / l_box).floor();
                        px.push(x);
                        py.push(y);
                        pz.push(z);
                        charges.push(q);
                    }
                }
            }
        }
        (px, py, pz, charges)
    }

    fn l_box_from_phi(phi: f64, n_particles: usize) -> f64 {
        let v_particle = 4.0 / 3.0 * core::f64::consts::PI * R_PARTICLE.powi(3);
        (n_particles as f64 * v_particle / phi).cbrt()
    }

    /// Total Ewald energy (real + reciprocal + self) for a given configuration.
    fn total_ewald_energy(ewald: &EwaldReciprocal, charges: &[f64]) -> f64 {
        ewald.energy() + ewald.self_energy(charges)
    }

    /// Verify that optimize preserves energy and reduces (or keeps) k-vector count.
    macro_rules! test_optimize {
        ($name:ident, phi = $phi:expr, debye = $debye:expr, expect_kvecs = $expect:expr) => {
            #[test]
            fn $name() {
                let n_grid = 2_usize;
                let n_particles = n_grid.pow(3);
                let l_box = l_box_from_phi($phi, n_particles);
                let r_c = l_box / 2.0 - 0.01;
                let debye_length: Option<f64> = $debye;
                let accuracy = 1e-3;

                let (px, py, pz, charges) = build_cppm_lattice(l_box, n_grid);

                // Analytic parameters (no optimization)
                let mut ewald_an = EwaldReciprocal::new([l_box; 3], r_c, accuracy, debye_length);
                ewald_an.update_all(&px, &py, &pz, &charges, None, false);
                let kvecs_an = ewald_an.num_k_vectors();

                // Real-space energy (shared across both; α is the same for analytic)
                let real_scheme_an = ewald_an.real_space_scheme();
                let e_real_an = real_space_energy(
                    &real_scheme_an, &px, &py, &pz, &charges, &[l_box; 3],
                );
                let e_total_an = e_real_an + total_ewald_energy(&ewald_an, &charges);

                // Optimized
                let mut ewald_opt = EwaldReciprocal::new([l_box; 3], r_c, accuracy, debye_length);
                ewald_opt.update_all(&px, &py, &pz, &charges, None, true);
                let kvecs_opt = ewald_opt.num_k_vectors();

                let real_scheme_opt = ewald_opt.real_space_scheme();
                let e_real_opt = real_space_energy(
                    &real_scheme_opt, &px, &py, &pz, &charges, &[l_box; 3],
                );
                let e_total_opt = e_real_opt + total_ewald_energy(&ewald_opt, &charges);

                eprintln!(
                    "phi={}, debye={:?}: analytic n_max={} kvecs={}, optimized n_max={} kvecs={}",
                    $phi, $debye,
                    ewald_an.n_max(), kvecs_an,
                    ewald_opt.n_max(), kvecs_opt,
                );
                assert_eq!(kvecs_opt, $expect,
                    "unexpected optimized k-vector count (phi={}, debye={:?})",
                    $phi, $debye,
                );

                // Total energy must be preserved within threshold
                let sum_q2: f64 = charges.iter().map(|q| q * q).sum();
                let threshold = accuracy * sum_q2;
                assert!(
                    (e_total_opt - e_total_an).abs() < threshold,
                    "energy mismatch: analytic={:.6}, optimized={:.6}, diff={:.2e}, threshold={:.2e}",
                    e_total_an, e_total_opt,
                    (e_total_opt - e_total_an).abs(), threshold,
                );
            }
        };
    }

    // Sweep over volume fractions and Debye lengths
    test_optimize!(
        test_optimize_phi01_debye50,
        phi = 0.10,
        debye = Some(50.0),
        expect_kvecs = 16
    );
    test_optimize!(
        test_optimize_phi01_debye100,
        phi = 0.10,
        debye = Some(100.0),
        expect_kvecs = 16
    );
    test_optimize!(
        test_optimize_phi01_debye300,
        phi = 0.10,
        debye = Some(300.0),
        expect_kvecs = 16
    );
    test_optimize!(
        test_optimize_phi02_debye50,
        phi = 0.20,
        debye = Some(50.0),
        expect_kvecs = 16
    );
    test_optimize!(
        test_optimize_phi02_debye100,
        phi = 0.20,
        debye = Some(100.0),
        expect_kvecs = 16
    );
    test_optimize!(
        test_optimize_phi02_debye300,
        phi = 0.20,
        debye = Some(300.0),
        expect_kvecs = 16
    );

    #[test]
    fn test_optimize_noop_coulomb() {
        let (px, py, pz, charges) = build_cppm_lattice(100.0, 2);
        let r_c = 49.99;
        let mut ewald = EwaldReciprocal::new([100.0; 3], r_c, 1e-3, None);
        let kvecs_before = ewald.num_k_vectors();
        ewald.update_all(&px, &py, &pz, &charges, None, true);
        assert_eq!(
            ewald.num_k_vectors(),
            kvecs_before,
            "optimize should be no-op for Coulomb"
        );
    }

    /// Yukawa CPPM lattice where `optimize` lowers α below the analytic value.
    fn optimized_cppm() -> (EwaldReciprocal, f64, [Vec<f64>; 4]) {
        let l_box = 100.0;
        let (px, py, pz, charges) = build_cppm_lattice(l_box, 2);
        let mut ewald = EwaldReciprocal::new([l_box; 3], l_box / 2.0 - 0.01, 1e-3, Some(30.0));
        let alpha_analytic = ewald.alpha();
        ewald.update_all(&px, &py, &pz, &charges, None, true);
        assert!(
            ewald.alpha() < alpha_analytic,
            "precondition: optimize must lower α"
        );
        (ewald, l_box, [px, py, pz, charges])
    }

    /// The real-space part keeps the optimized α, so a box change must too.
    #[test]
    fn box_rescale_keeps_optimized_alpha() {
        let (mut ewald, l_box, [px, py, pz, q]) = optimized_cppm();
        let alpha = ewald.alpha();
        ewald.update_all(&px, &py, &pz, &q, Some([l_box * 1.01; 3]), false);
        assert_eq!(ewald.alpha(), alpha);
    }

    /// A virtual volume move perturbs L by ~1e-8; changing n_max there would put
    /// k-space truncation noise into ΔU.
    #[test]
    fn tiny_box_perturbation_keeps_optimized_n_max() {
        let (mut ewald, l_box, [px, py, pz, q]) = optimized_cppm();
        let n_max = ewald.n_max();
        ewald.update_all(&px, &py, &pz, &q, Some([l_box * (1.0 + 1e-8); 3]), false);
        assert_eq!(ewald.n_max(), n_max);
    }

    /// NPT applies many small rescalings; the k-space cutoff must not creep upwards.
    #[test]
    fn repeated_rescaling_returns_to_original_n_max() {
        let (mut ewald, l_box, [px, py, pz, q]) = optimized_cppm();
        let n_max = ewald.n_max();
        for _ in 0..50 {
            ewald.update_all(&px, &py, &pz, &q, Some([l_box * 1.003; 3]), false);
            ewald.update_all(&px, &py, &pz, &q, Some([l_box; 3]), false);
        }
        assert_eq!(ewald.n_max(), n_max);
    }

    #[test]
    fn set_policy_keeps_optimized_alpha() {
        let (mut ewald, _, _) = optimized_cppm();
        let alpha = ewald.alpha();
        ewald.set_policy(EwaldPolicy::IPBC);
        assert_eq!(ewald.alpha(), alpha);
    }

    #[test]
    fn with_alpha_sets_alpha_and_analytic_n_max() {
        let (box_length, kappa) = ([100.0; 3], 1.0 / 30.0);
        let ewald = EwaldReciprocal::with_alpha(box_length, 20.0, 1e-3, Some(30.0), 0.05);
        assert_eq!(ewald.alpha(), 0.05);
        assert_eq!(
            ewald.n_max(),
            parameters::ewald_n_max(&box_length, 0.05, 1e-3, kappa)
        );
    }

    /// After rescaling, the energy equals that of a fresh build at the new box.
    #[test]
    fn rescaled_with_alpha_matches_fresh_build() {
        let (px, py, pz, q) = build_cppm_lattice(100.0, 2);
        let mut ewald = EwaldReciprocal::with_alpha([100.0; 3], 20.0, 1e-3, Some(30.0), 0.05);
        ewald.update_all(&px, &py, &pz, &q, Some([100.5; 3]), false);
        let mut fresh = EwaldReciprocal::with_alpha([100.5; 3], 20.0, 1e-3, Some(30.0), 0.05);
        fresh.update_all(&px, &py, &pz, &q, None, false);
        assert_eq!(
            ewald.n_max(),
            fresh.n_max(),
            "precondition: same k-vector shells"
        );
        assert_relative_eq!(ewald.energy(), fresh.energy(), max_relative = 1e-12);
    }

    /// Expanding the box must keep the accuracy target: n_max and α match a fresh build, both
    /// for default parameters (plain Coulomb and Yukawa NPT) and for a fixed α.
    #[test]
    fn rescale_matches_fresh_build() {
        let (px, py, pz, q) = build_cppm_lattice(100.0, 2);
        type Build = fn([f64; 3]) -> EwaldReciprocal;
        let builders: [(&str, Build); 3] = [
            ("coulomb", |b| EwaldReciprocal::new(b, 20.0, 1e-4, None)),
            ("yukawa", |b| {
                EwaldReciprocal::new(b, 20.0, 1e-4, Some(30.0))
            }),
            ("fixed α", |b| {
                EwaldReciprocal::with_alpha(b, 20.0, 1e-3, Some(30.0), 0.05)
            }),
        ];
        for (name, build) in builders {
            for i in 0..=40 {
                let l = 100.0 * (1.0 + 0.0025 * f64::from(i));
                let mut ewald = build([100.0; 3]);
                ewald.update_all(&px, &py, &pz, &q, Some([l; 3]), false);
                let fresh = build([l; 3]);
                assert_eq!(ewald.n_max(), fresh.n_max(), "{name}, L = {l}");
                assert_eq!(ewald.alpha(), fresh.alpha(), "{name}");
            }
        }
    }

    /// Default parameters re-derive n_max when switching policy, as before.
    #[test]
    fn analytic_set_policy_matches_fresh_build() {
        let mut ewald = EwaldReciprocal::new([100.0; 3], 20.0, 1e-4, Some(30.0));
        ewald.set_policy(EwaldPolicy::IPBC);
        let fresh = EwaldReciprocal::build([100.0; 3], 20.0, 1e-4, 1.0 / 30.0, EwaldPolicy::IPBC);
        assert_eq!(ewald.n_max(), fresh.n_max());
        assert_eq!(ewald.alpha(), fresh.alpha());
    }
}
