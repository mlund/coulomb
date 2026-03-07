use core::f64::consts::PI;

use super::EwaldPolicy;

/// K-vector storage (SoA, f32) with precomputed A(k) coefficients.
#[derive(Clone)]
pub(super) struct KVectors {
    pub(super) kx: Vec<f32>,
    pub(super) ky: Vec<f32>,
    pub(super) kz: Vec<f32>,
    pub(super) aks: Vec<f32>,
}

impl KVectors {
    /// Generate k-vectors for the given policy.
    pub fn new(
        box_length: [f64; 3],
        n_max: u32,
        alpha: f64,
        kappa: f64,
        policy: EwaldPolicy,
    ) -> Self {
        debug_assert!(alpha > 0.0, "alpha must be positive");
        debug_assert!(kappa >= 0.0, "kappa must be non-negative");
        debug_assert!(
            box_length.iter().all(|&l| l > 0.0),
            "box lengths must be positive"
        );

        let two_pi_over_l = box_length.map(|l| 2.0 * PI / l);
        let n_max_i = n_max as i32;
        let n_max_sq = n_max_i * n_max_i;
        let four_alpha_sq = 4.0 * alpha * alpha;
        let kappa_sq = kappa * kappa;

        // Estimate capacity from sphere volume to avoid reallocations.
        let n3 = (n_max as usize).pow(3);
        let estimate = match policy {
            EwaldPolicy::PBC => n3 * 2,  // ~half-shell of 4π/3 sphere
            EwaldPolicy::IPBC => n3 / 2, // ~first-octant
        };
        let mut kx = Vec::with_capacity(estimate);
        let mut ky = Vec::with_capacity(estimate);
        let mut kz = Vec::with_capacity(estimate);
        let mut aks = Vec::with_capacity(estimate);

        // PBC uses a half-shell: nx >= 0, with ny/nz going negative only when a
        // higher index is positive. This avoids counting each ±k pair twice.
        // IPBC uses the first octant (nx,ny,nz >= 0) since cos-product basis
        // functions are symmetric under sign flips of individual components.
        for nx in 0..=n_max_i {
            let ny_start = match policy {
                EwaldPolicy::PBC if nx > 0 => -n_max_i,
                _ => 0,
            };
            for ny in ny_start..=n_max_i {
                // Skip k=0 (nz_start=1 when nx==ny==0) since the k=0 term diverges
                // and cancels with the charge-neutrality background.
                let nz_start = match policy {
                    EwaldPolicy::PBC if nx > 0 || ny > 0 => -n_max_i,
                    _ if nx == 0 && ny == 0 => 1,
                    _ => 0,
                };
                for nz in nz_start..=n_max_i {
                    // Spherical cutoff: prefer over cubic because it reduces the number
                    // of k-vectors while maintaining isotropic accuracy.
                    let n_sq = nx * nx + ny * ny + nz * nz;
                    if n_sq > n_max_sq {
                        continue;
                    }
                    let kvx = nx as f64 * two_pi_over_l[0];
                    let kvy = ny as f64 * two_pi_over_l[1];
                    let kvz = nz as f64 * two_pi_over_l[2];
                    // For Yukawa, k² is replaced by k²+κ² in the Gaussian convergence factor.
                    let k_sq_kappa_sq = kvx * kvx + kvy * kvy + kvz * kvz + kappa_sq;
                    // Bake the symmetry multiplier into A(k) so inner loops need no branching.
                    // PBC: factor of 2 accounts for the missing -k half of each pair.
                    // IPBC: 2^(number of nonzero components) accounts for the missing octants,
                    // e.g. (1,0,2) represents 2 octants while (1,2,3) represents 8.
                    let ak = match policy {
                        EwaldPolicy::PBC => {
                            2.0 * (-k_sq_kappa_sq / four_alpha_sq).exp() / k_sq_kappa_sq
                        }
                        EwaldPolicy::IPBC => {
                            let nonzero = (nx > 0) as u32 + (ny > 0) as u32 + (nz > 0) as u32;
                            (1 << nonzero) as f64 * (-k_sq_kappa_sq / four_alpha_sq).exp()
                                / k_sq_kappa_sq
                        }
                    };
                    // Compute in f64, store as f32. The relative precision loss (~1e-7)
                    // is negligible for the reciprocal sum.
                    kx.push(kvx as f32);
                    ky.push(kvy as f32);
                    kz.push(kvz as f32);
                    aks.push(ak as f32);
                }
            }
        }

        Self { kx, ky, kz, aks }
    }

    pub fn len(&self) -> usize {
        self.kx.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_kvector_count() {
        let kvecs = KVectors::new([10.0, 10.0, 10.0], 2, 0.8, 0.0, EwaldPolicy::PBC);
        // Full sphere: 32, half-shell: 16
        assert_eq!(kvecs.len(), 16);
    }

    #[test]
    fn test_kvector_count_nmax1() {
        let kvecs = KVectors::new([10.0, 10.0, 10.0], 1, 0.8, 0.0, EwaldPolicy::PBC);
        assert_eq!(kvecs.len(), 3);
    }

    #[test]
    fn test_aks_coulomb() {
        let alpha = 0.8;
        let kvecs = KVectors::new([10.0, 10.0, 10.0], 1, alpha, 0.0, EwaldPolicy::PBC);
        let k2 = kvecs.kx[0] as f64 * kvecs.kx[0] as f64
            + kvecs.ky[0] as f64 * kvecs.ky[0] as f64
            + kvecs.kz[0] as f64 * kvecs.kz[0] as f64;
        let expected = 2.0 * (-k2 / (4.0 * alpha * alpha)).exp() / k2;
        assert_relative_eq!(kvecs.aks[0] as f64, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_aks_yukawa() {
        let kappa = 0.5;
        let alpha = 0.8;
        let kvecs = KVectors::new([10.0, 10.0, 10.0], 1, alpha, kappa, EwaldPolicy::PBC);
        let k2 = kvecs.kx[0] as f64 * kvecs.kx[0] as f64
            + kvecs.ky[0] as f64 * kvecs.ky[0] as f64
            + kvecs.kz[0] as f64 * kvecs.kz[0] as f64;
        let k2_kappa2 = k2 + kappa * kappa;
        let expected = 2.0 * (-k2_kappa2 / (4.0 * alpha * alpha)).exp() / k2_kappa2;
        assert_relative_eq!(kvecs.aks[0] as f64, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_yukawa_aks_smaller() {
        let kvecs_coulomb = KVectors::new([10.0, 10.0, 10.0], 3, 0.8, 0.0, EwaldPolicy::PBC);
        let kvecs_yukawa = KVectors::new([10.0, 10.0, 10.0], 3, 0.8, 0.5, EwaldPolicy::PBC);
        assert_eq!(kvecs_coulomb.len(), kvecs_yukawa.len());
        for (a_c, a_y) in kvecs_coulomb.aks.iter().zip(kvecs_yukawa.aks.iter()) {
            assert!(a_y < a_c);
        }
    }
}
