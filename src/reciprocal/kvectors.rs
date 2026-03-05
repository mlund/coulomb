use core::f64::consts::PI;

/// Private k-vector storage with precomputed A(k) coefficients.
///
/// Generates half-shell k-vectors within a spherical cutoff in integer k-space,
/// excluding k=0. Only one of each ±k pair is stored (nx > 0, or nx == 0 && ny > 0,
/// or nx == 0 && ny == 0 && nz > 0). The A(k) coefficients include a factor of 2
/// to account for the missing partner.
///
/// Stored in SoA layout for cache-friendly inner loops and future SIMD.
pub(super) struct KVectors {
    pub(super) kx: Vec<f64>,
    pub(super) ky: Vec<f64>,
    pub(super) kz: Vec<f64>,
    /// Precomputed 2 × exp(-(k² + κ²)/(4α²)) / (k² + κ²), doubled for half-shell
    pub(super) aks: Vec<f64>,
}

impl KVectors {
    /// Generate half-shell k-vectors for a box with given side lengths.
    ///
    /// - `box_length`: Side lengths `[Lx, Ly, Lz]` of the orthorhombic simulation box
    /// - `n_max`: Maximum integer k-space cutoff (spherical: n² ≤ n_max²)
    /// - `alpha`: Ewald splitting parameter
    /// - `kappa`: Inverse Debye screening length (0 for Coulomb)
    pub fn new(box_length: [f64; 3], n_max: u32, alpha: f64, kappa: f64) -> Self {
        debug_assert!(alpha > 0.0, "alpha must be positive");
        debug_assert!(kappa >= 0.0, "kappa must be non-negative");
        debug_assert!(
            box_length[0] > 0.0 && box_length[1] > 0.0 && box_length[2] > 0.0,
            "box lengths must be positive"
        );

        let two_pi_over_l = [
            2.0 * PI / box_length[0],
            2.0 * PI / box_length[1],
            2.0 * PI / box_length[2],
        ];
        let n_max_i = n_max as i32;
        let n_max_sq = n_max_i * n_max_i;
        let four_alpha_sq = 4.0 * alpha * alpha;
        let kappa_sq = kappa * kappa;

        let estimated = (2.0 / 3.0 * PI * (n_max as f64).powi(3)).ceil() as usize;
        let mut kx = Vec::with_capacity(estimated);
        let mut ky = Vec::with_capacity(estimated);
        let mut kz = Vec::with_capacity(estimated);
        let mut aks = Vec::with_capacity(estimated);

        // Half-shell: only generate k-vectors where the first non-zero component is positive
        for nx in 0..=n_max_i {
            let ny_start = if nx == 0 { 0 } else { -n_max_i };
            for ny in ny_start..=n_max_i {
                let nz_start = if nx == 0 && ny == 0 { 1 } else { -n_max_i };
                for nz in nz_start..=n_max_i {
                    let n_sq = nx * nx + ny * ny + nz * nz;
                    if n_sq == 0 || n_sq > n_max_sq {
                        continue;
                    }
                    let kvx = nx as f64 * two_pi_over_l[0];
                    let kvy = ny as f64 * two_pi_over_l[1];
                    let kvz = nz as f64 * two_pi_over_l[2];
                    let k_sq = kvx * kvx + kvy * kvy + kvz * kvz;
                    let k_sq_kappa_sq = k_sq + kappa_sq;
                    let ak = 2.0 * (-k_sq_kappa_sq / four_alpha_sq).exp() / k_sq_kappa_sq;
                    kx.push(kvx);
                    ky.push(kvy);
                    kz.push(kvz);
                    aks.push(ak);
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
        let kvecs = KVectors::new([10.0, 10.0, 10.0], 2, 0.8, 0.0);
        // Full sphere: 32, half-shell: 16
        assert_eq!(kvecs.len(), 16);
    }

    #[test]
    fn test_kvector_count_nmax1() {
        let kvecs = KVectors::new([10.0, 10.0, 10.0], 1, 0.8, 0.0);
        assert_eq!(kvecs.len(), 3);
    }

    #[test]
    fn test_aks_coulomb() {
        let alpha = 0.8;
        let kvecs = KVectors::new([10.0, 10.0, 10.0], 1, alpha, 0.0);
        let k2 = kvecs.kx[0] * kvecs.kx[0] + kvecs.ky[0] * kvecs.ky[0] + kvecs.kz[0] * kvecs.kz[0];
        let expected = 2.0 * (-k2 / (4.0 * alpha * alpha)).exp() / k2;
        assert_relative_eq!(kvecs.aks[0], expected);
    }

    #[test]
    fn test_aks_yukawa() {
        let kappa = 0.5;
        let alpha = 0.8;
        let kvecs = KVectors::new([10.0, 10.0, 10.0], 1, alpha, kappa);
        let k2 = kvecs.kx[0] * kvecs.kx[0] + kvecs.ky[0] * kvecs.ky[0] + kvecs.kz[0] * kvecs.kz[0];
        let k2_kappa2 = k2 + kappa * kappa;
        let expected = 2.0 * (-k2_kappa2 / (4.0 * alpha * alpha)).exp() / k2_kappa2;
        assert_relative_eq!(kvecs.aks[0], expected);
    }

    #[test]
    fn test_yukawa_aks_smaller() {
        let kvecs_coulomb = KVectors::new([10.0, 10.0, 10.0], 3, 0.8, 0.0);
        let kvecs_yukawa = KVectors::new([10.0, 10.0, 10.0], 3, 0.8, 0.5);
        assert_eq!(kvecs_coulomb.len(), kvecs_yukawa.len());
        for (a_c, a_y) in kvecs_coulomb.aks.iter().zip(kvecs_yukawa.aks.iter()) {
            assert!(a_y < a_c);
        }
    }
}
