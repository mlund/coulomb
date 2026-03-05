use crate::NalgebraVector3;
use core::f64::consts::PI;

/// Private k-vector storage with precomputed A(k) coefficients.
///
/// Generates half-shell k-vectors within a spherical cutoff in integer k-space,
/// excluding k=0. Only one of each ±k pair is stored (nx > 0, or nx == 0 && ny > 0,
/// or nx == 0 && ny == 0 && nz > 0). The A(k) coefficients include a factor of 2
/// to account for the missing partner. For each k-vector, precomputes the kernel
/// A(k) used in energy and force calculations.
pub(super) struct KVectors {
    /// k-vectors in reciprocal space (2π/L units already applied), half-shell only
    pub(super) vectors: Vec<NalgebraVector3>,
    /// Precomputed 2 × exp(-(k² + κ²)/(4α²)) / (k² + κ²), doubled for half-shell
    pub(super) aks: Vec<f64>,
}

impl KVectors {
    /// Generate half-shell k-vectors for a box with given side lengths.
    ///
    /// - `box_length`: Side lengths of the orthorhombic simulation box
    /// - `n_max`: Maximum integer k-space cutoff (spherical: n² ≤ n_max²)
    /// - `alpha`: Ewald splitting parameter
    /// - `kappa`: Inverse Debye screening length (0 for Coulomb)
    pub fn new(box_length: NalgebraVector3, n_max: u32, alpha: f64, kappa: f64) -> Self {
        debug_assert!(alpha > 0.0, "alpha must be positive");
        debug_assert!(kappa >= 0.0, "kappa must be non-negative");
        debug_assert!(
            box_length.x > 0.0 && box_length.y > 0.0 && box_length.z > 0.0,
            "box lengths must be positive"
        );

        let two_pi_over_l = NalgebraVector3::new(
            2.0 * PI / box_length.x,
            2.0 * PI / box_length.y,
            2.0 * PI / box_length.z,
        );
        let n_max_i = n_max as i32;
        let n_max_sq = n_max_i * n_max_i;
        let four_alpha_sq = 4.0 * alpha * alpha;
        let kappa_sq = kappa * kappa;

        // Estimate capacity: roughly half the full sphere volume
        let estimated = (2.0 / 3.0 * PI * (n_max as f64).powi(3)).ceil() as usize;
        let mut vectors = Vec::with_capacity(estimated);
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
                    let k = NalgebraVector3::new(
                        nx as f64 * two_pi_over_l.x,
                        ny as f64 * two_pi_over_l.y,
                        nz as f64 * two_pi_over_l.z,
                    );
                    let k_sq = k.norm_squared();
                    let k_sq_kappa_sq = k_sq + kappa_sq;
                    // Factor of 2 accounts for the -k partner
                    let ak = 2.0 * (-k_sq_kappa_sq / four_alpha_sq).exp() / k_sq_kappa_sq;
                    vectors.push(k);
                    aks.push(ak);
                }
            }
        }

        Self { vectors, aks }
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_kvector_count() {
        // Box 10x10x10, n_max=2: integer triplets with 0 < nx²+ny²+nz² ≤ 4
        let kvecs = KVectors::new(NalgebraVector3::new(10.0, 10.0, 10.0), 2, 0.8, 0.0);
        // Full sphere: n²=1: 6, n²=2: 12, n²=3: 8, n²=4: 6 → 32 total
        // Half-shell: 32 / 2 = 16
        assert_eq!(kvecs.len(), 16);
    }

    #[test]
    fn test_kvector_count_nmax1() {
        let kvecs = KVectors::new(NalgebraVector3::new(10.0, 10.0, 10.0), 1, 0.8, 0.0);
        // Full: 6 → half: 3
        assert_eq!(kvecs.len(), 3);
    }

    #[test]
    fn test_aks_coulomb() {
        // For κ=0: A(k) = 2 × exp(-k²/(4α²)) / k² (factor 2 for half-shell)
        let alpha = 0.8;
        let kvecs = KVectors::new(NalgebraVector3::new(10.0, 10.0, 10.0), 1, alpha, 0.0);
        let k = kvecs.vectors[0];
        let k2 = k.norm_squared();
        let expected = 2.0 * (-k2 / (4.0 * alpha * alpha)).exp() / k2;
        assert_relative_eq!(kvecs.aks[0], expected);
    }

    #[test]
    fn test_aks_yukawa() {
        // For κ>0: A(k) = 2 × exp(-(k²+κ²)/(4α²)) / (k²+κ²)
        let kappa = 0.5;
        let alpha = 0.8;
        let kvecs = KVectors::new(NalgebraVector3::new(10.0, 10.0, 10.0), 1, alpha, kappa);
        let k = kvecs.vectors[0];
        let k2 = k.norm_squared();
        let k2_kappa2 = k2 + kappa * kappa;
        let expected = 2.0 * (-k2_kappa2 / (4.0 * alpha * alpha)).exp() / k2_kappa2;
        assert_relative_eq!(kvecs.aks[0], expected);
    }

    #[test]
    fn test_yukawa_aks_smaller() {
        let box_len = NalgebraVector3::new(10.0, 10.0, 10.0);
        let kvecs_coulomb = KVectors::new(box_len, 3, 0.8, 0.0);
        let kvecs_yukawa = KVectors::new(box_len, 3, 0.8, 0.5);
        assert_eq!(kvecs_coulomb.len(), kvecs_yukawa.len());
        for (a_c, a_y) in kvecs_coulomb.aks.iter().zip(kvecs_yukawa.aks.iter()) {
            assert!(a_y < a_c);
        }
    }
}
