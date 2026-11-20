use core::f64::consts::PI;

use super::EwaldPolicy;

/// K-vector storage (SoA, f32) with precomputed A(k) coefficients.
#[derive(Clone)]
pub(super) struct KVectors {
    pub(super) kx: Vec<f32>,
    pub(super) ky: Vec<f32>,
    pub(super) kz: Vec<f32>,
    pub(super) aks: Vec<f32>,
    /// Runs of k-vectors sharing (nx, ny), in storage order.
    pub(super) rows: Vec<Row>,
    two_pi_over_l: [f64; 3],
    n_max: u32,
}

/// K-vectors `start..start + len` share (nx, ny) and have consecutive nz from `nz`.
/// Fields other than `start` and `len` are indices into a [`Phases`] axis table.
#[derive(Clone, Copy)]
pub(super) struct Row {
    pub start: u32,
    pub len: u32,
    pub nx: u32,
    pub ny: u32,
    pub nz: u32,
}

impl Row {
    #[inline(always)]
    pub fn range(&self) -> core::ops::Range<usize> {
        self.start as usize..(self.start + self.len) as usize
    }
}

/// Per-axis phase factors e^{i n θ_a}, θ_a = 2π r_a / L_a, indexed by n + n_max.
///
/// e^{ik·r} factorizes over axes, so a particle costs three sin_cos calls and a
/// recurrence instead of one sin_cos per k-vector. Within a [`Row`] only the z factor
/// varies, so inner loops are contiguous and vectorize. f64 throughout, which is also
/// more precise than f32 k·r.
pub(super) struct Phases {
    re: [Vec<f64>; 3],
    im: [Vec<f64>; 3],
}

impl Phases {
    pub fn new(kvecs: &KVectors, pos: [f64; 3]) -> Self {
        let n = kvecs.n_max as usize;
        let axis = |a: usize| {
            let (s, c) = (kvecs.two_pi_over_l[a] * pos[a]).sin_cos();
            let (mut re, mut im) = (vec![1.0_f64; 2 * n + 1], vec![0.0_f64; 2 * n + 1]);
            for m in 1..=n {
                let (r, i) = (re[n + m - 1], im[n + m - 1]);
                (re[n + m], im[n + m]) = (madd(r, c, -i * s), madd(r, s, i * c));
                (re[n - m], im[n - m]) = (re[n + m], -im[n + m]);
            }
            (re, im)
        };
        let [x, y, z] = [axis(0), axis(1), axis(2)];
        Self {
            re: [x.0, y.0, z.0],
            im: [x.1, y.1, z.1],
        }
    }

    /// (cos, sin) of k_a r_a on axis `a`.
    #[inline(always)]
    pub fn axis(&self, a: usize, i: u32) -> (f64, f64) {
        (self.re[a][i as usize], self.im[a][i as usize])
    }

    /// e^{i(kx·x + ky·y)} for the row, and its z factors as (cos, sin) slices.
    #[inline(always)]
    pub fn row(&self, row: &Row) -> ((f64, f64), &[f64], &[f64]) {
        let (c, d) = self.axis(1, row.ny);
        let z = row.nz as usize..(row.nz + row.len) as usize;
        (
            cmul(self.axis(0, row.nx), c, d),
            &self.re[2][z.clone()],
            &self.im[2][z],
        )
    }
}

/// a·b + c, fused only where FMA is a native instruction: elsewhere `mul_add` calls a
/// software fma, several times slower than the loop it sits in.
#[inline(always)]
pub(super) fn madd(a: f64, b: f64, c: f64) -> f64 {
    #[cfg(any(target_arch = "aarch64", target_feature = "fma"))]
    return a.mul_add(b, c);
    #[cfg(not(any(target_arch = "aarch64", target_feature = "fma")))]
    return a * b + c;
}

/// (a + ib)(c + id)
#[inline(always)]
pub(super) fn cmul((a, b): (f64, f64), c: f64, d: f64) -> (f64, f64) {
    (madd(a, c, -b * d), madd(a, d, b * c))
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
        debug_assert!(
            alpha > 0.0 || (alpha == 0.0 && n_max == 0 && kappa > 0.0),
            "alpha must be positive unless screened reciprocal space is empty"
        );
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
        let mut rows: Vec<Row> = Vec::new();

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
                    let offset = |n: i32| (n + n_max_i) as u32;
                    match rows.last_mut() {
                        Some(r) if r.nx == offset(nx) && r.ny == offset(ny) => r.len += 1,
                        _ => rows.push(Row {
                            start: kx.len() as u32 - 1,
                            len: 1,
                            nx: offset(nx),
                            ny: offset(ny),
                            nz: offset(nz),
                        }),
                    }
                }
            }
        }

        Self {
            kx,
            ky,
            kz,
            aks,
            rows,
            two_pi_over_l,
            n_max,
        }
    }

    pub fn len(&self) -> usize {
        self.kx.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Rows tile the k-vectors in order, and table products reproduce e^{ik·r}.
    #[test]
    fn phases_match_direct_exp_ikr() {
        let box_length = [10.0, 12.0, 15.0];
        for policy in [EwaldPolicy::PBC, EwaldPolicy::IPBC] {
            let kvecs = KVectors::new(box_length, 6, 0.8, 0.0, policy);
            let mut next = 0;
            for row in &kvecs.rows {
                assert_eq!(row.start as usize, next);
                next += row.len as usize;
            }
            assert_eq!(next, kvecs.len());

            // Far outside the box, so the recurrence is tested at large phases too
            let pos = [3.7, -41.2, 118.5];
            let phases = Phases::new(&kvecs, pos);
            for row in &kvecs.rows {
                let (exy, zr, zi) = phases.row(row);
                for (j, k) in row.range().enumerate() {
                    let n = |l: f64, kl: f32| (kl as f64 * l / (2.0 * PI)).round();
                    let kr: f64 = (0..3)
                        .map(|a| {
                            let kl = [kvecs.kx[k], kvecs.ky[k], kvecs.kz[k]][a];
                            2.0 * PI * n(box_length[a], kl) / box_length[a] * pos[a]
                        })
                        .sum();
                    let (re, im) = cmul(exy, zr[j], zi[j]);
                    assert!((re - kr.cos()).abs() < 1e-12, "{policy:?} k={k}");
                    assert!((im - kr.sin()).abs() < 1e-12, "{policy:?} k={k}");
                }
            }
        }
    }

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
