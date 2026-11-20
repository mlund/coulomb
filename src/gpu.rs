//! Reciprocal-space data packed for a GPU force kernel.
//!
//! Kernels live with the caller, so this crate needs no GPU dependency. The module
//! fixes the layout and the formula a kernel must implement to reproduce
//! [`EwaldReciprocal::force`]:
//!
//! $$S(\mathbf{k}) = \sum_j q_j e^{i\mathbf{k}\cdot\mathbf{r}_j}$$
//! $$\mathbf{F}_i = \frac{4\pi q_i}{V} \sum_{\mathbf{k}} A(k)\,\mathbf{k}\,
//!   \operatorname{Im}\bigl[e^{i\mathbf{k}\cdot\mathbf{r}_i} S(\mathbf{k})^*\bigr]$$
//!
//! The sum runs over a half-shell of k-vectors; A(k) already carries the factor 2
//! for the omitted −**k**, so the kernel needs no symmetry branch.
//! Tinfoil boundary only: [`EwaldReciprocal::add_surface_forces`] is not included.

use crate::reciprocal::EwaldReciprocal;
use crate::Error;

/// k-vectors and prefactor for PBC reciprocal forces; see the [module docs](self).
///
/// A snapshot: rebuild after the box or the k-vector set changes.
#[derive(Debug, Clone)]
pub struct GpuKSpace {
    kvectors: Vec<[f32; 4]>,
    force_prefactor: f64,
}

impl GpuKSpace {
    /// `(kx, ky, kz, A(k))` per k-vector: one aligned vec4 load per k on the device.
    pub fn kvectors(&self) -> &[[f32; 4]] {
        &self.kvectors
    }

    /// 4π/V, kept in f64 so the caller can fold in its own units before narrowing.
    pub const fn force_prefactor(&self) -> f64 {
        self.force_prefactor
    }
}

impl TryFrom<&EwaldReciprocal> for GpuKSpace {
    type Error = Error;

    fn try_from(ewald: &EwaldReciprocal) -> Result<Self, Error> {
        let (kvectors, force_prefactor) = ewald
            .packed_kvectors()
            .ok_or(Error::Unsupported("GPU k-space requires the PBC policy"))?;
        Ok(Self {
            kvectors,
            force_prefactor,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Deterministic neutral box: pairs of ±q at pseudo-random positions.
    fn neutral_box(n: usize, box_length: f64) -> ([Vec<f64>; 3], Vec<f64>) {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut uniform = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut pos = [Vec::new(), Vec::new(), Vec::new()];
        let mut charges: Vec<f64> = Vec::new();
        for i in 0..n {
            for axis in &mut pos {
                axis.push((uniform() - 0.5) * box_length);
            }
            let q = 0.5 + uniform();
            charges.push(if i % 2 == 0 { q } else { -charges[i - 1] });
        }
        (pos, charges)
    }

    /// Direct transcription of the documented formula, in f64.
    fn naive_forces(gpu: &GpuKSpace, pos: &[Vec<f64>; 3], charges: &[f64]) -> Vec<[f64; 3]> {
        let k_dot_r = |k: &[f32; 4], i: usize| (0..3).map(|a| k[a] as f64 * pos[a][i]).sum::<f64>();
        let sk: Vec<(f64, f64)> = gpu
            .kvectors()
            .iter()
            .map(|k| {
                (0..charges.len()).fold((0.0, 0.0), |(re, im), j| {
                    let (s, c) = k_dot_r(k, j).sin_cos();
                    (re + charges[j] * c, im + charges[j] * s)
                })
            })
            .collect();
        (0..charges.len())
            .map(|i| {
                let mut f = [0.0; 3];
                for (k, &(re, im)) in gpu.kvectors().iter().zip(&sk) {
                    let (s, c) = k_dot_r(k, i).sin_cos();
                    let term = k[3] as f64 * (s * re - c * im);
                    for a in 0..3 {
                        f[a] += term * k[a] as f64;
                    }
                }
                f.map(|x| gpu.force_prefactor() * charges[i] * x)
            })
            .collect()
    }

    #[test]
    fn packed_kspace_reproduces_reciprocal_forces() {
        let box_length = 20.0;
        let (pos, charges) = neutral_box(40, box_length);
        let mut ewald = EwaldReciprocal::new([box_length; 3], 8.0, 1e-5, Some(15.0));
        ewald.update_all(&pos[0], &pos[1], &pos[2], &charges, None, false);

        let gpu = GpuKSpace::try_from(&ewald).unwrap();
        assert_eq!(gpu.kvectors().len(), ewald.num_k_vectors());

        let naive = naive_forces(&gpu, &pos, &charges);
        let f_max = naive.iter().flatten().fold(0.0_f64, |m, f| m.max(f.abs()));
        assert!(f_max > 1e-3, "degenerate test, F ≈ 0");
        for (i, f) in naive.iter().enumerate() {
            let expected = ewald.force([pos[0][i], pos[1][i], pos[2][i]], charges[i]);
            for a in 0..3 {
                assert_relative_eq!(f[a], expected[a], epsilon = 1e-6 * f_max);
            }
        }
    }

    #[test]
    fn ipbc_is_refused() {
        let mut ewald = EwaldReciprocal::new([20.0; 3], 8.0, 1e-5, None);
        ewald.set_policy(crate::EwaldPolicy::IPBC);
        assert!(matches!(
            GpuKSpace::try_from(&ewald),
            Err(Error::Unsupported(_))
        ));
    }
}
