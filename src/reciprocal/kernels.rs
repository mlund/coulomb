//! Reciprocal-space inner loops.
//!
//! k-vectors and A(k) are stored as f32, halving memory bandwidth; phases,
//! structure factors and sums are f64. e^{ik·r} factorizes over axes (see [`Phases`]),
//! so a particle needs three sin_cos calls rather than one per k-vector, which beats a
//! per-k-vector SIMD polynomial on both x86_64 and aarch64.

use core::ops::Range;

use super::kvectors::{KVectors, Row};

/// a·b + c, fused only where FMA is a native instruction: elsewhere `mul_add` calls a
/// software fma, several times slower than the loop it sits in.
#[inline(always)]
fn madd(a: f64, b: f64, c: f64) -> f64 {
    #[cfg(any(target_arch = "aarch64", target_feature = "fma"))]
    return a.mul_add(b, c);
    #[cfg(not(any(target_arch = "aarch64", target_feature = "fma")))]
    return a * b + c;
}

/// (a + ib)(c + id)
#[inline(always)]
fn cmul((a, b): (f64, f64), c: f64, d: f64) -> (f64, f64) {
    (madd(a, c, -b * d), madd(a, d, b * c))
}

/// Per-axis phase factors e^{i n θ_a}, θ_a = 2π r_a / L_a, indexed by n + n_max.
///
/// Within a [`Row`] only the z factor varies, so inner loops are contiguous and
/// vectorize. One allocation: cos then sin, each as x, y, z tables of `stride`.
pub(super) struct Phases {
    t: Vec<f64>,
    stride: usize,
}

/// The phase factors of one [`Row`]: x and y as (cos, sin), z as slices.
struct RowPhases<'a> {
    x: (f64, f64),
    y: (f64, f64),
    z_cos: &'a [f64],
    z_sin: &'a [f64],
}

impl RowPhases<'_> {
    /// e^{i(kx·x + ky·y)}
    #[inline(always)]
    fn exy(&self) -> (f64, f64) {
        cmul(self.x, self.y.0, self.y.1)
    }
}

impl Phases {
    pub fn new(kvecs: &KVectors, pos: [f64; 3]) -> Self {
        let n = kvecs.n_max as usize;
        let stride = 2 * n + 1;
        let mut t = vec![0.0; 6 * stride];
        let (cos, sin) = t.split_at_mut(3 * stride);
        for a in 0..3 {
            let (s, c) = (kvecs.two_pi_over_l[a] * pos[a]).sin_cos();
            let (cos, sin) = (
                &mut cos[a * stride..][..stride],
                &mut sin[a * stride..][..stride],
            );
            cos[n] = 1.0;
            for m in 1..=n {
                (cos[n + m], sin[n + m]) = cmul((cos[n + m - 1], sin[n + m - 1]), c, s);
                (cos[n - m], sin[n - m]) = (cos[n + m], -sin[n + m]);
            }
        }
        Self { t, stride }
    }

    #[inline(always)]
    fn axis(&self, a: usize, i: u32) -> (f64, f64) {
        let i = a * self.stride + i as usize;
        (self.t[i], self.t[3 * self.stride + i])
    }

    #[inline(always)]
    fn row(&self, row: &Row) -> RowPhases<'_> {
        let z = 2 * self.stride + row.nz as usize;
        let (cos, sin) = self.t.split_at(3 * self.stride);
        let len = row.len as usize;
        RowPhases {
            x: self.axis(0, row.nx),
            y: self.axis(1, row.ny),
            z_cos: &cos[z..z + len],
            z_sin: &sin[z..z + len],
        }
    }
}

/// Accumulate one particle's contribution into PBC structure factors.
#[inline]
pub(super) fn update_all_pbc(
    kvecs: &KVectors,
    px: f64,
    py: f64,
    pz: f64,
    charge: f64,
    sk_re: &mut [f64],
    sk_im: &mut [f64],
) {
    let phases = Phases::new(kvecs, [px, py, pz]);
    for row in &kvecs.rows {
        let p = phases.row(row);
        let exy = p.exy();
        let r = row.range();
        for ((sr, si), (&c, &d)) in sk_re[r.clone()]
            .iter_mut()
            .zip(&mut sk_im[r])
            .zip(p.z_cos.iter().zip(p.z_sin))
        {
            let (cos_kr, sin_kr) = cmul(exy, c, d);
            *sr = madd(charge, cos_kr, *sr);
            *si = madd(charge, sin_kr, *si);
        }
    }
}

/// Accumulate one particle's contribution into IPBC structure factors.
///
/// IPBC uses cos-product basis: $Q_k = \sum_j q_j \prod_\alpha \cos(k_\alpha r_{j\alpha})$
/// (see eq. 2 in [Stenqvist & Lund, 2018](https://doi.org/10/css8)).
#[inline]
pub(super) fn update_all_ipbc(
    kvecs: &KVectors,
    px: f64,
    py: f64,
    pz: f64,
    charge: f64,
    sk_ipbc: &mut [f64],
) {
    let phases = Phases::new(kvecs, [px, py, pz]);
    for row in &kvecs.rows {
        let p = phases.row(row);
        let q_cxy = charge * p.x.0 * p.y.0;
        for (sk, &c) in sk_ipbc[row.range()].iter_mut().zip(p.z_cos) {
            *sk = madd(q_cxy, c, *sk);
        }
    }
}

/// PBC energy: Σ A(k) |S(k)|²
#[inline]
pub(super) fn energy_pbc(kvecs: &KVectors, sk_re: &[f64], sk_im: &[f64]) -> f64 {
    kvecs
        .aks
        .iter()
        .zip(sk_re.iter())
        .zip(sk_im.iter())
        .map(|((&ak, &re), &im)| ak as f64 * madd(re, re, im * im))
        .sum()
}

/// IPBC energy: Σ A(k) Q(k)²
#[inline]
pub(super) fn energy_ipbc(kvecs: &KVectors, sk_ipbc: &[f64]) -> f64 {
    kvecs
        .aks
        .iter()
        .zip(sk_ipbc.iter())
        .map(|(&ak, &q)| ak as f64 * q * q)
        .sum()
}

/// Calls `f(range, ΔS_re, ΔS_im)` per row when `charge` moves from `old` to `new`,
/// so callers slice their arrays once per row instead of indexing per k-vector.
#[inline(always)]
fn for_each_delta_pbc(
    kvecs: &KVectors,
    charge: f64,
    old: [f64; 3],
    new: [f64; 3],
    mut f: impl FnMut(Range<usize>, &[f64], &[f64]),
) {
    let (p_old, p_new) = (Phases::new(kvecs, old), Phases::new(kvecs, new));
    let mut ds = vec![0.0; 2 * p_old.stride];
    let (ds_re, ds_im) = ds.split_at_mut(p_old.stride);
    for row in &kvecs.rows {
        let (o, n) = (p_old.row(row), p_new.row(row));
        let (exy_old, exy_new) = (o.exy(), n.exy());
        let len = row.len as usize;
        for ((dr, di), ((&co, &so), (&cn, &sn))) in ds_re[..len]
            .iter_mut()
            .zip(&mut ds_im[..len])
            .zip(o.z_cos.iter().zip(o.z_sin).zip(n.z_cos.iter().zip(n.z_sin)))
        {
            let (cos_old, sin_old) = cmul(exy_old, co, so);
            let (cos_new, sin_new) = cmul(exy_new, cn, sn);
            (*dr, *di) = (charge * (cos_new - cos_old), charge * (sin_new - sin_old));
        }
        f(row.range(), &ds_re[..len], &ds_im[..len]);
    }
}

/// Calls `f(range, ΔQ)` per row when `charge` moves from `old` to `new`.
#[inline(always)]
fn for_each_delta_ipbc(
    kvecs: &KVectors,
    charge: f64,
    old: [f64; 3],
    new: [f64; 3],
    mut f: impl FnMut(Range<usize>, &[f64]),
) {
    let (p_old, p_new) = (Phases::new(kvecs, old), Phases::new(kvecs, new));
    let mut dq = vec![0.0; p_old.stride];
    for row in &kvecs.rows {
        let (o, n) = (p_old.row(row), p_new.row(row));
        let (q_old, q_new) = (charge * o.x.0 * o.y.0, charge * n.x.0 * n.y.0);
        let len = row.len as usize;
        for (d, (&co, &cn)) in dq[..len].iter_mut().zip(o.z_cos.iter().zip(n.z_cos)) {
            *d = madd(q_new, cn, -q_old * co);
        }
        f(row.range(), &dq[..len]);
    }
}

/// PBC energy change for a trial move: Σ A(k) [2S·ΔS + |ΔS|²]
#[inline]
pub(super) fn energy_change_pbc(
    kvecs: &KVectors,
    sk_re: &[f64],
    sk_im: &[f64],
    charge: f64,
    old: [f64; 3],
    new: [f64; 3],
) -> f64 {
    let mut sum = 0.0;
    for_each_delta_pbc(kvecs, charge, old, new, |r, ds_re, ds_im| {
        for ((&ak, (&s_re, &s_im)), (&d_re, &d_im)) in kvecs.aks[r.clone()]
            .iter()
            .zip(sk_re[r.clone()].iter().zip(&sk_im[r]))
            .zip(ds_re.iter().zip(ds_im))
        {
            let cross = 2.0 * madd(s_re, d_re, s_im * d_im);
            let ds_sq = madd(d_re, d_re, d_im * d_im);
            sum += ak as f64 * (cross + ds_sq);
        }
    });
    sum
}

/// IPBC energy change for a trial move: Σ A(k) [2Q·ΔQ + ΔQ²]
#[inline]
pub(super) fn energy_change_ipbc(
    kvecs: &KVectors,
    sk_ipbc: &[f64],
    charge: f64,
    old: [f64; 3],
    new: [f64; 3],
) -> f64 {
    let mut sum = 0.0;
    for_each_delta_ipbc(kvecs, charge, old, new, |r, dq| {
        for ((&ak, &q), &d) in kvecs.aks[r.clone()].iter().zip(&sk_ipbc[r]).zip(dq) {
            sum += ak as f64 * madd(d, d, 2.0 * q * d);
        }
    });
    sum
}

/// PBC delta update for a single-particle move.
///
/// Writes straight into S(k) rather than via [`for_each_delta_pbc`], saving a pass.
#[inline]
pub(super) fn update_particle_pbc(
    kvecs: &KVectors,
    charge: f64,
    old: [f64; 3],
    new: [f64; 3],
    sk_re: &mut [f64],
    sk_im: &mut [f64],
) {
    let (p_old, p_new) = (Phases::new(kvecs, old), Phases::new(kvecs, new));
    for row in &kvecs.rows {
        let (o, n) = (p_old.row(row), p_new.row(row));
        let (exy_old, exy_new) = (o.exy(), n.exy());
        let r = row.range();
        for ((sr, si), ((&co, &so), (&cn, &sn))) in sk_re[r.clone()]
            .iter_mut()
            .zip(&mut sk_im[r])
            .zip(o.z_cos.iter().zip(o.z_sin).zip(n.z_cos.iter().zip(n.z_sin)))
        {
            let (cos_old, sin_old) = cmul(exy_old, co, so);
            let (cos_new, sin_new) = cmul(exy_new, cn, sn);
            *sr = madd(charge, cos_new - cos_old, *sr);
            *si = madd(charge, sin_new - sin_old, *si);
        }
    }
}

/// IPBC delta update for a single-particle move.
#[inline]
pub(super) fn update_particle_ipbc(
    kvecs: &KVectors,
    charge: f64,
    old: [f64; 3],
    new: [f64; 3],
    sk_ipbc: &mut [f64],
) {
    for_each_delta_ipbc(kvecs, charge, old, new, |r, dq| {
        for (s, d) in sk_ipbc[r].iter_mut().zip(dq) {
            *s += d;
        }
    });
}

/// PBC force on one particle.
#[inline]
pub(super) fn force_pbc(kvecs: &KVectors, pos: [f64; 3], sk_re: &[f64], sk_im: &[f64]) -> [f64; 3] {
    let phases = Phases::new(kvecs, pos);
    let mut f = [0.0; 3];
    for row in &kvecs.rows {
        let p = phases.row(row);
        let exy = p.exy();
        for (j, k) in row.range().enumerate() {
            let (cos_kr, sin_kr) = cmul(exy, p.z_cos[j], p.z_sin[j]);
            let term = kvecs.aks[k] as f64 * madd(sin_kr, sk_re[k], -(cos_kr * sk_im[k]));
            f[0] = madd(term, kvecs.kx[k] as f64, f[0]);
            f[1] = madd(term, kvecs.ky[k] as f64, f[1]);
            f[2] = madd(term, kvecs.kz[k] as f64, f[2]);
        }
    }
    f
}

/// IPBC force on one particle.
///
/// $F_{i,\alpha} = -\frac{4\pi q_i}{V} \sum_k A_k Q_k \cdot k_\alpha \sin(k_\alpha r_{i\alpha}) \prod_{\beta \neq \alpha} \cos(k_\beta r_{i\beta})$
#[inline]
pub(super) fn force_ipbc(kvecs: &KVectors, pos: [f64; 3], sk_ipbc: &[f64]) -> [f64; 3] {
    let phases = Phases::new(kvecs, pos);
    let mut f = [0.0; 3];
    for row in &kvecs.rows {
        let p = phases.row(row);
        let ((cx, sx), (cy, sy)) = (p.x, p.y);
        for (j, k) in row.range().enumerate() {
            let (cz, sz) = (p.z_cos[j], p.z_sin[j]);
            let ak_q = kvecs.aks[k] as f64 * sk_ipbc[k];
            // d/dx [cos(kx·x)·cos(ky·y)·cos(kz·z)] = -kx·sin(kx·x)·cos(ky·y)·cos(kz·z)
            f[0] -= ak_q * kvecs.kx[k] as f64 * sx * cy * cz;
            f[1] -= ak_q * kvecs.ky[k] as f64 * cx * sy * cz;
            f[2] -= ak_q * kvecs.kz[k] as f64 * cx * cy * sz;
        }
    }
    f
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reciprocal::EwaldPolicy;
    use core::f64::consts::PI;

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
                let p = phases.row(row);
                for (j, k) in row.range().enumerate() {
                    let kr: f64 = (0..3)
                        .map(|a| {
                            let kl = [kvecs.kx[k], kvecs.ky[k], kvecs.kz[k]][a] as f64;
                            let n = (kl * box_length[a] / (2.0 * PI)).round();
                            2.0 * PI * n / box_length[a] * pos[a]
                        })
                        .sum();
                    let (re, im) = cmul(p.exy(), p.z_cos[j], p.z_sin[j]);
                    assert!((re - kr.cos()).abs() < 1e-12, "{policy:?} k={k}");
                    assert!((im - kr.sin()).abs() < 1e-12, "{policy:?} k={k}");
                }
            }
        }
    }
}
