use super::super::kvectors::{cmul, madd, KVectors, Phases};

/// Accumulate one particle's contribution into PBC structure factors.
#[inline]
pub(in super::super) fn update_all_pbc(
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
        let (exy, zr, zi) = phases.row(row);
        let r = row.range();
        for ((sr, si), (&c, &d)) in sk_re[r.clone()]
            .iter_mut()
            .zip(&mut sk_im[r])
            .zip(zr.iter().zip(zi))
        {
            let (cos_kr, sin_kr) = cmul(exy, c, d);
            *sr = madd(charge, cos_kr, *sr);
            *si = madd(charge, sin_kr, *si);
        }
    }
}

/// cos(kx·x)·cos(ky·y) for the row, and its cos(kz·z) slice: the IPBC basis factors.
#[inline(always)]
fn cos_row<'a>(phases: &'a Phases, row: &super::super::kvectors::Row) -> (f64, &'a [f64]) {
    let (_, zr, _) = phases.row(row);
    (phases.axis(0, row.nx).0 * phases.axis(1, row.ny).0, zr)
}

/// Accumulate one particle's contribution into IPBC structure factors.
///
/// IPBC uses cos-product basis: $Q_k = \sum_j q_j \prod_\alpha \cos(k_\alpha r_{j\alpha})$
/// (see eq. 2 in [Stenqvist & Lund, 2018](https://doi.org/10/css8)).
#[inline]
pub(in super::super) fn update_all_ipbc(
    kvecs: &KVectors,
    px: f64,
    py: f64,
    pz: f64,
    charge: f64,
    sk_ipbc: &mut [f64],
) {
    let phases = Phases::new(kvecs, [px, py, pz]);
    for row in &kvecs.rows {
        let (cxy, cz) = cos_row(&phases, row);
        for (sk, &c) in sk_ipbc[row.range()].iter_mut().zip(cz) {
            *sk = madd(charge * cxy, c, *sk);
        }
    }
}

/// PBC energy: Σ A(k) |S(k)|²
#[inline]
pub(in super::super) fn energy_pbc(kvecs: &KVectors, sk_re: &[f64], sk_im: &[f64]) -> f64 {
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
pub(in super::super) fn energy_ipbc(kvecs: &KVectors, sk_ipbc: &[f64]) -> f64 {
    kvecs
        .aks
        .iter()
        .zip(sk_ipbc.iter())
        .map(|(&ak, &q)| ak as f64 * q * q)
        .sum()
}

/// Calls `f(k, ΔS_re, ΔS_im)` for each k-vector when `charge` moves from `old` to `new`.
#[inline(always)]
fn for_each_delta_pbc(
    kvecs: &KVectors,
    charge: f64,
    old: [f64; 3],
    new: [f64; 3],
    mut f: impl FnMut(usize, f64, f64),
) {
    let (p_old, p_new) = (Phases::new(kvecs, old), Phases::new(kvecs, new));
    for row in &kvecs.rows {
        let (exy_old, zr_old, zi_old) = p_old.row(row);
        let (exy_new, zr_new, zi_new) = p_new.row(row);
        for (j, k) in row.range().enumerate() {
            let (cos_old, sin_old) = cmul(exy_old, zr_old[j], zi_old[j]);
            let (cos_new, sin_new) = cmul(exy_new, zr_new[j], zi_new[j]);
            f(
                k,
                charge * (cos_new - cos_old),
                charge * (sin_new - sin_old),
            );
        }
    }
}

/// Calls `f(k, ΔQ)` for each k-vector when `charge` moves from `old` to `new`.
#[inline(always)]
fn for_each_delta_ipbc(
    kvecs: &KVectors,
    charge: f64,
    old: [f64; 3],
    new: [f64; 3],
    mut f: impl FnMut(usize, f64),
) {
    let (p_old, p_new) = (Phases::new(kvecs, old), Phases::new(kvecs, new));
    for row in &kvecs.rows {
        let (cxy_old, cz_old) = cos_row(&p_old, row);
        let (cxy_new, cz_new) = cos_row(&p_new, row);
        for (j, k) in row.range().enumerate() {
            f(k, charge * madd(cxy_new, cz_new[j], -cxy_old * cz_old[j]));
        }
    }
}

/// PBC energy change for a trial move: Σ A(k) [2S·ΔS + |ΔS|²]
#[inline]
pub(in super::super) fn energy_change_pbc(
    kvecs: &KVectors,
    sk_re: &[f64],
    sk_im: &[f64],
    charge: f64,
    old: [f64; 3],
    new: [f64; 3],
) -> f64 {
    let mut sum = 0.0;
    for_each_delta_pbc(kvecs, charge, old, new, |k, ds_re, ds_im| {
        let cross = 2.0 * madd(sk_re[k], ds_re, sk_im[k] * ds_im);
        let ds_sq = madd(ds_re, ds_re, ds_im * ds_im);
        sum += kvecs.aks[k] as f64 * (cross + ds_sq);
    });
    sum
}

/// IPBC energy change for a trial move: Σ A(k) [2Q·ΔQ + ΔQ²]
#[inline]
pub(in super::super) fn energy_change_ipbc(
    kvecs: &KVectors,
    sk_ipbc: &[f64],
    charge: f64,
    old: [f64; 3],
    new: [f64; 3],
) -> f64 {
    let mut sum = 0.0;
    for_each_delta_ipbc(kvecs, charge, old, new, |k, dq| {
        sum += kvecs.aks[k] as f64 * madd(dq, dq, 2.0 * sk_ipbc[k] * dq);
    });
    sum
}

/// PBC delta update for a single-particle move.
#[inline]
pub(in super::super) fn update_particle_pbc(
    kvecs: &KVectors,
    charge: f64,
    old: [f64; 3],
    new: [f64; 3],
    sk_re: &mut [f64],
    sk_im: &mut [f64],
) {
    for_each_delta_pbc(kvecs, charge, old, new, |k, ds_re, ds_im| {
        sk_re[k] += ds_re;
        sk_im[k] += ds_im;
    });
}

/// IPBC delta update for a single-particle move.
#[inline]
pub(in super::super) fn update_particle_ipbc(
    kvecs: &KVectors,
    charge: f64,
    old: [f64; 3],
    new: [f64; 3],
    sk_ipbc: &mut [f64],
) {
    for_each_delta_ipbc(kvecs, charge, old, new, |k, dq| sk_ipbc[k] += dq);
}

/// PBC force on one particle.
#[inline]
pub(in super::super) fn force_pbc(
    kvecs: &KVectors,
    pos: [f64; 3],
    sk_re: &[f64],
    sk_im: &[f64],
) -> [f64; 3] {
    let phases = Phases::new(kvecs, pos);
    let mut f = [0.0; 3];
    for row in &kvecs.rows {
        let (exy, zr, zi) = phases.row(row);
        for (j, k) in row.range().enumerate() {
            let (cos_kr, sin_kr) = cmul(exy, zr[j], zi[j]);
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
pub(in super::super) fn force_ipbc(kvecs: &KVectors, pos: [f64; 3], sk_ipbc: &[f64]) -> [f64; 3] {
    let phases = Phases::new(kvecs, pos);
    let mut f = [0.0; 3];
    for row in &kvecs.rows {
        let (cx, sx) = phases.axis(0, row.nx);
        let (cy, sy) = phases.axis(1, row.ny);
        let (_, zr, zi) = phases.row(row);
        for (j, k) in row.range().enumerate() {
            let (cz, sz) = (zr[j], zi[j]);
            let (kx, ky, kz) = (kvecs.kx[k], kvecs.ky[k], kvecs.kz[k]);
            let ak_q = kvecs.aks[k] as f64 * sk_ipbc[k];
            // d/dx [cos(kx·x)·cos(ky·y)·cos(kz·z)] = -kx·sin(kx·x)·cos(ky·y)·cos(kz·z)
            f[0] -= ak_q * kx as f64 * sx * cy * cz;
            f[1] -= ak_q * ky as f64 * cx * sy * cz;
            f[2] -= ak_q * kz as f64 * cx * cy * sz;
        }
    }
    f
}
