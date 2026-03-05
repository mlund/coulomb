use super::super::kvectors::KVectors;

/// Accumulate one particle's contribution into PBC structure factors.
/// K-vectors are f32; sin/cos computed in f32; accumulated into f64.
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
    let px = px as f32;
    let py = py as f32;
    let pz = pz as f32;
    for ((((&kx, &ky), &kz), sr), si) in kvecs
        .kx
        .iter()
        .zip(kvecs.ky.iter())
        .zip(kvecs.kz.iter())
        .zip(sk_re.iter_mut())
        .zip(sk_im.iter_mut())
    {
        let (sin_kr, cos_kr) = kz.mul_add(pz, kx.mul_add(px, ky * py)).sin_cos();
        *sr += charge * cos_kr as f64;
        *si += charge * sin_kr as f64;
    }
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
    let px = px as f32;
    let py = py as f32;
    let pz = pz as f32;
    for (((&kx, &ky), &kz), sk) in kvecs
        .kx
        .iter()
        .zip(kvecs.ky.iter())
        .zip(kvecs.kz.iter())
        .zip(sk_ipbc.iter_mut())
    {
        let cx = (kx * px).cos();
        let cy = (ky * py).cos();
        let cz = (kz * pz).cos();
        *sk += charge * (cx * cy * cz) as f64;
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
        .map(|((&ak, &re), &im)| ak as f64 * re.mul_add(re, im * im))
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
    let old_f32 = [old[0] as f32, old[1] as f32, old[2] as f32];
    let new_f32 = [new[0] as f32, new[1] as f32, new[2] as f32];
    let mut sum = 0.0;
    for (((((&kx, &ky), &kz), &ak), &s_re), &s_im) in kvecs
        .kx
        .iter()
        .zip(kvecs.ky.iter())
        .zip(kvecs.kz.iter())
        .zip(kvecs.aks.iter())
        .zip(sk_re.iter())
        .zip(sk_im.iter())
    {
        let (sin_old, cos_old) = kz
            .mul_add(old_f32[2], kx.mul_add(old_f32[0], ky * old_f32[1]))
            .sin_cos();
        let (sin_new, cos_new) = kz
            .mul_add(new_f32[2], kx.mul_add(new_f32[0], ky * new_f32[1]))
            .sin_cos();
        let ds_re = charge * (cos_new - cos_old) as f64;
        let ds_im = charge * (sin_new - sin_old) as f64;
        let cross = 2.0 * s_re.mul_add(ds_re, s_im * ds_im);
        let ds_sq = ds_re.mul_add(ds_re, ds_im * ds_im);
        sum += ak as f64 * (cross + ds_sq);
    }
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
    let old_f32 = [old[0] as f32, old[1] as f32, old[2] as f32];
    let new_f32 = [new[0] as f32, new[1] as f32, new[2] as f32];
    let mut sum = 0.0;
    for ((((&kx, &ky), &kz), &ak), &q) in kvecs
        .kx
        .iter()
        .zip(kvecs.ky.iter())
        .zip(kvecs.kz.iter())
        .zip(kvecs.aks.iter())
        .zip(sk_ipbc.iter())
    {
        let cos_old = (kx * old_f32[0]).cos() * (ky * old_f32[1]).cos() * (kz * old_f32[2]).cos();
        let cos_new = (kx * new_f32[0]).cos() * (ky * new_f32[1]).cos() * (kz * new_f32[2]).cos();
        let dq = charge * (cos_new - cos_old) as f64;
        sum += ak as f64 * dq.mul_add(dq, 2.0 * q * dq);
    }
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
    let old_f32 = [old[0] as f32, old[1] as f32, old[2] as f32];
    let new_f32 = [new[0] as f32, new[1] as f32, new[2] as f32];
    for ((((&kx, &ky), &kz), sr), si) in kvecs
        .kx
        .iter()
        .zip(kvecs.ky.iter())
        .zip(kvecs.kz.iter())
        .zip(sk_re.iter_mut())
        .zip(sk_im.iter_mut())
    {
        let (sin_old, cos_old) = kz
            .mul_add(old_f32[2], kx.mul_add(old_f32[0], ky * old_f32[1]))
            .sin_cos();
        let (sin_new, cos_new) = kz
            .mul_add(new_f32[2], kx.mul_add(new_f32[0], ky * new_f32[1]))
            .sin_cos();
        *sr += charge * (cos_new - cos_old) as f64;
        *si += charge * (sin_new - sin_old) as f64;
    }
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
    let old_f32 = [old[0] as f32, old[1] as f32, old[2] as f32];
    let new_f32 = [new[0] as f32, new[1] as f32, new[2] as f32];
    for (((&kx, &ky), &kz), sk) in kvecs
        .kx
        .iter()
        .zip(kvecs.ky.iter())
        .zip(kvecs.kz.iter())
        .zip(sk_ipbc.iter_mut())
    {
        let cos_old = (kx * old_f32[0]).cos() * (ky * old_f32[1]).cos() * (kz * old_f32[2]).cos();
        let cos_new = (kx * new_f32[0]).cos() * (ky * new_f32[1]).cos() * (kz * new_f32[2]).cos();
        *sk += charge * (cos_new - cos_old) as f64;
    }
}

/// PBC force on one particle.
#[inline]
pub(in super::super) fn force_pbc(
    kvecs: &KVectors,
    pos: [f64; 3],
    sk_re: &[f64],
    sk_im: &[f64],
) -> [f64; 3] {
    let pos_f32 = [pos[0] as f32, pos[1] as f32, pos[2] as f32];
    let mut f = [0.0; 3];
    for (((((&kx, &ky), &kz), &ak), &s_re), &s_im) in kvecs
        .kx
        .iter()
        .zip(kvecs.ky.iter())
        .zip(kvecs.kz.iter())
        .zip(kvecs.aks.iter())
        .zip(sk_re.iter())
        .zip(sk_im.iter())
    {
        let (sin_kr, cos_kr) = kz
            .mul_add(pos_f32[2], kx.mul_add(pos_f32[0], ky * pos_f32[1]))
            .sin_cos();
        let term = ak as f64 * (sin_kr as f64).mul_add(s_re, -(cos_kr as f64 * s_im));
        f[0] = term.mul_add(kx as f64, f[0]);
        f[1] = term.mul_add(ky as f64, f[1]);
        f[2] = term.mul_add(kz as f64, f[2]);
    }
    f
}

/// IPBC force on one particle.
///
/// $F_{i,\alpha} = -\frac{4\pi q_i}{V} \sum_k A_k Q_k \cdot k_\alpha \sin(k_\alpha r_{i\alpha}) \prod_{\beta \neq \alpha} \cos(k_\beta r_{i\beta})$
#[inline]
pub(in super::super) fn force_ipbc(kvecs: &KVectors, pos: [f64; 3], sk_ipbc: &[f64]) -> [f64; 3] {
    let pos_f32 = [pos[0] as f32, pos[1] as f32, pos[2] as f32];
    let mut f = [0.0; 3];
    for ((((&kx, &ky), &kz), &ak), &q) in kvecs
        .kx
        .iter()
        .zip(kvecs.ky.iter())
        .zip(kvecs.kz.iter())
        .zip(kvecs.aks.iter())
        .zip(sk_ipbc.iter())
    {
        let (sx, cx) = (kx * pos_f32[0]).sin_cos();
        let (sy, cy) = (ky * pos_f32[1]).sin_cos();
        let (sz, cz) = (kz * pos_f32[2]).sin_cos();
        let (sx, cx) = (sx as f64, cx as f64);
        let (sy, cy) = (sy as f64, cy as f64);
        let (sz, cz) = (sz as f64, cz as f64);
        let ak_q = ak as f64 * q;
        // d/dx [cos(kx·x)·cos(ky·y)·cos(kz·z)] = -kx·sin(kx·x)·cos(ky·y)·cos(kz·z)
        f[0] -= ak_q * kx as f64 * sx * cy * cz;
        f[1] -= ak_q * ky as f64 * cx * sy * cz;
        f[2] -= ak_q * kz as f64 * cx * cy * sz;
    }
    f
}
