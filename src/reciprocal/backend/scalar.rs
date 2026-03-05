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
        let (sin_kr, cos_kr) = (kx * px + ky * py + kz * pz).sin_cos();
        *sr += charge * cos_kr as f64;
        *si += charge * sin_kr as f64;
    }
}

/// Accumulate one particle's contribution into IPBC structure factors.
#[inline]
pub(in super::super) fn update_all_ipbc(
    kvecs: &KVectors,
    px: f64,
    py: f64,
    pz: f64,
    charge: f64,
    sk_ipbc: &mut [[f64; 8]],
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
        let (sx, cx) = (kx * px).sin_cos();
        let (sy, cy) = (ky * py).sin_cos();
        let (sz, cz) = (kz * pz).sin_cos();
        let (sx, cx) = (sx as f64, cx as f64);
        let (sy, cy) = (sy as f64, cy as f64);
        let (sz, cz) = (sz as f64, cz as f64);
        sk[0] += charge * cx * cy * cz;
        sk[1] += charge * cx * cy * sz;
        sk[2] += charge * cx * sy * cz;
        sk[3] += charge * cx * sy * sz;
        sk[4] += charge * sx * cy * cz;
        sk[5] += charge * sx * cy * sz;
        sk[6] += charge * sx * sy * cz;
        sk[7] += charge * sx * sy * sz;
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
        .map(|((&ak, &re), &im)| ak as f64 * (re * re + im * im))
        .sum()
}

/// IPBC energy: Σ A(k) Σ_αβγ Q²_αβγ
#[inline]
pub(in super::super) fn energy_ipbc(kvecs: &KVectors, sk_ipbc: &[[f64; 8]]) -> f64 {
    kvecs
        .aks
        .iter()
        .zip(sk_ipbc.iter())
        .map(|(&ak, sk)| ak as f64 * sk.iter().map(|&v| v * v).sum::<f64>())
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
        let (sin_old, cos_old) =
            (kx * old_f32[0] + ky * old_f32[1] + kz * old_f32[2]).sin_cos();
        let (sin_new, cos_new) =
            (kx * new_f32[0] + ky * new_f32[1] + kz * new_f32[2]).sin_cos();
        let ds_re = charge * (cos_new - cos_old) as f64;
        let ds_im = charge * (sin_new - sin_old) as f64;
        let cross = 2.0 * (s_re * ds_re + s_im * ds_im);
        let ds_sq = ds_re * ds_re + ds_im * ds_im;
        sum += ak as f64 * (cross + ds_sq);
    }
    sum
}

/// IPBC energy change for a trial move.
#[inline]
pub(in super::super) fn energy_change_ipbc(
    kvecs: &KVectors,
    sk_ipbc: &[[f64; 8]],
    charge: f64,
    old: [f64; 3],
    new: [f64; 3],
) -> f64 {
    let old_f32 = [old[0] as f32, old[1] as f32, old[2] as f32];
    let new_f32 = [new[0] as f32, new[1] as f32, new[2] as f32];
    let mut sum = 0.0;
    for ((((&kx, &ky), &kz), &ak), sk) in kvecs
        .kx
        .iter()
        .zip(kvecs.ky.iter())
        .zip(kvecs.kz.iter())
        .zip(kvecs.aks.iter())
        .zip(sk_ipbc.iter())
    {
        let (sox, cox) = (kx * old_f32[0]).sin_cos();
        let (soy, coy) = (ky * old_f32[1]).sin_cos();
        let (soz, coz) = (kz * old_f32[2]).sin_cos();
        let (snx, cnx) = (kx * new_f32[0]).sin_cos();
        let (sny, cny) = (ky * new_f32[1]).sin_cos();
        let (snz, cnz) = (kz * new_f32[2]).sin_cos();
        let (sox, cox) = (sox as f64, cox as f64);
        let (soy, coy) = (soy as f64, coy as f64);
        let (soz, coz) = (soz as f64, coz as f64);
        let (snx, cnx) = (snx as f64, cnx as f64);
        let (sny, cny) = (sny as f64, cny as f64);
        let (snz, cnz) = (snz as f64, cnz as f64);
        let ds = [
            charge * (cnx * cny * cnz - cox * coy * coz),
            charge * (cnx * cny * snz - cox * coy * soz),
            charge * (cnx * sny * cnz - cox * soy * coz),
            charge * (cnx * sny * snz - cox * soy * soz),
            charge * (snx * cny * cnz - sox * coy * coz),
            charge * (snx * cny * snz - sox * coy * soz),
            charge * (snx * sny * cnz - sox * soy * coz),
            charge * (snx * sny * snz - sox * soy * soz),
        ];
        let (cross, ds_sq) = sk
            .iter()
            .zip(ds.iter())
            .fold((0.0, 0.0), |(c, d), (&s, &delta)| {
                (c + s * delta, d + delta * delta)
            });
        sum += ak as f64 * (2.0 * cross + ds_sq);
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
        let (sin_old, cos_old) =
            (kx * old_f32[0] + ky * old_f32[1] + kz * old_f32[2]).sin_cos();
        let (sin_new, cos_new) =
            (kx * new_f32[0] + ky * new_f32[1] + kz * new_f32[2]).sin_cos();
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
    sk_ipbc: &mut [[f64; 8]],
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
        let (sox, cox) = (kx * old_f32[0]).sin_cos();
        let (soy, coy) = (ky * old_f32[1]).sin_cos();
        let (soz, coz) = (kz * old_f32[2]).sin_cos();
        let (snx, cnx) = (kx * new_f32[0]).sin_cos();
        let (sny, cny) = (ky * new_f32[1]).sin_cos();
        let (snz, cnz) = (kz * new_f32[2]).sin_cos();
        let (sox, cox) = (sox as f64, cox as f64);
        let (soy, coy) = (soy as f64, coy as f64);
        let (soz, coz) = (soz as f64, coz as f64);
        let (snx, cnx) = (snx as f64, cnx as f64);
        let (sny, cny) = (sny as f64, cny as f64);
        let (snz, cnz) = (snz as f64, cnz as f64);
        sk[0] += charge * (cnx * cny * cnz - cox * coy * coz);
        sk[1] += charge * (cnx * cny * snz - cox * coy * soz);
        sk[2] += charge * (cnx * sny * cnz - cox * soy * coz);
        sk[3] += charge * (cnx * sny * snz - cox * soy * soz);
        sk[4] += charge * (snx * cny * cnz - sox * coy * coz);
        sk[5] += charge * (snx * cny * snz - sox * coy * soz);
        sk[6] += charge * (snx * sny * cnz - sox * soy * coz);
        sk[7] += charge * (snx * sny * snz - sox * soy * soz);
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
    let (mut fx, mut fy, mut fz) = (0.0, 0.0, 0.0);
    for (((((&kx, &ky), &kz), &ak), &s_re), &s_im) in kvecs
        .kx
        .iter()
        .zip(kvecs.ky.iter())
        .zip(kvecs.kz.iter())
        .zip(kvecs.aks.iter())
        .zip(sk_re.iter())
        .zip(sk_im.iter())
    {
        let (sin_kr, cos_kr) =
            (kx * pos_f32[0] + ky * pos_f32[1] + kz * pos_f32[2]).sin_cos();
        let term = ak as f64 * (sin_kr as f64 * s_re - cos_kr as f64 * s_im);
        fx += term * kx as f64;
        fy += term * ky as f64;
        fz += term * kz as f64;
    }
    [fx, fy, fz]
}

/// IPBC force on one particle.
#[inline]
pub(in super::super) fn force_ipbc(
    kvecs: &KVectors,
    pos: [f64; 3],
    sk_ipbc: &[[f64; 8]],
) -> [f64; 3] {
    let pos_f32 = [pos[0] as f32, pos[1] as f32, pos[2] as f32];
    let (mut fx, mut fy, mut fz) = (0.0, 0.0, 0.0);
    for ((((&kx, &ky), &kz), &ak), sk) in kvecs
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
        let (kxf, kyf, kzf) = (kx as f64, ky as f64, kz as f64);
        let akf = ak as f64;
        let dfx = kxf
            * (-sx * (sk[0] * cy * cz + sk[1] * cy * sz + sk[2] * sy * cz + sk[3] * sy * sz)
                + cx * (sk[4] * cy * cz + sk[5] * cy * sz + sk[6] * sy * cz + sk[7] * sy * sz));
        let dfy = kyf
            * (cx * (-sy * (sk[0] * cz + sk[1] * sz) + cy * (sk[2] * cz + sk[3] * sz))
                + sx * (-sy * (sk[4] * cz + sk[5] * sz) + cy * (sk[6] * cz + sk[7] * sz)));
        let dfz = kzf
            * (cx * (cy * (-sz * sk[0] + cz * sk[1]) + sy * (-sz * sk[2] + cz * sk[3]))
                + sx * (cy * (-sz * sk[4] + cz * sk[5]) + sy * (-sz * sk[6] + cz * sk[7])));
        fx -= akf * dfx;
        fy -= akf * dfy;
        fz -= akf * dfz;
    }
    [fx, fy, fz]
}
