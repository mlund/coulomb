use super::super::kvectors::KVectors;

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
    for ((((&kx, &ky), &kz), sr), si) in kvecs
        .kx
        .iter()
        .zip(kvecs.ky.iter())
        .zip(kvecs.kz.iter())
        .zip(sk_re.iter_mut())
        .zip(sk_im.iter_mut())
    {
        let (sin_kr, cos_kr) = (kx * px + ky * py + kz * pz).sin_cos();
        *sr += charge * cos_kr;
        *si += charge * sin_kr;
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
        .map(|((&ak, &re), &im)| ak * (re * re + im * im))
        .sum()
}

/// IPBC energy: Σ A(k) Σ_αβγ Q²_αβγ
#[inline]
pub(in super::super) fn energy_ipbc(kvecs: &KVectors, sk_ipbc: &[[f64; 8]]) -> f64 {
    kvecs
        .aks
        .iter()
        .zip(sk_ipbc.iter())
        .map(|(&ak, sk)| ak * sk.iter().map(|&v| v * v).sum::<f64>())
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
        let (sin_old, cos_old) = (kx * old[0] + ky * old[1] + kz * old[2]).sin_cos();
        let (sin_new, cos_new) = (kx * new[0] + ky * new[1] + kz * new[2]).sin_cos();
        let ds_re = charge * (cos_new - cos_old);
        let ds_im = charge * (sin_new - sin_old);
        let cross = 2.0 * (s_re * ds_re + s_im * ds_im);
        let ds_sq = ds_re * ds_re + ds_im * ds_im;
        sum += ak * (cross + ds_sq);
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
    let mut sum = 0.0;
    for ((((&kx, &ky), &kz), &ak), sk) in kvecs
        .kx
        .iter()
        .zip(kvecs.ky.iter())
        .zip(kvecs.kz.iter())
        .zip(kvecs.aks.iter())
        .zip(sk_ipbc.iter())
    {
        let (sox, cox) = (kx * old[0]).sin_cos();
        let (soy, coy) = (ky * old[1]).sin_cos();
        let (soz, coz) = (kz * old[2]).sin_cos();
        let (snx, cnx) = (kx * new[0]).sin_cos();
        let (sny, cny) = (ky * new[1]).sin_cos();
        let (snz, cnz) = (kz * new[2]).sin_cos();
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
        sum += ak * (2.0 * cross + ds_sq);
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
    for ((((&kx, &ky), &kz), sr), si) in kvecs
        .kx
        .iter()
        .zip(kvecs.ky.iter())
        .zip(kvecs.kz.iter())
        .zip(sk_re.iter_mut())
        .zip(sk_im.iter_mut())
    {
        let (sin_old, cos_old) = (kx * old[0] + ky * old[1] + kz * old[2]).sin_cos();
        let (sin_new, cos_new) = (kx * new[0] + ky * new[1] + kz * new[2]).sin_cos();
        *sr += charge * (cos_new - cos_old);
        *si += charge * (sin_new - sin_old);
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
    for (((&kx, &ky), &kz), sk) in kvecs
        .kx
        .iter()
        .zip(kvecs.ky.iter())
        .zip(kvecs.kz.iter())
        .zip(sk_ipbc.iter_mut())
    {
        let (sox, cox) = (kx * old[0]).sin_cos();
        let (soy, coy) = (ky * old[1]).sin_cos();
        let (soz, coz) = (kz * old[2]).sin_cos();
        let (snx, cnx) = (kx * new[0]).sin_cos();
        let (sny, cny) = (ky * new[1]).sin_cos();
        let (snz, cnz) = (kz * new[2]).sin_cos();
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
        let (sin_kr, cos_kr) = (kx * pos[0] + ky * pos[1] + kz * pos[2]).sin_cos();
        let term = ak * (sin_kr * s_re - cos_kr * s_im);
        fx += term * kx;
        fy += term * ky;
        fz += term * kz;
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
    let (mut fx, mut fy, mut fz) = (0.0, 0.0, 0.0);
    for ((((&kx, &ky), &kz), &ak), sk) in kvecs
        .kx
        .iter()
        .zip(kvecs.ky.iter())
        .zip(kvecs.kz.iter())
        .zip(kvecs.aks.iter())
        .zip(sk_ipbc.iter())
    {
        let (sx, cx) = (kx * pos[0]).sin_cos();
        let (sy, cy) = (ky * pos[1]).sin_cos();
        let (sz, cz) = (kz * pos[2]).sin_cos();
        let dfx = kx
            * (-sx
                * (sk[0] * cy * cz + sk[1] * cy * sz + sk[2] * sy * cz + sk[3] * sy * sz)
                + cx * (sk[4] * cy * cz + sk[5] * cy * sz + sk[6] * sy * cz + sk[7] * sy * sz));
        let dfy = ky
            * (cx * (-sy * (sk[0] * cz + sk[1] * sz) + cy * (sk[2] * cz + sk[3] * sz))
                + sx * (-sy * (sk[4] * cz + sk[5] * sz) + cy * (sk[6] * cz + sk[7] * sz)));
        let dfz = kz
            * (cx * (cy * (-sz * sk[0] + cz * sk[1]) + sy * (-sz * sk[2] + cz * sk[3]))
                + sx * (cy * (-sz * sk[4] + cz * sk[5]) + sy * (-sz * sk[6] + cz * sk[7])));
        fx -= ak * dfx;
        fy -= ak * dfy;
        fz -= ak * dfz;
    }
    [fx, fy, fz]
}
