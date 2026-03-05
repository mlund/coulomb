use super::super::kvectors::KVectors;
use wide::f64x4;

/// Compute sin and cos for each lane of an f64x4.
///
/// Uses scalar `f64::sin_cos` per lane for full precision. The SIMD benefit
/// comes from the surrounding FMA arithmetic on k-vectors, not from this
/// function itself. A future polynomial approximation or wgpu shader can
/// replace this without changing the callers.
#[inline]
fn sin_cos_f64x4(x: f64x4) -> (f64x4, f64x4) {
    let x_arr: [f64; 4] = unsafe { core::mem::transmute(x) };
    let (s0, c0) = x_arr[0].sin_cos();
    let (s1, c1) = x_arr[1].sin_cos();
    let (s2, c2) = x_arr[2].sin_cos();
    let (s3, c3) = x_arr[3].sin_cos();
    (
        f64x4::new([s0, s1, s2, s3]),
        f64x4::new([c0, c1, c2, c3]),
    )
}

/// Accumulate one particle's contribution into PBC structure factors (SIMD).
pub(in super::super) fn update_all_pbc(
    kvecs: &KVectors,
    px: f64,
    py: f64,
    pz: f64,
    charge: f64,
    sk_re: &mut [f64],
    sk_im: &mut [f64],
) {
    let n = kvecs.len();
    let charge4 = f64x4::splat(charge);
    let px4 = f64x4::splat(px);
    let py4 = f64x4::splat(py);
    let pz4 = f64x4::splat(pz);

    let mut i = 0;
    while i + 4 <= n {
        let kx = f64x4::new(unsafe { *(kvecs.kx.as_ptr().add(i) as *const [f64; 4]) });
        let ky = f64x4::new(unsafe { *(kvecs.ky.as_ptr().add(i) as *const [f64; 4]) });
        let kz = f64x4::new(unsafe { *(kvecs.kz.as_ptr().add(i) as *const [f64; 4]) });

        let kr = kx * px4 + ky * py4 + kz * pz4;
        let (sin_kr, cos_kr) = sin_cos_f64x4(kr);
        let dre = charge4 * cos_kr;
        let dim = charge4 * sin_kr;

        let dre_arr = unsafe { core::mem::transmute::<f64x4, [f64; 4]>(dre) };
        let dim_arr = unsafe { core::mem::transmute::<f64x4, [f64; 4]>(dim) };
        for j in 0..4 {
            sk_re[i + j] += dre_arr[j];
            sk_im[i + j] += dim_arr[j];
        }
        i += 4;
    }
    // Scalar tail
    for k in i..n {
        let (sin_kr, cos_kr) =
            (kvecs.kx[k] * px + kvecs.ky[k] * py + kvecs.kz[k] * pz).sin_cos();
        sk_re[k] += charge * cos_kr;
        sk_im[k] += charge * sin_kr;
    }
}

/// PBC energy (SIMD).
pub(in super::super) fn energy_pbc(kvecs: &KVectors, sk_re: &[f64], sk_im: &[f64]) -> f64 {
    let n = kvecs.len();
    let mut sum4 = f64x4::ZERO;
    let mut i = 0;
    while i + 4 <= n {
        let ak = f64x4::new(unsafe { *(kvecs.aks.as_ptr().add(i) as *const [f64; 4]) });
        let re = f64x4::new(unsafe { *(sk_re.as_ptr().add(i) as *const [f64; 4]) });
        let im = f64x4::new(unsafe { *(sk_im.as_ptr().add(i) as *const [f64; 4]) });
        sum4 += ak * (re * re + im * im);
        i += 4;
    }
    let sum_arr = unsafe { core::mem::transmute::<f64x4, [f64; 4]>(sum4) };
    let mut sum = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];
    for k in i..n {
        sum += kvecs.aks[k] * (sk_re[k] * sk_re[k] + sk_im[k] * sk_im[k]);
    }
    sum
}

/// PBC energy change for a trial move (SIMD).
pub(in super::super) fn energy_change_pbc(
    kvecs: &KVectors,
    sk_re: &[f64],
    sk_im: &[f64],
    charge: f64,
    old: [f64; 3],
    new: [f64; 3],
) -> f64 {
    let n = kvecs.len();
    let charge4 = f64x4::splat(charge);
    let old_x4 = f64x4::splat(old[0]);
    let old_y4 = f64x4::splat(old[1]);
    let old_z4 = f64x4::splat(old[2]);
    let new_x4 = f64x4::splat(new[0]);
    let new_y4 = f64x4::splat(new[1]);
    let new_z4 = f64x4::splat(new[2]);
    let two4 = f64x4::splat(2.0);

    let mut sum4 = f64x4::ZERO;
    let mut i = 0;
    while i + 4 <= n {
        let kx = f64x4::new(unsafe { *(kvecs.kx.as_ptr().add(i) as *const [f64; 4]) });
        let ky = f64x4::new(unsafe { *(kvecs.ky.as_ptr().add(i) as *const [f64; 4]) });
        let kz = f64x4::new(unsafe { *(kvecs.kz.as_ptr().add(i) as *const [f64; 4]) });
        let ak = f64x4::new(unsafe { *(kvecs.aks.as_ptr().add(i) as *const [f64; 4]) });
        let s_re = f64x4::new(unsafe { *(sk_re.as_ptr().add(i) as *const [f64; 4]) });
        let s_im = f64x4::new(unsafe { *(sk_im.as_ptr().add(i) as *const [f64; 4]) });

        let kr_old = kx * old_x4 + ky * old_y4 + kz * old_z4;
        let kr_new = kx * new_x4 + ky * new_y4 + kz * new_z4;
        let (sin_old, cos_old) = sin_cos_f64x4(kr_old);
        let (sin_new, cos_new) = sin_cos_f64x4(kr_new);

        let ds_re = charge4 * (cos_new - cos_old);
        let ds_im = charge4 * (sin_new - sin_old);
        let cross = two4 * (s_re * ds_re + s_im * ds_im);
        let ds_sq = ds_re * ds_re + ds_im * ds_im;
        sum4 += ak * (cross + ds_sq);
        i += 4;
    }
    let sum_arr = unsafe { core::mem::transmute::<f64x4, [f64; 4]>(sum4) };
    let mut sum = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];
    for k in i..n {
        let (sin_old, cos_old) =
            (kvecs.kx[k] * old[0] + kvecs.ky[k] * old[1] + kvecs.kz[k] * old[2]).sin_cos();
        let (sin_new, cos_new) =
            (kvecs.kx[k] * new[0] + kvecs.ky[k] * new[1] + kvecs.kz[k] * new[2]).sin_cos();
        let ds_re = charge * (cos_new - cos_old);
        let ds_im = charge * (sin_new - sin_old);
        let cross = 2.0 * (sk_re[k] * ds_re + sk_im[k] * ds_im);
        let ds_sq = ds_re * ds_re + ds_im * ds_im;
        sum += kvecs.aks[k] * (cross + ds_sq);
    }
    sum
}

/// PBC delta update for a single-particle move (SIMD).
pub(in super::super) fn update_particle_pbc(
    kvecs: &KVectors,
    charge: f64,
    old: [f64; 3],
    new: [f64; 3],
    sk_re: &mut [f64],
    sk_im: &mut [f64],
) {
    let n = kvecs.len();
    let charge4 = f64x4::splat(charge);
    let old_x4 = f64x4::splat(old[0]);
    let old_y4 = f64x4::splat(old[1]);
    let old_z4 = f64x4::splat(old[2]);
    let new_x4 = f64x4::splat(new[0]);
    let new_y4 = f64x4::splat(new[1]);
    let new_z4 = f64x4::splat(new[2]);

    let mut i = 0;
    while i + 4 <= n {
        let kx = f64x4::new(unsafe { *(kvecs.kx.as_ptr().add(i) as *const [f64; 4]) });
        let ky = f64x4::new(unsafe { *(kvecs.ky.as_ptr().add(i) as *const [f64; 4]) });
        let kz = f64x4::new(unsafe { *(kvecs.kz.as_ptr().add(i) as *const [f64; 4]) });

        let kr_old = kx * old_x4 + ky * old_y4 + kz * old_z4;
        let kr_new = kx * new_x4 + ky * new_y4 + kz * new_z4;
        let (sin_old, cos_old) = sin_cos_f64x4(kr_old);
        let (sin_new, cos_new) = sin_cos_f64x4(kr_new);

        let dre = charge4 * (cos_new - cos_old);
        let dim = charge4 * (sin_new - sin_old);

        let dre_arr = unsafe { core::mem::transmute::<f64x4, [f64; 4]>(dre) };
        let dim_arr = unsafe { core::mem::transmute::<f64x4, [f64; 4]>(dim) };
        for j in 0..4 {
            sk_re[i + j] += dre_arr[j];
            sk_im[i + j] += dim_arr[j];
        }
        i += 4;
    }
    for k in i..n {
        let (sin_old, cos_old) =
            (kvecs.kx[k] * old[0] + kvecs.ky[k] * old[1] + kvecs.kz[k] * old[2]).sin_cos();
        let (sin_new, cos_new) =
            (kvecs.kx[k] * new[0] + kvecs.ky[k] * new[1] + kvecs.kz[k] * new[2]).sin_cos();
        sk_re[k] += charge * (cos_new - cos_old);
        sk_im[k] += charge * (sin_new - sin_old);
    }
}

/// PBC force on one particle (SIMD).
pub(in super::super) fn force_pbc(
    kvecs: &KVectors,
    pos: [f64; 3],
    sk_re: &[f64],
    sk_im: &[f64],
) -> [f64; 3] {
    let n = kvecs.len();
    let px4 = f64x4::splat(pos[0]);
    let py4 = f64x4::splat(pos[1]);
    let pz4 = f64x4::splat(pos[2]);

    let mut fx4 = f64x4::ZERO;
    let mut fy4 = f64x4::ZERO;
    let mut fz4 = f64x4::ZERO;

    let mut i = 0;
    while i + 4 <= n {
        let kx = f64x4::new(unsafe { *(kvecs.kx.as_ptr().add(i) as *const [f64; 4]) });
        let ky = f64x4::new(unsafe { *(kvecs.ky.as_ptr().add(i) as *const [f64; 4]) });
        let kz = f64x4::new(unsafe { *(kvecs.kz.as_ptr().add(i) as *const [f64; 4]) });
        let ak = f64x4::new(unsafe { *(kvecs.aks.as_ptr().add(i) as *const [f64; 4]) });
        let s_re = f64x4::new(unsafe { *(sk_re.as_ptr().add(i) as *const [f64; 4]) });
        let s_im = f64x4::new(unsafe { *(sk_im.as_ptr().add(i) as *const [f64; 4]) });

        let kr = kx * px4 + ky * py4 + kz * pz4;
        let (sin_kr, cos_kr) = sin_cos_f64x4(kr);
        let term = ak * (sin_kr * s_re - cos_kr * s_im);
        fx4 += term * kx;
        fy4 += term * ky;
        fz4 += term * kz;
        i += 4;
    }

    let fx_arr = unsafe { core::mem::transmute::<f64x4, [f64; 4]>(fx4) };
    let fy_arr = unsafe { core::mem::transmute::<f64x4, [f64; 4]>(fy4) };
    let fz_arr = unsafe { core::mem::transmute::<f64x4, [f64; 4]>(fz4) };
    let mut fx = fx_arr[0] + fx_arr[1] + fx_arr[2] + fx_arr[3];
    let mut fy = fy_arr[0] + fy_arr[1] + fy_arr[2] + fy_arr[3];
    let mut fz = fz_arr[0] + fz_arr[1] + fz_arr[2] + fz_arr[3];

    for k in i..n {
        let (sin_kr, cos_kr) =
            (kvecs.kx[k] * pos[0] + kvecs.ky[k] * pos[1] + kvecs.kz[k] * pos[2]).sin_cos();
        let term = kvecs.aks[k] * (sin_kr * sk_re[k] - cos_kr * sk_im[k]);
        fx += term * kvecs.kx[k];
        fy += term * kvecs.ky[k];
        fz += term * kvecs.kz[k];
    }
    [fx, fy, fz]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sin_cos_f64x4_basic() {
        let x = f64x4::new([0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI, 1.0]);
        let (s, c) = sin_cos_f64x4(x);
        let s_arr = unsafe { core::mem::transmute::<f64x4, [f64; 4]>(s) };
        let c_arr = unsafe { core::mem::transmute::<f64x4, [f64; 4]>(c) };

        assert!((s_arr[0] - 0.0).abs() < 1e-14, "sin(0) = {}", s_arr[0]);
        assert!((c_arr[0] - 1.0).abs() < 1e-14, "cos(0) = {}", c_arr[0]);
        assert!((s_arr[1] - 1.0).abs() < 1e-14, "sin(π/2) = {}", s_arr[1]);
        assert!((c_arr[1] - 0.0).abs() < 1e-14, "cos(π/2) = {}", c_arr[1]);
        assert!((s_arr[2] - 0.0).abs() < 1e-14, "sin(π) = {}", s_arr[2]);
        assert!(
            (c_arr[2] - (-1.0)).abs() < 1e-14,
            "cos(π) = {}",
            c_arr[2]
        );

        let (expected_s, expected_c) = 1.0f64.sin_cos();
        assert!(
            (s_arr[3] - expected_s).abs() < 1e-12,
            "sin(1) = {} vs {}",
            s_arr[3],
            expected_s
        );
        assert!(
            (c_arr[3] - expected_c).abs() < 1e-12,
            "cos(1) = {} vs {}",
            c_arr[3],
            expected_c
        );
    }

    #[test]
    fn test_sin_cos_f64x4_negative() {
        let x = f64x4::new([-1.0, -3.0, -0.5, -6.0]);
        let (s, c) = sin_cos_f64x4(x);
        let s_arr = unsafe { core::mem::transmute::<f64x4, [f64; 4]>(s) };
        let c_arr = unsafe { core::mem::transmute::<f64x4, [f64; 4]>(c) };

        for (i, &val) in [-1.0f64, -3.0, -0.5, -6.0].iter().enumerate() {
            let (es, ec) = val.sin_cos();
            assert!(
                (s_arr[i] - es).abs() < 1e-12,
                "sin({val}) = {} vs {es}",
                s_arr[i]
            );
            assert!(
                (c_arr[i] - ec).abs() < 1e-12,
                "cos({val}) = {} vs {ec}",
                c_arr[i]
            );
        }
    }

    #[test]
    fn test_sin_cos_f64x4_large() {
        let x = f64x4::new([100.0, -100.0, 1000.0, -1000.0]);
        let (s, c) = sin_cos_f64x4(x);
        let s_arr = unsafe { core::mem::transmute::<f64x4, [f64; 4]>(s) };
        let c_arr = unsafe { core::mem::transmute::<f64x4, [f64; 4]>(c) };

        for (i, &val) in [100.0f64, -100.0, 1000.0, -1000.0].iter().enumerate() {
            let (es, ec) = val.sin_cos();
            assert!(
                (s_arr[i] - es).abs() < 1e-10,
                "sin({val}) = {} vs {es}",
                s_arr[i]
            );
            assert!(
                (c_arr[i] - ec).abs() < 1e-10,
                "cos({val}) = {} vs {ec}",
                c_arr[i]
            );
        }
    }
}
