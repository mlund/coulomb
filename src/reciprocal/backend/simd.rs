use super::super::kvectors::KVectors;
use wide::{f64x4, i64x4, CmpEq};

type Simd = f64x4;
const LANES: usize = 4;

/// SIMD polynomial sin_cos with f32-level accuracy (~7 digits).
///
/// Uses Cody-Waite range reduction (3-part π/2 for extended precision) and
/// minimax polynomials: degree 7 for sin, degree 8 for cos. Coefficients
/// from Sleef (https://sleef.org). Quadrant handling via branchless SIMD
/// bit manipulation.
///
/// This is ~40% fewer FMAs than wide's built-in f64x4::sin_cos() (which
/// targets full f64 precision), and avoids per-lane scalar libm calls.
#[inline]
fn sin_cos_poly(x: Simd) -> (Simd, Simd) {
    // Range reduction constants: DP1 + DP2 + DP3 = π/2 (extended precision)
    const FRAC_2_PI: f64 = std::f64::consts::FRAC_2_PI;
    const DP1: f64 = 1.5703125;
    const DP2: f64 = 4.837512969970703125e-4;
    const DP3: f64 = 7.54978995489188216e-8;

    // Minimax sin coefficients: sin(r) = r + r³·(S0 + r²·(S1 + r²·S2))
    const S0: f64 = -1.66666537523269653320312e-1;
    const S1: f64 = 8.33215750753879547119141e-3;
    const S2: f64 = -1.95169282960705459117889e-4;

    // Minimax cos coefficients: cos(r) = 1 - r²/2 + r⁴·(C0 + r²·(C1 + r²·(C2 + r²·C3)))
    const C0: f64 = 4.16666641831398010253906e-2;
    const C1: f64 = -1.38888787478208541870117e-3;
    const C2: f64 = 2.47990446951007470488548e-5;
    const C3: f64 = -2.71811842367242206819355e-7;

    let xa = x.abs();
    let y = (xa * Simd::splat(FRAC_2_PI)).round();
    let q = y.round_int();

    // Cody-Waite subtraction: r = |x| - y·(π/2)
    let r = ((xa - y * Simd::splat(DP1)) - y * Simd::splat(DP2)) - y * Simd::splat(DP3);
    let r2 = r * r;

    // sin(r) ≈ r + r³·P(r²), degree 7
    let s = r
        + r * r2
            * (Simd::splat(S0) + r2 * (Simd::splat(S1) + r2 * Simd::splat(S2)));

    // cos(r) ≈ 1 - r²/2 + r⁴·Q(r²), degree 8
    let c = Simd::splat(1.0)
        + r2 * (Simd::splat(-0.5)
            + r2 * (Simd::splat(C0)
                + r2 * (Simd::splat(C1) + r2 * (Simd::splat(C2) + r2 * Simd::splat(C3)))));

    // Quadrant selection (branchless, matching wide's own approach):
    //   swap = q & 1 (odd quadrants: swap sin↔cos)
    //   sign_sin = (q << 62) ^ sign(x)   (negate in quadrants 2,3 × input sign)
    //   sign_cos = ((q+1) & 2) << 62      (negate in quadrants 1,2)
    let swap: Simd = cast::<i64x4, Simd>(!((q & i64x4::from(1)).simd_eq(i64x4::from(0))));
    let sin_poly = swap.blend(c, s);
    let cos_poly = swap.blend(s, c);

    let sign_sin: i64x4 = (q << 62) ^ cast::<Simd, i64x4>(x);
    let sin_out = sin_poly.flip_signs(cast::<i64x4, Simd>(sign_sin));

    let sign_cos: i64x4 = ((q + i64x4::from(1)) & i64x4::from(2)) << 62;
    let cos_out: Simd = cos_poly ^ cast::<i64x4, Simd>(sign_cos);

    (sin_out, cos_out)
}

/// Reinterpret-cast between same-size SIMD types.
#[inline(always)]
fn cast<A: Copy, B: Copy>(a: A) -> B {
    debug_assert_eq!(core::mem::size_of::<A>(), core::mem::size_of::<B>());
    unsafe { core::mem::transmute_copy(&a) }
}

/// Load LANES f32 elements from a slice, widening to f64x4.
#[inline(always)]
unsafe fn load_widen(slice: &[f32], i: usize) -> Simd {
    let p = slice.as_ptr().add(i);
    Simd::new([
        *p as f64,
        *p.add(1) as f64,
        *p.add(2) as f64,
        *p.add(3) as f64,
    ])
}

/// Load LANES f64 elements from a slice.
#[inline(always)]
unsafe fn load_f64(slice: &[f64], i: usize) -> Simd {
    core::mem::transmute::<[f64; LANES], Simd>(*(slice.as_ptr().add(i) as *const [f64; LANES]))
}

/// Horizontal sum of all lanes.
#[inline(always)]
fn hsum(v: Simd) -> f64 {
    let a: [f64; LANES] = unsafe { core::mem::transmute::<Simd, [f64; LANES]>(v) };
    a[0] + a[1] + a[2] + a[3]
}

/// Store LANES elements back into a mutable slice, accumulating (+=).
#[inline(always)]
fn store_add(slice: &mut [f64], i: usize, v: Simd) {
    let a: [f64; LANES] = unsafe { core::mem::transmute::<Simd, [f64; LANES]>(v) };
    for j in 0..LANES {
        slice[i + j] += a[j];
    }
}

// ── PBC operations ──────────────────────────────────────────────────────────

/// Accumulate one particle's contribution into PBC structure factors.
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
    let qv = Simd::splat(charge);
    let pxv = Simd::splat(px);
    let pyv = Simd::splat(py);
    let pzv = Simd::splat(pz);

    let mut i = 0;
    while i + LANES <= n {
        let kx = unsafe { load_widen(&kvecs.kx, i) };
        let ky = unsafe { load_widen(&kvecs.ky, i) };
        let kz = unsafe { load_widen(&kvecs.kz, i) };
        let kr = kx * pxv + ky * pyv + kz * pzv;
        let (sin_kr, cos_kr) = sin_cos_poly(kr);
        store_add(sk_re, i, qv * cos_kr);
        store_add(sk_im, i, qv * sin_kr);
        i += LANES;
    }
    let px = px as f32;
    let py = py as f32;
    let pz = pz as f32;
    for k in i..n {
        let (s, c) = (kvecs.kx[k] * px + kvecs.ky[k] * py + kvecs.kz[k] * pz).sin_cos();
        sk_re[k] += charge * c as f64;
        sk_im[k] += charge * s as f64;
    }
}

/// PBC energy: Σ A(k) |S(k)|²
pub(in super::super) fn energy_pbc(kvecs: &KVectors, sk_re: &[f64], sk_im: &[f64]) -> f64 {
    let n = kvecs.len();
    let mut acc = Simd::ZERO;
    let mut i = 0;
    while i + LANES <= n {
        let ak = unsafe { load_widen(&kvecs.aks, i) };
        let re = unsafe { load_f64(sk_re, i) };
        let im = unsafe { load_f64(sk_im, i) };
        acc += ak * (re * re + im * im);
        i += LANES;
    }
    let mut sum = hsum(acc);
    for k in i..n {
        sum += kvecs.aks[k] as f64 * (sk_re[k] * sk_re[k] + sk_im[k] * sk_im[k]);
    }
    sum
}

/// PBC energy change for a trial move.
pub(in super::super) fn energy_change_pbc(
    kvecs: &KVectors,
    sk_re: &[f64],
    sk_im: &[f64],
    charge: f64,
    old: [f64; 3],
    new: [f64; 3],
) -> f64 {
    let n = kvecs.len();
    let qv = Simd::splat(charge);
    let ox = Simd::splat(old[0]);
    let oy = Simd::splat(old[1]);
    let oz = Simd::splat(old[2]);
    let nx = Simd::splat(new[0]);
    let ny = Simd::splat(new[1]);
    let nz = Simd::splat(new[2]);
    let two = Simd::splat(2.0);

    let mut acc = Simd::ZERO;
    let mut i = 0;
    while i + LANES <= n {
        let kx = unsafe { load_widen(&kvecs.kx, i) };
        let ky = unsafe { load_widen(&kvecs.ky, i) };
        let kz = unsafe { load_widen(&kvecs.kz, i) };
        let ak = unsafe { load_widen(&kvecs.aks, i) };
        let s_re = unsafe { load_f64(sk_re, i) };
        let s_im = unsafe { load_f64(sk_im, i) };

        let (so, co) = sin_cos_poly(kx * ox + ky * oy + kz * oz);
        let (sn, cn) = sin_cos_poly(kx * nx + ky * ny + kz * nz);

        let ds_re = qv * (cn - co);
        let ds_im = qv * (sn - so);
        let cross = two * (s_re * ds_re + s_im * ds_im);
        let ds_sq = ds_re * ds_re + ds_im * ds_im;
        acc += ak * (cross + ds_sq);
        i += LANES;
    }
    let mut sum = hsum(acc);
    let old_f32 = [old[0] as f32, old[1] as f32, old[2] as f32];
    let new_f32 = [new[0] as f32, new[1] as f32, new[2] as f32];
    for k in i..n {
        let (so, co) =
            (kvecs.kx[k] * old_f32[0] + kvecs.ky[k] * old_f32[1] + kvecs.kz[k] * old_f32[2])
                .sin_cos();
        let (sn, cn) =
            (kvecs.kx[k] * new_f32[0] + kvecs.ky[k] * new_f32[1] + kvecs.kz[k] * new_f32[2])
                .sin_cos();
        let ds_re = charge * (cn - co) as f64;
        let ds_im = charge * (sn - so) as f64;
        sum += kvecs.aks[k] as f64
            * (2.0 * (sk_re[k] * ds_re + sk_im[k] * ds_im) + ds_re * ds_re + ds_im * ds_im);
    }
    sum
}

/// PBC delta update for a single-particle move.
pub(in super::super) fn update_particle_pbc(
    kvecs: &KVectors,
    charge: f64,
    old: [f64; 3],
    new: [f64; 3],
    sk_re: &mut [f64],
    sk_im: &mut [f64],
) {
    let n = kvecs.len();
    let qv = Simd::splat(charge);
    let ox = Simd::splat(old[0]);
    let oy = Simd::splat(old[1]);
    let oz = Simd::splat(old[2]);
    let nx = Simd::splat(new[0]);
    let ny = Simd::splat(new[1]);
    let nz = Simd::splat(new[2]);

    let mut i = 0;
    while i + LANES <= n {
        let kx = unsafe { load_widen(&kvecs.kx, i) };
        let ky = unsafe { load_widen(&kvecs.ky, i) };
        let kz = unsafe { load_widen(&kvecs.kz, i) };

        let (so, co) = sin_cos_poly(kx * ox + ky * oy + kz * oz);
        let (sn, cn) = sin_cos_poly(kx * nx + ky * ny + kz * nz);

        store_add(sk_re, i, qv * (cn - co));
        store_add(sk_im, i, qv * (sn - so));
        i += LANES;
    }
    let old_f32 = [old[0] as f32, old[1] as f32, old[2] as f32];
    let new_f32 = [new[0] as f32, new[1] as f32, new[2] as f32];
    for k in i..n {
        let (so, co) =
            (kvecs.kx[k] * old_f32[0] + kvecs.ky[k] * old_f32[1] + kvecs.kz[k] * old_f32[2])
                .sin_cos();
        let (sn, cn) =
            (kvecs.kx[k] * new_f32[0] + kvecs.ky[k] * new_f32[1] + kvecs.kz[k] * new_f32[2])
                .sin_cos();
        sk_re[k] += charge * (cn - co) as f64;
        sk_im[k] += charge * (sn - so) as f64;
    }
}

/// PBC force on one particle.
pub(in super::super) fn force_pbc(
    kvecs: &KVectors,
    pos: [f64; 3],
    sk_re: &[f64],
    sk_im: &[f64],
) -> [f64; 3] {
    let n = kvecs.len();
    let pxv = Simd::splat(pos[0]);
    let pyv = Simd::splat(pos[1]);
    let pzv = Simd::splat(pos[2]);

    let mut fxv = Simd::ZERO;
    let mut fyv = Simd::ZERO;
    let mut fzv = Simd::ZERO;

    let mut i = 0;
    while i + LANES <= n {
        let kx = unsafe { load_widen(&kvecs.kx, i) };
        let ky = unsafe { load_widen(&kvecs.ky, i) };
        let kz = unsafe { load_widen(&kvecs.kz, i) };
        let ak = unsafe { load_widen(&kvecs.aks, i) };
        let s_re = unsafe { load_f64(sk_re, i) };
        let s_im = unsafe { load_f64(sk_im, i) };

        let (sin_kr, cos_kr) = sin_cos_poly(kx * pxv + ky * pyv + kz * pzv);
        let term = ak * (sin_kr * s_re - cos_kr * s_im);
        fxv += term * kx;
        fyv += term * ky;
        fzv += term * kz;
        i += LANES;
    }

    let mut fx = hsum(fxv);
    let mut fy = hsum(fyv);
    let mut fz = hsum(fzv);
    let pos_f32 = [pos[0] as f32, pos[1] as f32, pos[2] as f32];
    for k in i..n {
        let (s, c) = (kvecs.kx[k] * pos_f32[0] + kvecs.ky[k] * pos_f32[1] + kvecs.kz[k] * pos_f32[2]).sin_cos();
        let term = kvecs.aks[k] as f64 * (s as f64 * sk_re[k] - c as f64 * sk_im[k]);
        fx += term * kvecs.kx[k] as f64;
        fy += term * kvecs.ky[k] as f64;
        fz += term * kvecs.kz[k] as f64;
    }
    [fx, fy, fz]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_sin_cos(values: &[f64], tol: f64) {
        for chunk in values.chunks(LANES) {
            let mut arr = [0.0f64; LANES];
            arr[..chunk.len()].copy_from_slice(chunk);
            let x: Simd = unsafe { core::mem::transmute::<[f64; LANES], Simd>(arr) };
            let (s, c) = sin_cos_poly(x);
            let s_arr: [f64; LANES] = unsafe { core::mem::transmute::<Simd, [f64; LANES]>(s) };
            let c_arr: [f64; LANES] = unsafe { core::mem::transmute::<Simd, [f64; LANES]>(c) };
            for (i, &val) in chunk.iter().enumerate() {
                let (es, ec) = val.sin_cos();
                assert!(
                    (s_arr[i] - es).abs() < tol,
                    "sin({val}) = {} vs {es}, err={}",
                    s_arr[i],
                    (s_arr[i] - es).abs()
                );
                assert!(
                    (c_arr[i] - ec).abs() < tol,
                    "cos({val}) = {} vs {ec}, err={}",
                    c_arr[i],
                    (c_arr[i] - ec).abs()
                );
            }
        }
    }

    #[test]
    fn test_sin_cos_basic() {
        check_sin_cos(
            &[0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI, 1.0],
            1e-6,
        );
    }

    #[test]
    fn test_sin_cos_negative() {
        check_sin_cos(&[-1.0, -3.0, -0.5, -6.0], 1e-6);
    }

    #[test]
    fn test_sin_cos_large() {
        // k·r values up to ~60 for typical Ewald parameters
        check_sin_cos(&[10.0, -10.0, 50.0, -50.0], 1e-5);
    }

    #[test]
    fn test_sin_cos_quadrants() {
        // Test all four quadrants
        check_sin_cos(&[0.5, 2.0, 3.5, 5.0], 1e-6);
        check_sin_cos(&[-0.5, -2.0, -3.5, -5.0], 1e-6);
    }

    #[test]
    fn test_sin_cos_near_multiples_of_pi() {
        use std::f64::consts::PI;
        check_sin_cos(&[PI, 2.0 * PI, -PI, -2.0 * PI], 1e-5);
        check_sin_cos(
            &[PI / 2.0, 3.0 * PI / 2.0, -PI / 2.0, -3.0 * PI / 2.0],
            1e-5,
        );
    }
}
