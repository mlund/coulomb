use core::f64::consts::PI;

/// Ewald splitting parameter α from cutoff and accuracy target.
///
/// Coulomb (κ = 0): α = √(−ln ε) / r_c (Kolafa–Perram).
/// Yukawa (κ > 0): uses the tighter Pålsson–Tornberg bound
/// ([arXiv:1911.04875](https://arxiv.org/abs/1911.04875)),
/// returning 0 when the reciprocal sum can be skipped entirely.
pub(crate) fn ewald_alpha(cutoff: f64, accuracy: f64, kappa: f64) -> f64 {
    debug_assert!(cutoff > 0.0, "cutoff must be positive");
    debug_assert!(
        accuracy > 0.0 && accuracy < 1.0,
        "accuracy must be in (0, 1)"
    );
    debug_assert!(kappa >= 0.0, "kappa must be non-negative");
    let neg_ln_eps = -accuracy.ln();
    if kappa == 0.0 {
        return neg_ln_eps.sqrt() / cutoff;
    }
    // Pålsson–Tornberg bound: exp(-(αr_c + κ/(2α))²) = ε is tighter than
    // Kolafa–Perram because it includes the screening term κ/(2α) that adds
    // to the Gaussian damping, yielding a smaller α and thus fewer k-vectors.
    // Substituting u = αr_c gives quadratic u² - √(-ln ε)·u + κr_c/2 = 0.
    let discriminant = neg_ln_eps - 2.0 * kappa * cutoff;
    if discriminant <= 0.0 {
        // 2κr_c ≥ -ln ε: Yukawa potential decays fast enough that the
        // real-space sum alone achieves the target accuracy — no reciprocal sum needed.
        return 0.0;
    }
    // Upper root: gives a larger α (more work in reciprocal space) but ensures
    // the real-space error bound is satisfied. The lower root would under-damp
    // real space, violating the accuracy target.
    let u = (neg_ln_eps.sqrt() + discriminant.sqrt()) / 2.0;
    u / cutoff
}

/// Reciprocal-space integer cutoff n_max for given Ewald parameters.
///
/// k²_c = max(4α²(−ln ε) − κ², 0), then n_max = ⌈k_c · L_max / 2π⌉.
/// Returns 0 when screening makes the reciprocal sum unnecessary.
pub(crate) fn ewald_n_max(box_length: &[f64; 3], alpha: f64, accuracy: f64, kappa: f64) -> u32 {
    debug_assert!(
        box_length.iter().all(|&l| l > 0.0),
        "box lengths must be positive"
    );
    debug_assert!(alpha >= 0.0, "alpha must be non-negative");
    debug_assert!(
        accuracy > 0.0 && accuracy < 1.0,
        "accuracy must be in (0, 1)"
    );
    debug_assert!(kappa >= 0.0, "kappa must be non-negative");
    if alpha == 0.0 {
        return 0;
    }
    let l_max = box_length[0].max(box_length[1]).max(box_length[2]);
    let neg_ln_eps = -accuracy.ln();
    let k_c_sq = (4.0 * alpha * alpha * neg_ln_eps - kappa * kappa).max(0.0);
    if k_c_sq == 0.0 {
        return 0;
    }
    let n_max_f = k_c_sq.sqrt() * l_max / (2.0 * PI);
    (n_max_f.ceil() as u32).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alpha_and_n_max() {
        let alpha = ewald_alpha(5.0, 1e-5, 0.0);
        assert!(alpha > 0.5 && alpha < 1.0);
        let n_max = ewald_n_max(&[10.0; 3], alpha, 1e-5, 0.0);
        assert!(n_max >= 5 && n_max <= 12);
    }

    #[test]
    fn test_higher_accuracy_needs_more_kvectors() {
        let (alpha_low, alpha_high) = (ewald_alpha(5.0, 1e-3, 0.0), ewald_alpha(5.0, 1e-7, 0.0));
        let n_low = ewald_n_max(&[10.0; 3], alpha_low, 1e-3, 0.0);
        let n_high = ewald_n_max(&[10.0; 3], alpha_high, 1e-7, 0.0);
        assert!(n_high >= n_low);
        assert!(alpha_high >= alpha_low);
    }

    #[test]
    fn test_non_cubic_box() {
        let alpha = ewald_alpha(5.0, 1e-5, 0.0);
        let n_max = ewald_n_max(&[10.0, 10.0, 20.0], alpha, 1e-5, 0.0);
        let n_max_cubic = ewald_n_max(&[10.0; 3], alpha, 1e-5, 0.0);
        assert!(n_max >= n_max_cubic);
    }

    #[test]
    fn test_yukawa_reduces_alpha() {
        let alpha_c = ewald_alpha(15.0, 1e-3, 0.0);
        let kappa = 1.0 / 30.0;
        let alpha_y = ewald_alpha(15.0, 1e-3, kappa);
        assert!(alpha_y < alpha_c);
        assert!(alpha_y > 0.0);
    }

    #[test]
    fn test_yukawa_reduces_n_max() {
        let kappa = 0.2;
        let alpha_c = ewald_alpha(15.0, 1e-3, 0.0);
        let alpha_y = ewald_alpha(15.0, 1e-3, kappa);
        let n_c = ewald_n_max(&[400.0; 3], alpha_c, 1e-3, 0.0);
        let n_y = ewald_n_max(&[400.0; 3], alpha_y, 1e-3, kappa);
        assert!(
            n_y < n_c,
            "Yukawa n_max={n_y} should be < Coulomb n_max={n_c}"
        );
    }

    #[test]
    fn test_very_short_range_yukawa_skips_reciprocal() {
        let kappa = 10.0;
        let alpha = ewald_alpha(5.0, 1e-3, kappa);
        assert_eq!(alpha, 0.0);
        let n_max = ewald_n_max(&[10.0; 3], alpha, 1e-3, kappa);
        assert_eq!(n_max, 0);
    }
}
