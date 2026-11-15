use core::f64::consts::PI;

/// Ewald splitting parameter α from cutoff and accuracy target.
///
/// Coulomb (κ = 0): α = √(−ln ε) / r_c (Kolafa–Perram).
/// Yukawa (κ > 0): bounds the dimensionless real-space cutoff residual
/// r_c u_real(r_c) of the untruncated unit-charge kernel by ε,
/// returning 0 only when exp(−κr_c) ≤ ε.
/// This is a pair-potential bound, not a system energy or force error bound.
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
    let zeta = kappa * cutoff;
    if zeta >= neg_ln_eps {
        // The bare Yukawa cutoff residual exp(−κr_c) already meets ε.
        return 0.0;
    }
    // Salin & Caillol, JCP 113, 10459 (2000), Eq. 2.17:
    // R = [exp(ζ) erfc(η + b) + exp(−ζ) erfc(η − b)] / 2,
    // where η = αr_c and b = ζ/(2η). For η ≥ b, erfc(x) ≤ exp(−x²)
    // gives R ≤ exp(−η² − b²). Solve η² + ζ²/(4η²) = −ln ε,
    // taking the larger root so both erfc arguments are nonnegative.
    // Normalize the discriminant to avoid squaring −ln ε.
    let ratio = zeta / neg_ln_eps;
    let discriminant = (1.0 - ratio) * (1.0 + ratio);
    let eta = neg_ln_eps.sqrt() * (0.5 * (1.0 + discriminant.sqrt())).sqrt();
    eta / cutoff
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
    n_max_from_k_cut(ewald_k_cut(alpha, accuracy, kappa), box_length, 0.0)
}

/// n_max = ⌈k_c · L_max / 2π − tolerance⌉, at least 1; 0 when k_c = 0.
pub(crate) fn n_max_from_k_cut(k_cut: f64, box_length: &[f64; 3], tolerance: f64) -> u32 {
    if k_cut == 0.0 {
        return 0;
    }
    ((k_cut * longest_side(box_length) / (2.0 * PI) - tolerance).ceil() as u32).max(1)
}

pub(crate) fn longest_side(box_length: &[f64; 3]) -> f64 {
    box_length[0].max(box_length[1]).max(box_length[2])
}

/// k-space cutoff k_c = √max(4α²(−ln ε) − κ², 0); zero for α = 0.
pub(crate) fn ewald_k_cut(alpha: f64, accuracy: f64, kappa: f64) -> f64 {
    (4.0 * alpha * alpha * -accuracy.ln() - kappa * kappa)
        .max(0.0)
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn screened_cutoff_residual_meets_accuracy() {
        use crate::pairwise::{RealSpaceEwald, ShortRangeFunction};

        // Covers the issue #5 regime, the old shortcut boundary ζ = t/2,
        // and values immediately around the corrected boundary ζ = t.
        for cutoff in [5.0, 20.0, 40.0] {
            for accuracy in [1e-12_f64, 1e-6, 1e-4, 1e-3, 0.1, 0.9] {
                let t = -accuracy.ln();
                for fraction in [0.0, 0.01, 0.25, 0.499, 0.5, 0.75, 0.999, 1.0, 1.001, 2.0] {
                    let zeta = fraction * t;
                    let kappa = zeta / cutoff;
                    let alpha = ewald_alpha(cutoff, accuracy, kappa);
                    let residual = if alpha == 0.0 {
                        assert!(kappa * cutoff >= t);
                        (-zeta).exp()
                    } else {
                        let debye_length = (kappa > 0.0).then(|| 1.0 / kappa);
                        let scheme = RealSpaceEwald::from_alpha(cutoff, alpha, debye_length);
                        scheme.short_range_f0(1.0) * (-zeta).exp()
                    };
                    assert!(residual <= accuracy * (1.0 + 1e-6),
                        "cutoff={cutoff}, accuracy={accuracy}, ζ={zeta}, α={alpha}, residual={residual}");
                }
            }
        }
    }

    #[test]
    fn issue_5_keeps_reciprocal_sum() {
        let alpha = ewald_alpha(40.0, 1e-3, 0.095);
        approx::assert_relative_eq!(alpha, 0.06293941653423037, epsilon = 1e-14);
        assert!(ewald_n_max(&[150.0; 3], alpha, 1e-3, 0.095) > 0);
    }

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
