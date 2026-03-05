use core::f64::consts::PI;

/// Result of Ewald parameter optimization.
pub(super) struct OptimizedParameters {
    pub alpha: f64,
    pub real_cutoff: f64,
    pub n_max: u32,
}

/// Analytical Ewald parameter selection for a given accuracy target.
///
/// Uses the Kolafa–Perram error estimates:
/// - `r_c = L_min / 2` (maximum cutoff under minimum image convention)
/// - `α = √(−ln ε) / r_c` (balances real-space error ~ exp(−α²r_c²))
/// - `n_max = ⌈α·L_max/π · √(−ln ε)⌉` (reciprocal-space convergence)
///
/// For Yukawa (κ > 0), Salin & Caillol (<https://doi.org/10.1063/1.1326477>)
/// proved that the optimal splitting parameter is independent of κ (Eq. 3.3),
/// so the same parameters work for both Coulomb and Yukawa potentials.
pub(super) fn optimize_ewald_parameters(
    box_length: &[f64; 3],
    accuracy: f64,
) -> OptimizedParameters {
    debug_assert!(
        accuracy > 0.0 && accuracy < 1.0,
        "accuracy must be in (0, 1)"
    );
    debug_assert!(
        box_length.iter().all(|&l| l > 0.0),
        "box lengths must be positive"
    );

    let l_min = box_length[0].min(box_length[1]).min(box_length[2]);
    let l_max = box_length[0].max(box_length[1]).max(box_length[2]);
    let neg_ln_eps = -accuracy.ln(); // always positive since ε < 1

    // Real-space cutoff: half the shortest box side (minimum image convention limit)
    let real_cutoff = l_min / 2.0;

    // Splitting parameter: ensures real-space error ~ exp(−α²r_c²) ≈ ε
    let alpha = neg_ln_eps.sqrt() / real_cutoff;

    // Reciprocal cutoff: ensures k-space error ~ exp(−k_min²/(4α²)) ≈ ε
    // For non-cubic boxes, scale by L_max/L_min to cover the longest axis.
    let n_max_f = alpha * l_max / PI * neg_ln_eps.sqrt();
    let n_max = (n_max_f.ceil() as u32).max(1);

    OptimizedParameters {
        alpha,
        real_cutoff,
        n_max,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimize_returns_valid_parameters() {
        let params = optimize_ewald_parameters(&[10.0, 10.0, 10.0], 1e-5);
        assert!(params.alpha > 0.0);
        assert_eq!(params.real_cutoff, 5.0); // L_min / 2
        assert!(params.n_max >= 1);
    }

    #[test]
    fn test_higher_accuracy_needs_more_kvectors() {
        let params_low = optimize_ewald_parameters(&[10.0, 10.0, 10.0], 1e-3);
        let params_high = optimize_ewald_parameters(&[10.0, 10.0, 10.0], 1e-7);
        assert!(params_high.n_max >= params_low.n_max);
        assert!(params_high.alpha >= params_low.alpha);
    }

    #[test]
    fn test_non_cubic_box() {
        let params = optimize_ewald_parameters(&[10.0, 10.0, 20.0], 1e-5);
        // r_c based on shortest side
        assert_eq!(params.real_cutoff, 5.0);
        // n_max should be larger than cubic due to L_max/L_min scaling
        let params_cubic = optimize_ewald_parameters(&[10.0, 10.0, 10.0], 1e-5);
        assert!(params.n_max >= params_cubic.n_max);
    }

    #[test]
    fn test_reference_parameters() {
        // For L=10, ε≈1e-5: α ≈ √(11.5)/5 ≈ 0.678, n_max ≈ ⌈0.678·10/π·√11.5⌉ = ⌈7.3⌉ = 8
        let params = optimize_ewald_parameters(&[10.0; 3], 1e-5);
        assert!(params.alpha > 0.5 && params.alpha < 1.0);
        assert!(params.n_max >= 5 && params.n_max <= 12);
    }
}
