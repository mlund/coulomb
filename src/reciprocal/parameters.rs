use crate::NalgebraVector3;
use core::f64::consts::PI;

/// Result of Ewald parameter optimization.
pub(super) struct OptimizedParameters {
    pub alpha: f64,
    pub real_cutoff: f64,
    pub k_max: u32,
}

/// Optimize Ewald parameters for a given accuracy target.
///
/// Balances real-space and reciprocal-space errors to minimize total work.
/// Uses the error estimates from Kolafa & Perram (1992) for Coulomb,
/// extended for Yukawa following Salin & Caillol (2000).
///
/// - `box_length`: Side lengths of the orthorhombic box
/// - `kappa`: Inverse Debye screening length (0 for Coulomb)
/// - `accuracy`: Target relative energy accuracy (e.g. 1e-5)
/// - `num_particles`: Number of particles (affects cost balance)
pub(super) fn optimize_ewald_parameters(
    box_length: &NalgebraVector3,
    kappa: f64,
    accuracy: f64,
    num_particles: usize,
) -> OptimizedParameters {
    let l_min = box_length.x.min(box_length.y).min(box_length.z);
    let half_l = l_min / 2.0;
    let n = num_particles as f64;

    // For Yukawa, screening reduces both real and reciprocal errors.
    // We use a simple heuristic: start with Coulomb estimates then adjust.

    // Kolafa-Perram style: balance real and reciprocal errors.
    // Real-space error ~ exp(-α² r_c²) / r_c
    // Reciprocal error ~ exp(-π² k_max² / (α² L²)) / L
    // Optimal: α ~ sqrt(π) * (n / V²)^(1/6) * sqrt(ln(1/accuracy))
    // But simpler: scan α to find good balance.

    // Start with a reasonable alpha and iterate
    let volume = box_length.x * box_length.y * box_length.z;

    // Search over alpha to find best parameters
    let mut best_alpha = 1.0;
    let mut best_rc = half_l * 0.5;
    let mut best_kmax = 5_u32;
    let mut best_cost = f64::INFINITY;

    // Scan alpha values (finer grid, wider range)
    for i in 1..=100 {
        let alpha = 0.05 * i as f64;

        // Real-space cutoff: exp(-α²r_c²)/r_c < accuracy
        // r_c = sqrt(-ln(accuracy)) / α
        let rc = (-accuracy.ln()).sqrt() / alpha;
        if rc >= half_l || rc <= 0.0 {
            continue;
        }

        // Reciprocal cutoff from accuracy: exp(-k²/(4α²)) < accuracy for k = 2πn/L
        // k_max such that exp(-(2π k_max / L)² / (4α²)) < accuracy
        // k_max > α L / π × sqrt(-ln(accuracy))
        let k_max_f = alpha * l_min / PI * (-accuracy.ln()).sqrt();
        let k_max = (k_max_f.ceil() as u32).max(1);

        // For Yukawa, the exponential decay is faster, so we can use fewer k-vectors
        let k_max = if kappa > 0.0 {
            let kappa_factor = (-kappa * kappa / (4.0 * alpha * alpha)).exp();
            let k_max_adjusted = (k_max as f64 * kappa_factor.sqrt().max(0.3)) as u32;
            k_max_adjusted.max(1)
        } else {
            k_max
        };

        // Cost estimate: real ~ N × (4/3 π r_c³ / V) × N + recip ~ N × k_max³
        let real_cost = n * (4.0 / 3.0 * PI * rc.powi(3) / volume * n);
        let recip_cost = n * (k_max as f64).powi(3);
        let cost = real_cost + recip_cost;

        // Check that accuracy is met (rough check)
        let real_err =
            (-alpha * alpha * rc * rc).exp() / rc * (-kappa * rc).exp();
        let k_min = 2.0 * PI / l_min;
        let recip_err = (-(k_min * k_min + kappa * kappa) / (4.0 * alpha * alpha)).exp();

        if real_err < accuracy && recip_err < accuracy && cost < best_cost {
            best_cost = cost;
            best_alpha = alpha;
            best_rc = rc;
            best_kmax = k_max;
        }
    }

    OptimizedParameters {
        alpha: best_alpha,
        real_cutoff: best_rc,
        k_max: best_kmax,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimize_returns_valid_parameters() {
        let box_length = NalgebraVector3::new(10.0, 10.0, 10.0);
        let params = optimize_ewald_parameters(&box_length, 0.0, 1e-5, 100);
        assert!(params.alpha > 0.0);
        assert!(params.real_cutoff > 0.0);
        assert!(params.real_cutoff < 5.0); // must be < L/2
        assert!(params.k_max >= 1);
    }

    #[test]
    fn test_yukawa_needs_fewer_kvectors() {
        let box_length = NalgebraVector3::new(10.0, 10.0, 10.0);
        let params_c = optimize_ewald_parameters(&box_length, 0.0, 1e-5, 100);
        let params_y = optimize_ewald_parameters(&box_length, 0.5, 1e-5, 100);
        assert!(params_y.k_max <= params_c.k_max);
    }

    #[test]
    fn test_higher_accuracy_needs_more_kvectors() {
        let box_length = NalgebraVector3::new(10.0, 10.0, 10.0);
        let params_low = optimize_ewald_parameters(&box_length, 0.0, 1e-3, 100);
        let params_high = optimize_ewald_parameters(&box_length, 0.0, 1e-7, 100);
        assert!(params_high.k_max >= params_low.k_max);
    }
}
