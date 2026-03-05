use core::f64::consts::PI;

/// Compute the Ewald splitting parameter α for a given real-space cutoff and accuracy.
///
/// Uses the Kolafa–Perram error estimate where the real-space error scales as
/// exp(−α²r_c²), giving α = √(−ln ε) / r_c.
///
/// For Yukawa (κ > 0), Salin & Caillol (<https://doi.org/10.1063/1.1326477>)
/// proved that the optimal α is independent of κ (Eq. 3.3).
///
/// # Arguments
/// - `cutoff`: Real-space cutoff distance, r_c
/// - `accuracy`: Target relative energy accuracy ε (e.g. `1e-5`), must be in (0, 1)
///
/// # Example
/// ```
/// # use approx::assert_relative_eq;
/// use coulomb::reciprocal::ewald_alpha;
/// let alpha = ewald_alpha(5.0, 1e-5);
/// assert_relative_eq!(alpha, (-1e-5_f64.ln()).sqrt() / 5.0, epsilon = 1e-10);
/// ```
pub fn ewald_alpha(cutoff: f64, accuracy: f64) -> f64 {
    debug_assert!(cutoff > 0.0, "cutoff must be positive");
    debug_assert!(
        accuracy > 0.0 && accuracy < 1.0,
        "accuracy must be in (0, 1)"
    );
    (-accuracy.ln()).sqrt() / cutoff
}

/// Compute the reciprocal-space integer cutoff n_max for given Ewald parameters.
///
/// Uses the Kolafa–Perram estimate:
/// $n_\text{max} = \lceil \alpha \cdot L_\text{max} / \pi \cdot \sqrt{-\ln\varepsilon} \rceil$
///
/// # Arguments
/// - `box_length`: Side lengths `[Lx, Ly, Lz]` of the orthorhombic simulation box
/// - `alpha`: Ewald splitting parameter α
/// - `accuracy`: Target relative energy accuracy ε (e.g. `1e-5`), must be in (0, 1)
///
/// # Example
/// ```
/// use coulomb::reciprocal::{ewald_alpha, ewald_n_max};
/// let alpha = ewald_alpha(5.0, 1e-5);
/// let n_max = ewald_n_max(&[10.0; 3], alpha, 1e-5);
/// assert!(n_max >= 5);
/// ```
pub fn ewald_n_max(box_length: &[f64; 3], alpha: f64, accuracy: f64) -> u32 {
    debug_assert!(
        box_length.iter().all(|&l| l > 0.0),
        "box lengths must be positive"
    );
    debug_assert!(alpha > 0.0, "alpha must be positive");
    debug_assert!(
        accuracy > 0.0 && accuracy < 1.0,
        "accuracy must be in (0, 1)"
    );
    let l_max = box_length[0].max(box_length[1]).max(box_length[2]);
    let neg_ln_eps = -accuracy.ln();
    let n_max_f = alpha * l_max / PI * neg_ln_eps.sqrt();
    (n_max_f.ceil() as u32).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alpha_and_n_max() {
        let alpha = ewald_alpha(5.0, 1e-5);
        assert!(alpha > 0.5 && alpha < 1.0);
        let n_max = ewald_n_max(&[10.0; 3], alpha, 1e-5);
        assert!(n_max >= 5 && n_max <= 12);
    }

    #[test]
    fn test_higher_accuracy_needs_more_kvectors() {
        let (alpha_low, alpha_high) = (ewald_alpha(5.0, 1e-3), ewald_alpha(5.0, 1e-7));
        let n_low = ewald_n_max(&[10.0; 3], alpha_low, 1e-3);
        let n_high = ewald_n_max(&[10.0; 3], alpha_high, 1e-7);
        assert!(n_high >= n_low);
        assert!(alpha_high >= alpha_low);
    }

    #[test]
    fn test_non_cubic_box() {
        let alpha = ewald_alpha(5.0, 1e-5);
        let n_max = ewald_n_max(&[10.0, 10.0, 20.0], alpha, 1e-5);
        let n_max_cubic = ewald_n_max(&[10.0; 3], alpha, 1e-5);
        assert!(n_max >= n_max_cubic);
    }
}
