use coulomb::pairwise::{RealSpaceEwald, ShortRangeFunction};

#[test]
fn screened_real_space_cutoff_meets_requested_accuracy() {
    let (cutoff, kappa, accuracy) = (40.0, 0.095, 1e-4);
    let scheme = RealSpaceEwald::new(cutoff, accuracy, Some(1.0 / kappa));
    let residual = scheme.short_range_f0(1.0) * f64::exp(-kappa * cutoff);
    assert!(
        residual <= accuracy,
        "residual={residual}, accuracy={accuracy}"
    );
}

#[test]
fn reciprocal_sum_is_kept_when_bare_yukawa_tail_exceeds_accuracy() {
    let ewald =
        coulomb::reciprocal::EwaldReciprocal::new([150.0; 3], 40.0, 1e-3, Some(1.0 / 0.095));
    assert!(ewald.num_k_vectors() > 0);
}

#[test]
fn screened_cutoff_residual_meets_accuracy() {
    // Covers the issue #5 regime, the old shortcut boundary ζ = t/2,
    // and values immediately around the corrected boundary ζ = t.
    for cutoff in [5.0, 20.0, 40.0] {
        for accuracy in [1e-12_f64, 1e-6, 1e-4, 1e-3, 0.1, 0.9] {
            let t = -accuracy.ln();
            for fraction in [0.0, 0.01, 0.25, 0.499, 0.5, 0.75, 0.999, 1.0, 1.001, 2.0] {
                let zeta = fraction * t;
                let kappa = zeta / cutoff;
                let debye_length = (kappa > 0.0).then(|| 1.0 / kappa);
                let scheme = RealSpaceEwald::new(cutoff, accuracy, debye_length);
                let residual = scheme.short_range_f0(1.0) * (-zeta).exp();
                assert!(
                    residual <= accuracy * (1.0 + 1e-6),
                    "cutoff={cutoff}, accuracy={accuracy}, ζ={zeta}, residual={residual}"
                );
            }
        }
    }
}
