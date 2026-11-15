use coulomb::reciprocal::{EwaldPolicy, EwaldReciprocal, EwaldSplitting};

const BOX: [f64; 3] = [100.0; 3];

/// Fixed α = 0.05 at r_c = 20 Å, ε = 10⁻³, λ_D = 30 Å.
fn fixed() -> EwaldSplitting {
    EwaldSplitting::with_alpha(20.0, 1e-3, Some(30.0), 0.05).unwrap()
}

/// Two opposite charges, enough to give a non-trivial reciprocal energy.
fn charged_pair(ewald: &mut EwaldReciprocal) -> f64 {
    let (x, y, z, q) = ([10.0, 13.0], [20.0, 20.0], [30.0, 31.0], [1.0, -1.0]);
    ewald.update_all(&x, &y, &z, &q, None, false);
    ewald.energy() + ewald.self_energy(&q)
}

/// The default parameters must not change: same α, n_max and energy as `EwaldReciprocal::new`,
/// for Coulomb, Yukawa, and screening strong enough to drop reciprocal space (α = 0).
#[test]
fn analytic_splitting_matches_new() {
    for debye_length in [None, Some(30.0), Some(1.0)] {
        let splitting = EwaldSplitting::new(20.0, 1e-4, debye_length);
        let mut from = EwaldReciprocal::from_splitting(&splitting, BOX, EwaldPolicy::PBC);
        let mut new = EwaldReciprocal::new(BOX, 20.0, 1e-4, debye_length);
        assert_eq!(splitting.alpha(), new.alpha(), "{debye_length:?}");
        assert_eq!(from.alpha(), new.alpha());
        assert_eq!(from.n_max(), new.n_max());
        assert_eq!(charged_pair(&mut from), charged_pair(&mut new));
    }
    assert_eq!(EwaldSplitting::new(20.0, 1e-4, Some(1.0)).alpha(), 0.0);
}

#[test]
fn fixed_alpha_must_be_positive_and_finite() {
    for alpha in [0.0, -0.1, f64::NAN, f64::INFINITY] {
        assert!(
            EwaldSplitting::with_alpha(20.0, 1e-3, Some(30.0), alpha).is_err(),
            "alpha = {alpha}"
        );
    }
    let splitting = fixed();
    assert_eq!(splitting.alpha(), 0.05);
}

/// The real-space potential comes from the splitting, so both halves share one α.
#[test]
fn real_space_scheme_follows_splitting() {
    let splitting = fixed();
    let ewald = EwaldReciprocal::from_splitting(&splitting, BOX, EwaldPolicy::PBC);
    assert_eq!(splitting.real_space_scheme(), ewald.real_space_scheme());
    assert_eq!(ewald.splitting(), splitting);
}

/// The splitting in use survives box and policy changes unchanged.
#[test]
fn splitting_survives_rescale_and_policy() {
    for splitting in [EwaldSplitting::new(20.0, 1e-4, Some(30.0)), fixed()] {
        let mut ewald = EwaldReciprocal::from_splitting(&splitting, BOX, EwaldPolicy::PBC);
        ewald.update_all(&[1.0], &[1.0], &[1.0], &[1.0], Some([101.0; 3]), false);
        ewald.set_policy(EwaldPolicy::IPBC);
        assert_eq!(ewald.splitting(), splitting);
    }
}

/// The deprecated constructor keeps its old contract: no validation, and n_max from the
/// accuracy formula (away from the 10⁻⁶ rounding tolerance).
#[test]
#[allow(deprecated)]
fn deprecated_with_alpha_is_unvalidated_and_keeps_n_max() {
    let strong = EwaldReciprocal::with_alpha(BOX, 20.0, 1e-4, Some(1.0), 0.0);
    assert_eq!(strong.n_max(), 0);
    let fixed_ewald = EwaldReciprocal::with_alpha(BOX, 20.0, 1e-3, Some(30.0), 0.05);
    assert_eq!(fixed_ewald.splitting(), fixed());
    let expected = (f64::sqrt(4.0 * 0.05 * 0.05 * -(1e-3_f64).ln() - 1.0 / 900.0) * 100.0
        / (2.0 * std::f64::consts::PI))
        .ceil() as u32;
    assert_eq!(fixed_ewald.n_max(), expected);
}

/// α = 0 means screening alone meets the target, so it is not a severe truncation.
#[test]
fn truncation_guard() {
    assert!(!EwaldSplitting::new(20.0, 1e-4, Some(1.0)).truncation_too_severe());
    assert!(!EwaldSplitting::new(20.0, 1e-4, Some(30.0)).truncation_too_severe());
    let tiny = EwaldSplitting::with_alpha(20.0, 1e-3, Some(30.0), 0.01).unwrap();
    assert!(tiny.truncation_too_severe());
}
