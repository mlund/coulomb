use crate::pairwise::{
    MultipoleEnergy, MultipoleField, MultipoleForce, MultipolePotential, SelfEnergyPrefactors,
    ShortRangeFunction,
};
use crate::{math::erf_x, math::erfc_x, Cutoff};

/// Truncated Ewald summation scheme
/// 
/// From the abstract of <https://doi.org/dsd6>:
/// 
/// _We present the widespread Ewald summation method in a new light
/// by utilizing a truncated Gaussian screening charge distribution.
/// This choice entails an exact formalism, also as particle mesh Ewald,
/// which in practice is not always the case when using a Gaussian screening function.
/// The presented approach reduces the number of dependent parameters compared to a Gaussian
/// and, for an infinite reciprocal space cutoff, makes the screening charge distribution
/// width truly arbitrary. As such, this arbitrary variable becomes an ideal tool for
/// computational optimization while maintaining accuracy, which is in contrast to when a
/// Gaussian is used._
///
#[derive(Debug, Clone)]
pub struct EwaldTruncated {
    /// Cutoff radius
    cutoff: f64,
    /// Reduced alpha = alpha * cutoff
    eta: f64,
    /// erfc(eta)
    erfc_eta: f64,
    /// exp(-eta^2)
    exp_minus_eta2: f64,
    /// f0 = 1 - erfc(eta) - 2 * eta / sqrt(pi) * exp(-eta^2)
    f0: f64,
}

impl EwaldTruncated {
    /// Inverse square root of pi, 1/sqrt(pi)
    const FRAC_1_SQRT_PI: f64 = 0.5 * core::f64::consts::FRAC_2_SQRT_PI;

    pub fn new(cutoff: f64, alpha: f64) -> Self {
        let eta = alpha * cutoff;
        let f0 = 1.0 - erfc_x(eta) - 2.0 * eta * Self::FRAC_1_SQRT_PI * (-eta * eta).exp();
        Self {
            cutoff,
            eta,
            erfc_eta: erfc_x(eta),
            exp_minus_eta2: (-eta * eta).exp(),
            f0,
        }
    }
}

impl Cutoff for EwaldTruncated {
    fn cutoff(&self) -> f64 {
        self.cutoff
    }
}

impl ShortRangeFunction for EwaldTruncated {
    const URL: &'static str = "https://doi.org/dsd6";

    fn self_energy_prefactors(&self) -> SelfEnergyPrefactors {
        let c1 = -self.eta * Self::FRAC_1_SQRT_PI * (1.0 - self.exp_minus_eta2) / self.f0;
        let c2 = -2.0 * self.eta.powi(3)
            / (3.0
                * (erf_x(self.eta) / Self::FRAC_1_SQRT_PI - 2.0 * self.eta * self.exp_minus_eta2));
        SelfEnergyPrefactors {
            monopole: Some(c1),
            dipole: Some(c2),
        }
    }
    fn kappa(&self) -> Option<f64> {
        None
    }
    fn short_range_f0(&self, q: f64) -> f64 {
        (erfc_x(self.eta * q)
            - self.erfc_eta
            - (1.0 - q) * 2.0 * self.eta * Self::FRAC_1_SQRT_PI * self.exp_minus_eta2)
            / self.f0
    }
    fn short_range_f1(&self, q: f64) -> f64 {
        -2.0 * self.eta
            * ((-(self.eta * q).powi(2)).exp() - self.exp_minus_eta2)
            * Self::FRAC_1_SQRT_PI
            / self.f0
    }
    fn short_range_f2(&self, q: f64) -> f64 {
        4.0 * self.eta.powi(3) * q * (-(self.eta * q).powi(2)).exp() * Self::FRAC_1_SQRT_PI
            / self.f0
    }
    fn short_range_f3(&self, q: f64) -> f64 {
        -8.0 * ((self.eta * q).powi(2) - 0.5)
            * self.eta.powi(3)
            * (-(self.eta * q).powi(2)).exp()
            * Self::FRAC_1_SQRT_PI
            / self.f0
    }
}

impl MultipoleEnergy for EwaldTruncated {}
impl MultipoleField for EwaldTruncated {}
impl MultipoleForce for EwaldTruncated {}
impl MultipolePotential for EwaldTruncated {}

#[test]
fn test_truncated_ewald() {
    use approx::assert_relative_eq;
    let pot = EwaldTruncated::new(29.0, 0.1);
    let eps = 1e-9;
    assert_relative_eq!(pot.short_range_f0(0.5), 0.03993019621374575, epsilon = eps);
    assert_relative_eq!(pot.short_range_f1(0.5), -0.39929238172082965, epsilon = eps);
    assert_relative_eq!(pot.short_range_f2(0.5), 3.364180431728417, epsilon = eps);
    assert_relative_eq!(pot.short_range_f3(0.5), -21.56439656737916, epsilon = eps);
    assert_relative_eq!(
        pot.self_energy(&[2.0], &[0.0]),
        -0.22579937362074382,
        epsilon = eps
    );
    assert_relative_eq!(
        pot.self_energy(&[0.0], &[f64::sqrt(2.0)]),
        -0.0007528321650,
        epsilon = eps
    );
}
