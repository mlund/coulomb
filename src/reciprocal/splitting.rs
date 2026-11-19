//! The Ewald splitting shared by the real- and reciprocal-space parts.

use super::parameters;
use crate::pairwise::{RealSpaceEwald, ShortRangeFunction};
use core::f64::consts::PI;

/// How α and n_max follow a box or policy change.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    /// Re-derived from `accuracy`, as at construction.
    Analytic,
    /// Optimized or user-set α: the real-space part keeps it. `k_cut` sets n_max when an
    /// instance is built; box changes then keep that index set (see [`EwaldSplitting::rebuilt_n_max`]).
    Fixed { k_cut: f64 },
}

/// Ewald splitting parameter α together with the cutoff, accuracy and screening it belongs to.
///
/// One value owns α for both halves of the sum: build the real-space pair potential with
/// [`Self::real_space_scheme`] and the reciprocal part with
/// [`EwaldReciprocal::from_splitting`](super::EwaldReciprocal::from_splitting).
/// It needs no box, so the pair potential can be set up before the simulation cell exists.
///
/// Equality also compares how n_max is chosen; compare [`Self::real_space_scheme`]s to
/// ask whether two splittings give the same pair potential.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EwaldSplitting {
    pub(super) real_cutoff: f64,
    pub(super) accuracy: f64,
    pub(super) kappa: f64,
    /// As given, so the real-space potential sees the same λ_D a caller passed (not 1/κ)
    debye_length: Option<f64>,
    pub(super) alpha: f64,
    mode: Mode,
}

impl EwaldSplitting {
    /// α derived from `accuracy`; see [`EwaldReciprocal::new`](super::EwaldReciprocal::new).
    ///
    /// Pass `debye_length: None` for Coulomb, `Some(λ_D)` for Yukawa.
    pub fn new(real_cutoff: f64, accuracy: f64, debye_length: Option<f64>) -> Self {
        let kappa = kappa(debye_length);
        Self {
            real_cutoff,
            accuracy,
            kappa,
            debye_length,
            alpha: parameters::ewald_alpha(real_cutoff, accuracy, kappa),
            mode: Mode::Analytic,
        }
    }

    /// A fixed α, e.g. from a fitted model; $n_\text{max}$ still follows from `accuracy`.
    ///
    /// Fails unless α is positive and finite.
    pub fn with_alpha(
        real_cutoff: f64,
        accuracy: f64,
        debye_length: Option<f64>,
        alpha: f64,
    ) -> crate::Result<Self> {
        // A new Error variant would break exhaustive matches downstream, so reuse Unsupported
        if !(alpha.is_finite() && alpha > 0.0) {
            return Err(crate::Error::Unsupported(
                "Ewald splitting parameter alpha must be positive and finite",
            ));
        }
        Ok(Self::fixed_alpha(
            real_cutoff,
            accuracy,
            debye_length,
            alpha,
        ))
    }

    /// Fixed α without validation, for the deprecated `EwaldReciprocal::with_alpha`.
    pub(super) fn fixed_alpha(
        real_cutoff: f64,
        accuracy: f64,
        debye_length: Option<f64>,
        alpha: f64,
    ) -> Self {
        let kappa = kappa(debye_length);
        Self {
            real_cutoff,
            accuracy,
            kappa,
            debye_length,
            alpha,
            mode: Mode::Fixed {
                k_cut: parameters::ewald_k_cut(alpha, accuracy, kappa),
            },
        }
    }

    /// Splitting parameter α (Å⁻¹).
    pub const fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Real-space cutoff.
    pub const fn real_cutoff(&self) -> f64 {
        self.real_cutoff
    }

    /// Accuracy target this splitting was chosen for.
    pub const fn accuracy(&self) -> f64 {
        self.accuracy
    }

    /// Debye screening length, or `None` for Coulomb.
    pub const fn debye_length(&self) -> Option<f64> {
        self.debye_length
    }

    /// The real-space pair potential for this α; pair it with the reciprocal part built from
    /// the same splitting.
    pub fn real_space_scheme(&self) -> RealSpaceEwald {
        RealSpaceEwald::from_alpha(self.real_cutoff, self.alpha, self.debye_length())
    }

    /// Short-range function at the cutoff, S(1): the fraction of the screened pair potential
    /// the real-space truncation leaves out.
    pub fn short_range_at_cutoff(&self) -> f64 {
        self.real_space_scheme().short_range_f0(1.0)
    }

    /// True when more than 10% of the pair potential is left beyond the cutoff: the limit
    /// `optimize` will not cross, and a sign that a fixed α is too small.
    /// α = 0 is exempt: screening alone then meets the accuracy target.
    pub fn truncation_too_severe(&self) -> bool {
        self.alpha > 0.0 && self.short_range_at_cutoff() > 0.1
    }

    /// n_max for a new instance in `box_length`. A fixed k-space cutoff is rounded up to whole
    /// shells with a small tolerance, so rebuilding an optimized splitting at its own box
    /// cannot add a shell.
    pub(super) fn n_max(&self, box_length: &[f64; 3]) -> u32 {
        match self.mode {
            Mode::Analytic => {
                parameters::ewald_n_max(box_length, self.alpha, self.accuracy, self.kappa)
            }
            Mode::Fixed { k_cut } => parameters::n_max_from_k_cut(k_cut, box_length, 1e-6),
        }
    }

    /// n_max after an existing instance with `current` shells moves to `box_length`.
    ///
    /// A fixed α keeps its index set and lets |k| scale with the box, so the energy is
    /// smooth in L. Re-deriving from `k_cut` would add a whole shell for any expansion
    /// past the rounding tolerance, since an optimized cutoff sits exactly on a shell.
    pub(super) fn rebuilt_n_max(&self, current: u32, box_length: &[f64; 3]) -> u32 {
        match self.mode {
            Mode::Analytic => self.n_max(box_length),
            Mode::Fixed { .. } => current,
        }
    }

    /// The analytic splitting for the same cutoff, accuracy and screening.
    pub(super) fn analytic(self) -> Self {
        Self {
            alpha: parameters::ewald_alpha(self.real_cutoff, self.accuracy, self.kappa),
            mode: Mode::Analytic,
            ..self
        }
    }

    /// The same splitting at another α, for scanning candidates; in analytic mode n_max then
    /// follows the accuracy formula at that α.
    pub(super) fn at_alpha(self, alpha: f64) -> Self {
        Self { alpha, ..self }
    }

    /// Fix α and the k-space cutoff reached with `n_max` shells in `box_length`.
    pub(super) fn fixed(self, n_max: u32, box_length: &[f64; 3]) -> Self {
        let k_cut = 2.0 * PI * f64::from(n_max) / parameters::longest_side(box_length);
        Self {
            mode: Mode::Fixed { k_cut },
            ..self
        }
    }
}

fn kappa(debye_length: Option<f64>) -> f64 {
    debye_length.map_or(0.0, |d| 1.0 / d)
}
