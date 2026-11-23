/// Spherical interaction range.
///
/// Callers use `cutoff()` to cull pairs and to size tables, so it must bound the
/// interaction: the energy is zero at and beyond it (∞ if unbounded, 0 for no
/// interaction). A sum of potentials reaches as far as its longest term, so its
/// `cutoff()` is the max of the terms', and so is its `lower_cutoff()`.
pub trait Cutoff {
    /// Squared cutoff distance
    fn cutoff_squared(&self) -> f64 {
        self.cutoff().powi(2)
    }

    /// Distance beyond which the energy is zero
    fn cutoff(&self) -> f64;

    /// Start of the tabulation domain; shorter distances are not sampled
    fn lower_cutoff(&self) -> f64 {
        0.0
    }

    /// Test if cutoff is finite
    fn is_finite(&self) -> bool {
        self.cutoff_squared().is_finite()
    }
}
