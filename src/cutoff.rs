/// Defines a spherical cut-off distance
pub trait Cutoff {
    /// Squared cutoff distance
    fn cutoff_squared(&self) -> f64 {
        self.cutoff().powi(2)
    }

    /// Cutoff distance (upper)
    fn cutoff(&self) -> f64;

    /// Cutoff distance (lower)
    fn lower_cutoff(&self) -> f64 {
        0.0
    }
}
