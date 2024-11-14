/// Defines a spherical cut-off distance
pub trait Cutoff {
    /// Squared cutoff distance
    fn cutoff_squared(&self) -> f64 {
        self.cutoff().powi(2)
    }

    /// Cutoff distance
    fn cutoff(&self) -> f64;
}
