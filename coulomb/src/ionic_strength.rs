use anyhow::Result;

/// Trait for objects that has an ionic strength
pub trait IonicStrength {
    /// Get the ionic strength in mol/l
    ///
    /// The default implementation returns `None`.
    fn ionic_strength(&self) -> Option<f64> {
        None
    }
    /// Try to set the ionic strength in mol/l
    ///
    /// The default implementation returns an error.
    fn set_ionic_strength(&mut self, _ionic_strength: f64) -> Result<()> {
        Err(anyhow::anyhow!(
            "Setting the ionic strength is not implemented"
        ))
    }
}
