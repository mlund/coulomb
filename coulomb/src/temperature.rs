use anyhow::Result;

/// Trait for objects with a temperature
pub trait Temperature {
    /// Get the temperature in Kelvin
    fn temperature(&self) -> f64;
    /// Set the temperature in Kelvin.
    ///
    /// The default implementation returns an error.
    fn set_temperature(&mut self, _temperature: f64) -> Result<()> {
        Err(anyhow::anyhow!(
            "Setting the temperature is not implemented"
        ))
    }
}
