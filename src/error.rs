/// Errors returned by this crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Cannot resolve stoichiometry; provide both positive and negative ions.
    #[error("cannot resolve stoichiometry; provide both positive and negative ions")]
    Stoichiometry,
    /// Temperature out of range for the permittivity model.
    #[error("temperature out of range for permittivity model")]
    TemperatureOutOfRange,
    /// Molarity must be positive and finite.
    #[error("molarity must be positive and finite")]
    InvalidMolarity,
    /// Cannot set molarity without a salt.
    #[error("cannot set molarity without a salt")]
    MissingSalt,
    /// Operation not supported.
    #[error("operation not supported: {0}")]
    Unsupported(&'static str),
    /// Spline error.
    #[error("spline error: {0}")]
    Spline(&'static str),
}
