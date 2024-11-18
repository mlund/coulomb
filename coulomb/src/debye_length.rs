use crate::ionic_strength::IonicStrength;
use crate::{permittivity, Temperature, ANGSTROM_PER_METER, LITER_PER_ANGSTROM3};
use physical_constants::{
    AVOGADRO_CONSTANT, BOLTZMANN_CONSTANT, ELEMENTARY_CHARGE, VACUUM_ELECTRIC_PERMITTIVITY,
};
use std::f64::consts::PI;

/// Trait for objects where a Debye length can be calculated
pub trait DebyeLength: IonicStrength + permittivity::RelativePermittivity + Temperature {
    /// # Debye length in angstrom or `None` if the ionic strength is zero.
    ///
    /// May perform expensive operations so avoid use in speed critical code,
    /// such as inside tight interaction loops.
    fn debye_length(&self) -> Option<f64> {
        let temperature = self.temperature();
        let permittivity = self.permittivity(temperature).unwrap();
        self.ionic_strength()
            .map(|i| debye_length(temperature, permittivity, i))
    }
    /// Inverse Debye length in inverse angstrom or `None` if the ionic strength is zero.
    ///
    /// May perform expensive operations so avoid use in speed critical code,
    /// such as inside tight interaction loops.
    fn kappa(&self) -> Option<f64> {
        self.debye_length().map(f64::recip)
    }
}

/// Calculates the Bjerrum length, λ𝐵 = e²/4πε𝑘𝑇 commonly used in electrostatics (ångström).
///
/// More information at <https://en.wikipedia.org/wiki/Bjerrum_length>.
///
/// # Examples
/// ~~~
/// use coulomb::bjerrum_length;
/// let lB = bjerrum_length(293.0, 80.0); // angstroms
/// assert_eq!(lB, 7.1288799871283);
/// ~~~
pub const fn bjerrum_length(kelvin: f64, relative_permittivity: f64) -> f64 {
    ELEMENTARY_CHARGE * ELEMENTARY_CHARGE * ANGSTROM_PER_METER
        / (4.0
            * PI
            * relative_permittivity
            * VACUUM_ELECTRIC_PERMITTIVITY
            * BOLTZMANN_CONSTANT
            * kelvin)
}

/// Calculates the Debye length in angstrom, λ𝐷 = 1/√(8π·λ𝐵·𝐼·𝑁𝐴·𝑉), where 𝐼 is the ionic strength in molar units (mol/l).
///
/// # Examples
/// ~~~
/// use coulomb::debye_length;
/// let molarity = 0.03;                              // mol/l
/// let lambda = debye_length(293.0, 80.0, molarity); // angstroms
/// assert_eq!(lambda, 17.576538097378368);
/// ~~~
pub fn debye_length(kelvin: f64, relative_permittivity: f64, ionic_strength: f64) -> f64 {
    (8.0 * PI
        * bjerrum_length(kelvin, relative_permittivity)
        * ionic_strength
        * AVOGADRO_CONSTANT
        * LITER_PER_ANGSTROM3)
        .sqrt()
        .recip()
}
