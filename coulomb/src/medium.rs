use crate::*;

use anyhow::Result;
use core::fmt::{Display, Formatter};

/// # Implicit solvent medium such as water or a salt solution
///
/// Stores the following properties from which ionic strength,
/// Debye and Bjerrum lengths can be obtained through traits:
///
/// - Relative permittivity
/// - Salt type
/// - Salt molarity
/// - Temperature
///
/// # Examples
///
/// ## Pure water
/// ~~~
/// use coulomb::{Medium, Salt, DebyeLength, RelativePermittivity, IonicStrength};
/// let medium = Medium::neat_water(298.15);
/// assert_eq!(medium.permittivity(298.15).unwrap(), 78.35565171480539);
/// assert_eq!(medium.ionic_strength(), 0.0);
/// assert!(medium.debye_length().is_none());
/// ~~~
/// ## Salty water
/// ~~~
/// # use coulomb::{Medium, Salt, DebyeLength, IonicStrength};
/// let medium = Medium::salt_water(298.15, Salt::CalciumChloride, 0.1);
/// approx::assert_abs_diff_eq!(medium.ionic_strength(), 0.3);
/// approx::assert_abs_diff_eq!(medium.debye_length().unwrap(), 5.548902662386284);
/// ~~~
pub struct Medium {
    /// Relative permittivity of the medium
    permittivity: Box<dyn RelativePermittivity>,
    /// Salt type
    salt: Option<Salt>,
    /// Salt molarity in mol/l
    molarity: f64,
    /// Temperature in Kelvin
    temperature: f64,
}

impl DebyeLength for Medium {}

impl Medium {
    /// Creates a new medium
    pub fn new(
        temperature: f64,
        permittivity: Box<dyn RelativePermittivity>,
        molarity: f64,
        salt: Option<Salt>,
    ) -> Self {
        Self {
            permittivity,
            salt,
            molarity,
            temperature,
        }
    }
    /// Medium with neat water using the `PermittivityNR::WATER` model
    pub fn neat_water(temperature: f64) -> Self {
        Self {
            permittivity: Box::new(EmpiricalPermittivity::WATER),
            salt: None,
            molarity: 0.0,
            temperature,
        }
    }
    /// Medium with salt water using the `PermittivityNR::WATER` model
    pub fn salt_water(temperature: f64, salt: Salt, molarity: f64) -> Self {
        Self {
            permittivity: Box::new(EmpiricalPermittivity::WATER),
            salt: Some(salt),
            molarity,
            temperature,
        }
    }
    /// Change the molarity of the salt solution. Error if no salt type is defined.
    pub fn set_molarity(&mut self, molality: f64) -> Result<()> {
        if self.salt.is_some() {
            self.molarity = molality;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Cannot set molarity without a salt"))
        }
    }
    /// Bjerrum length in angstrom, lB = e²/4πεkT
    pub fn bjerrum_length(&self) -> f64 {
        bjerrum_length(
            self.temperature,
            self.permittivity.permittivity(self.temperature).unwrap(),
        )
    }
}

impl Display for Medium {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Medium: 𝑇 = {:.2} K, εᵣ = {:.1}, 𝐼 = {:.1} mM, λᴮ = {:.1} Å, λᴰ = {:.1} Å",
            self.temperature,
            self.permittivity.permittivity(self.temperature).unwrap(),
            self.ionic_strength() * 1e3,
            self.bjerrum_length(),
            self.debye_length().unwrap_or(f64::INFINITY),
        )
        .unwrap();
        if self.salt.is_some() {
            write!(f, ", {}", self.salt.as_ref().unwrap()).unwrap()
        };
        Ok(())
    }
}

impl Temperature for Medium {
    fn temperature(&self) -> f64 {
        self.temperature
    }
    fn set_temperature(&mut self, temperature: f64) -> anyhow::Result<()> {
        self.temperature = temperature;
        Ok(())
    }
}
impl RelativePermittivity for Medium {
    fn permittivity(&self, temperature: f64) -> Result<f64> {
        self.permittivity.permittivity(temperature)
    }
}
impl IonicStrength for Medium {
    fn ionic_strength(&self) -> f64 {
        self.salt
            .as_ref()
            .map(|salt| salt.ionic_strength(self.molarity))
            .unwrap_or(0.0)
    }
}
