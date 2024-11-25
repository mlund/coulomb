// Copyright 2023 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.
// You may obtain a copy of the license at
//
//     http://www.apache.org/licenses/license-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the license is distributed on an "as is" basis,
// without warranties or conditions of any kind, either express or implied.
// See the license for the specific language governing permissions and
// limitations under the license.

use crate::permittivity::RelativePermittivity;
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
/// use coulomb::{Medium, DebyeLength, IonicStrength, Salt};
/// let medium = Medium::neat_water(298.15);
/// assert_eq!(medium.permittivity(), 78.35565171480539);
/// assert!(medium.ionic_strength().is_none());
/// assert!(medium.debye_length().is_none());
/// ~~~
/// ## Salty water
/// ~~~
/// # use coulomb::{Medium, DebyeLength, IonicStrength, Salt};
/// let medium = Medium::salt_water(298.15, Salt::CalciumChloride, 0.1);
/// approx::assert_abs_diff_eq!(medium.ionic_strength().unwrap(), 0.3);
/// approx::assert_abs_diff_eq!(medium.debye_length().unwrap(), 5.548902662386284);
/// ~~~
#[derive(Clone)]
pub struct Medium {
    /// Relative permittivity of the medium
    permittivity: Box<dyn RelativePermittivity>,
    /// Salt type and molarity (mol/l)
    salt: Option<(Salt, f64)>,
    /// Temperature in Kelvin
    temperature: f64,
}

impl DebyeLength for Medium {}

impl Medium {
    /// Creates a new medium
    pub fn new(
        temperature: f64,
        permittivity: Box<dyn RelativePermittivity>,
        salt: Option<(Salt, f64)>,
    ) -> Self {
        Self {
            permittivity,
            salt,
            temperature,
        }
    }
    /// Medium with neat water using the `PermittivityNR::WATER` model
    pub fn neat_water(temperature: f64) -> Self {
        Self {
            permittivity: Box::new(permittivity::WATER),
            salt: None,
            temperature,
        }
    }
    /// Medium with salt water using the `PermittivityNR::WATER` model
    pub fn salt_water(temperature: f64, salt: Salt, molarity: f64) -> Self {
        Self {
            permittivity: Box::new(permittivity::WATER),
            salt: Some((salt, molarity)),
            temperature,
        }
    }

    /// Get molarity of the salt solution, if any
    pub fn molarity(&self) -> Option<f64> {
        self.salt.as_ref().map(|(_, molarity)| molarity).copied()
    }

    /// Change the molarity of the salt solution. Error if no salt type is defined.
    pub fn set_molarity(&mut self, molality: f64) -> Result<()> {
        if molality < 0.0 {
            anyhow::bail!("Molarity must be positive")
        }
        let Some((salt, _)) = &self.salt else {
            anyhow::bail!("Cannot set molarity without a salt")
        };
        self.salt = Some((salt.clone(), molality));
        Ok(())
    }
    /// Bjerrum length in angstrom, lB = e²/4πεkT
    pub fn bjerrum_length(&self) -> f64 {
        bjerrum_length(
            self.temperature,
            self.permittivity.permittivity(self.temperature).unwrap(),
        )
    }

    /// Get relative permittivity of the medium at the current temperature
    pub fn permittivity(&self) -> f64 {
        self.permittivity.permittivity(self.temperature).unwrap()
    }
}

impl Display for Medium {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Medium: 𝑇 = {:.2} K, εᵣ = {:.1}, λᴮ = {:.1} Å",
            self.temperature,
            self.permittivity.permittivity(self.temperature).unwrap(),
            self.bjerrum_length(),
        )?;
        if let Some((salt, molarity)) = &self.salt {
            write!(
                f,
                ", 𝐼 = {:.1} mM, λᴰ = {:.1} Å, {:.1} M {}",
                self.ionic_strength().unwrap() * 1e3,
                self.debye_length().unwrap(),
                molarity,
                salt
            )?;
        };
        Ok(())
    }
}

impl Temperature for Medium {
    fn temperature(&self) -> f64 {
        self.temperature
    }
    /// Set temperature and ensure that it's within the range of the permittivity model
    fn set_temperature(&mut self, temperature: f64) -> anyhow::Result<()> {
        if self.permittivity.temperature_is_ok(temperature) {
            self.temperature = temperature;
            Ok(())
        } else {
            Err(anyhow::anyhow!(
                "Temperature out of range for permittivity model"
            ))
        }
    }
}
impl RelativePermittivity for Medium {
    fn permittivity(&self, temperature: f64) -> Result<f64> {
        self.permittivity.permittivity(temperature)
    }
}

impl IonicStrength for Medium {
    fn ionic_strength(&self) -> Option<f64> {
        self.salt.as_ref().map(|salt| salt.0.ionic_strength(salt.1))
    }
}
