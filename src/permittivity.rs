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

//! Relative permittivity models

use anyhow::Result;
use core::fmt;
use core::fmt::{Display, Formatter};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Trait for objects that has a relative permittivity
pub trait RelativePermittivity {
    /// Relative permittivity or error if temperature is out of range.
    fn permittivity(&self, temperature: f64) -> Result<f64>;

    /// Test if temperature is within range
    fn temperature_is_ok(&self, temperature: f64) -> bool {
        self.permittivity(temperature).is_ok()
    }
    /// Convert to constant permittivity for a given temperature
    fn to_const_permittivity(&self, temperature: f64) -> Result<ConstantPermittivity> {
        Ok(ConstantPermittivity::new(self.permittivity(temperature)?))
    }
    /// Tries to set the permittivity to a constant value
    fn set_permittivity(&mut self, permittivity: f64) -> Result<()> {
        let _ = permittivity;
        anyhow::bail!("Cannot set permittivity for this model")
    }
}

/// Enum for all available permittivity models
#[derive(Debug, PartialEq, Clone)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(deny_unknown_fields)
)]
pub enum Permittivity {
    /// Custom constant permittivity, independent of temperature
    Fixed(f64),
    /// Custom empirical permittivity model with temperature dependence
    Empirical(EmpiricalPermittivity),
    /// Relative permittivity of water using the NR model for temperature dependence
    Water,
    /// Relative permittivity of ethanol using the NR model for temperature dependence
    Ethanol,
    /// Relative permittivity of methanol using the NR model for temperature dependence
    Methanol,
    /// Perfect conductor with infinite permittivity, εᵣ = ∞
    Metal,
    /// Relative permittivity of free space, εᵣ = 1.0
    Vacuum,
    /// Relative permittivity of water at 25 degree Celcius, εᵣ = 78.4
    Water25,
}

impl RelativePermittivity for Permittivity {
    fn permittivity(&self, temperature: f64) -> Result<f64> {
        Box::<dyn RelativePermittivity>::from(self.clone()).permittivity(temperature)
    }
}

impl From<Permittivity> for Box<dyn RelativePermittivity> {
    fn from(model: Permittivity) -> Box<dyn RelativePermittivity> {
        match model {
            Permittivity::Fixed(d) => Box::new(ConstantPermittivity::from(d)),
            Permittivity::Empirical(d) => Box::new(d),
            Permittivity::Water => Box::new(WATER),
            Permittivity::Ethanol => Box::new(ETHANOL),
            Permittivity::Methanol => Box::new(METHANOL),
            Permittivity::Metal => Box::new(METAL),
            Permittivity::Vacuum => Box::new(VACUUM),
            Permittivity::Water25 => Box::new(WATER_25C),
        }
    }
}

impl Display for Permittivity {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Permittivity::Fixed(d) => write!(f, "{}", d),
            Permittivity::Empirical(d) => write!(f, "{}", d),
            Permittivity::Water => write!(f, "{}", WATER),
            Permittivity::Ethanol => write!(f, "{}", ETHANOL),
            Permittivity::Methanol => write!(f, "{}", METHANOL),
            Permittivity::Metal => write!(f, "{}", METAL),
            Permittivity::Vacuum => write!(f, "{}", VACUUM),
            Permittivity::Water25 => write!(f, "{}", WATER_25C),
        }
    }
}

/// Perfect conductor with infinite permittivity, εᵣ = ∞
pub const METAL: ConstantPermittivity = ConstantPermittivity::new(f64::INFINITY);

/// Relative permittivity of water at 25 degree Celcius, εᵣ = 78.4
pub const WATER_25C: ConstantPermittivity = ConstantPermittivity::new(78.4);

/// Relative permittivity of free space, εᵣ = 1.0
pub const VACUUM: ConstantPermittivity = ConstantPermittivity::new(1.0);

/// Relative permittivity of water, εᵣ(𝑇)
///
/// See <https://doi.org/ggddkk>.
pub const WATER: EmpiricalPermittivity = EmpiricalPermittivity::new(
    &[-1664.4988, -0.884533, 0.0003635, 64839.1736, 308.3394],
    (273.0, 403.0),
);
/// Relative permittivity of methanol, εᵣ(𝑇)
///
/// See <https://doi.org/ggddkk>.
pub const METHANOL: EmpiricalPermittivity = EmpiricalPermittivity::new(
    &[-1750.3069, -0.99026, 0.0004666, 51360.2652, 327.3124],
    (176.0, 318.0),
);
/// Relative permittivity of ethanol, εᵣ(𝑇)
///
/// See <https://doi.org/ggddkk>.
pub const ETHANOL: EmpiricalPermittivity = EmpiricalPermittivity::new(
    &[-1522.2782, -1.00508, 0.0005211, 38733.9481, 293.1133],
    (288.0, 328.0),
);

/// Temperature independent relative permittivity, εᵣ = constant
///
/// # Example
/// ~~~
/// use coulomb::permittivity::*;
/// let dielec = ConstantPermittivity::new(2.0);
/// assert_eq!(dielec.permittivity(298.15).unwrap(), 2.0);
/// assert!(dielec.temperature_is_ok(f64::INFINITY));
///
/// assert_eq!(VACUUM.permittivity(298.15).unwrap(), 1.0);
/// assert_eq!(METAL.to_string(), "εᵣ = ∞");
/// ~~~
#[derive(Debug, PartialEq, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ConstantPermittivity {
    #[cfg_attr(feature = "serde", serde(rename = "epsr", alias = "εᵣ"))]
    permittivity: f64,
}

impl ConstantPermittivity {
    /// New constant permittivity
    pub const fn new(permittivity: f64) -> Self {
        Self { permittivity }
    }
}

impl From<ConstantPermittivity> for f64 {
    fn from(d: ConstantPermittivity) -> f64 {
        d.permittivity
    }
}

impl From<f64> for ConstantPermittivity {
    fn from(d: f64) -> ConstantPermittivity {
        ConstantPermittivity::new(d)
    }
}

impl RelativePermittivity for ConstantPermittivity {
    fn permittivity(&self, _: f64) -> Result<f64> {
        Ok(self.permittivity)
    }
    fn set_permittivity(&mut self, permittivity: f64) -> Result<()> {
        self.permittivity = permittivity;
        Ok(())
    }
}

impl Display for ConstantPermittivity {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "εᵣ = {}",
            match self.permittivity.is_infinite() {
                true => "∞".to_string(),
                false => format!("{:.2}", self.permittivity),
            }
        )
    }
}

/// Empirical model for the temperature dependent relative permittivity, εᵣ(𝑇),
///
/// For more information, see
/// [Neau and Raspo](https://doi.org/10.1016/j.fluid.2019.112371).
///
/// # Example
/// ~~~
/// use coulomb::permittivity::*;
/// assert_eq!(WATER.permittivity(298.15).unwrap(), 78.35565171480539);
/// assert_eq!(METHANOL.permittivity(298.15).unwrap(), 33.081980713895064);
/// assert_eq!(ETHANOL.permittivity(298.15).unwrap(), 24.33523434183735);
/// ~~~
///
/// We can also pretty print the model:
/// ~~~
/// # use coulomb::permittivity::*;
/// assert_eq!(WATER.to_string(),
///            "εᵣ(𝑇) = -1.66e3 + -8.85e-1𝑇 + 3.63e-4𝑇² + 6.48e4/𝑇 + 3.08e2㏑(𝑇); 𝑇 = [273.0, 403.0]");
/// ~~~
#[derive(Debug, PartialEq, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EmpiricalPermittivity {
    /// Coefficients for the model
    coeffs: [f64; 5],
    /// Closed temperature interval in which the model is valid
    temperature_interval: (f64, f64),
}

impl EmpiricalPermittivity {
    /// Creates a new instance of the NR model
    pub const fn new(coeffs: &[f64; 5], temperature_interval: (f64, f64)) -> EmpiricalPermittivity {
        EmpiricalPermittivity {
            coeffs: *coeffs,
            temperature_interval,
        }
    }
}

impl RelativePermittivity for EmpiricalPermittivity {
    fn permittivity(&self, temperature: f64) -> Result<f64> {
        if temperature < self.temperature_interval.0 || temperature > self.temperature_interval.1 {
            Err(anyhow::anyhow!(
                "Temperature out of range for permittivity model"
            ))
        } else {
            Ok(self.coeffs[0]
                + self.coeffs[1] * temperature
                + self.coeffs[2] * temperature.powi(2)
                + self.coeffs[3] / temperature
                + self.coeffs[4] * temperature.ln())
        }
    }
}

impl Display for EmpiricalPermittivity {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "εᵣ(𝑇) = {:.2e} + {:.2e}𝑇 + {:.2e}𝑇² + {:.2e}/𝑇 + {:.2e}㏑(𝑇); 𝑇 = [{:.1}, {:.1}]",
            self.coeffs[0],
            self.coeffs[1],
            self.coeffs[2],
            self.coeffs[3],
            self.coeffs[4],
            self.temperature_interval.0,
            self.temperature_interval.1
        )
    }
}
