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

use anyhow::Result;
use core::fmt;
use core::fmt::{Display, Formatter};
use dyn_clone::DynClone;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Trait for objects that has a relative permittivity
pub trait RelativePermittivity: DynClone {
    /// Get the relative permittivity. May error if the temperature is out of range.
    fn permittivity(&self, temperature: f64) -> Result<f64>;

    /// Test is temperature is within range
    fn temperature_is_ok(&self, temperature: f64) -> bool {
        self.permittivity(temperature).is_ok()
    }
}

dyn_clone::clone_trait_object!(RelativePermittivity);

/// Temperature independent relative permittivity, εᵣ = constant
///
/// # Example
/// ~~~
/// use coulomb::{ConstantPermittivity, RelativePermittivity};
/// let dielec = ConstantPermittivity::new(2.0);
/// assert_eq!(dielec.permittivity(298.15).unwrap(), 2.0);
/// assert!(dielec.temperature_is_ok(f64::INFINITY));
///
/// let vacuum = ConstantPermittivity::vacuum();
/// assert_eq!(vacuum.permittivity(298.15).unwrap(), 1.0);
///
/// let perfect_conductor = ConstantPermittivity::perfect_conductor();
/// assert_eq!(perfect_conductor.to_string(), "εᵣ = ∞ for all 𝑇");
/// ~~~
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct ConstantPermittivity {
    permittivity: f64,
}

impl ConstantPermittivity {
    /// New constant permittivity
    pub const fn new(permittivity: f64) -> Self {
        Self { permittivity }
    }
    /// New constant permittivity for vacuum, εᵣ = 1.0
    pub const fn vacuum() -> Self {
        Self::new(1.0)
    }
    /// New _perfect conductor_ with infinity permittivity, εᵣ = ∞
    pub const fn perfect_conductor() -> Self {
        Self::new(f64::INFINITY)
    }
}

impl RelativePermittivity for ConstantPermittivity {
    fn permittivity(&self, _: f64) -> Result<f64> {
        Ok(self.permittivity)
    }
}

impl Display for ConstantPermittivity {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let s = if self.permittivity.is_infinite() {
            "∞".to_string()
        } else {
            format!("{:.2}", self.permittivity)
        };
        write!(f, "εᵣ = {} for all 𝑇", s)
    }
}

/// Empirical model for the temperature dependent relative permittivity, εᵣ(𝑇),
///
/// For more information, see
/// [Neau and Raspo](https://doi.org/10.1016/j.fluid.2019.112371).
///
/// # Example
/// ~~~
/// use coulomb::{EmpiricalPermittivity, RelativePermittivity};
/// assert_eq!(EmpiricalPermittivity::WATER.permittivity(298.15).unwrap(), 78.35565171480539);
/// assert_eq!(EmpiricalPermittivity::METHANOL.permittivity(298.15).unwrap(), 33.081980713895064);
/// assert_eq!(EmpiricalPermittivity::ETHANOL.permittivity(298.15).unwrap(), 24.33523434183735);
/// ~~~
///
/// We can also pretty print the model:
/// ~~~
/// # use coulomb::EmpiricalPermittivity;
/// assert_eq!(EmpiricalPermittivity::WATER.to_string(),
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
    /// Relative permittivity of water
    pub const WATER: EmpiricalPermittivity = EmpiricalPermittivity::new(
        &[-1664.4988, -0.884533, 0.0003635, 64839.1736, 308.3394],
        (273.0, 403.0),
    );
    /// Relative permittivity of methanol
    pub const METHANOL: EmpiricalPermittivity = EmpiricalPermittivity::new(
        &[-1750.3069, -0.99026, 0.0004666, 51360.2652, 327.3124],
        (176.0, 318.0),
    );
    /// Relative permittivity of ethanol
    pub const ETHANOL: EmpiricalPermittivity = EmpiricalPermittivity::new(
        &[-1522.2782, -1.00508, 0.0005211, 38733.9481, 293.1133],
        (288.0, 328.0),
    );
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
