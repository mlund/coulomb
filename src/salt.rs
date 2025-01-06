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

use core::fmt::{Display, Formatter};
use num_integer::gcd;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// # Common salts and valencies
///
/// Stores the valencies ions in a salt and is used to calculate
/// the ionic strength of arbitrary salt types.
/// Below are some common salts and their valencies:
///
/// Salt      | `valencies`
/// --------- | -----------
/// NaCl      | `[1, -1]`
/// CaCl₂     | `[2, -1]`
/// KAl(SO₄)₂ | `[1, 3, -2]`
///
/// # Examples
/// ~~~
/// use coulomb::Salt;
/// let molarity = 0.1;
///
/// let salt = Salt::SodiumChloride;
/// assert_eq!(salt.valencies(), [1, -1]);
/// assert_eq!(salt.stoichiometry(), [1, 1]);
/// assert_eq!(salt.ionic_strength(molarity), 0.1);
/// ~~~
///
/// We can also define arbitrary salts where the stoichiometry is
/// automatically deduced from the valencies:
///
/// ~~~
/// # use coulomb::Salt;
/// # let molarity = 0.1;
/// let alum = Salt::Custom(vec![1, 3, -2]); // e.g. KAl(SO₄)₂
/// assert_eq!(alum.stoichiometry(), [1, 1, 2]);
/// assert_eq!(alum.ionic_strength(molarity), 0.9);
/// ~~~
///
/// The `Display` trait is implemented to pretty print the salt type:
/// ~~~
/// # use coulomb::Salt;
/// assert_eq!(Salt::PotassiumAlum.to_string(), "🧂Salt = KAl(SO₄)₂");
/// ~~~
///
#[derive(Debug, PartialEq, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub enum Salt {
    /// Sodium chloride, NaCl. This is an example of a 1:1 electrolyte and is the default salt type.
    #[cfg_attr(feature = "serde", serde(rename = "NaCl"))]
    #[default]
    SodiumChloride,
    /// Calcium chloride, CaCl₂
    #[cfg_attr(feature = "serde", serde(rename = "CaCl₂"))]
    CalciumChloride,
    /// Calcium sulfate, CaSO₄
    #[cfg_attr(feature = "serde", serde(rename = "CaSO₄"))]
    CalciumSulfate,
    /// Potassium alum, KAl(SO₄)₂
    #[cfg_attr(feature = "serde", serde(rename = "KAl(SO₄)₂"))]
    PotassiumAlum,
    /// Sodium sulfate, Na₂SO₄
    #[cfg_attr(feature = "serde", serde(rename = "Na₂SO₄"))]
    SodiumSulfate,
    /// Lanthanum chloride, LaCl₃
    #[cfg_attr(feature = "serde", serde(rename = "LaCl₃"))]
    LanthanumChloride,
    /// Salt with custom valencies
    Custom(Vec<isize>),
}

impl Salt {
    /// Valencies of participating ions, zᵢ
    pub fn valencies(&self) -> Vec<isize> {
        match self {
            Salt::SodiumChloride => vec![1, -1],
            Salt::CalciumChloride => vec![2, -1],
            Salt::CalciumSulfate => vec![2, -2],
            Salt::PotassiumAlum => vec![1, 3, -2],
            Salt::SodiumSulfate => vec![1, -2],
            Salt::LanthanumChloride => vec![3, -1],
            Salt::Custom(valencies) => valencies.clone(),
        }
    }

    /// Deduce stoichiometry of the salt, νᵢ
    pub fn stoichiometry(&self) -> Vec<usize> {
        let valencies = self.valencies();
        let sum_positive: isize = valencies.iter().filter(|i| i.is_positive()).sum();
        let sum_negative: isize = valencies.iter().filter(|i| i.is_negative()).sum();
        let gcd = gcd(sum_positive, sum_negative);
        if sum_positive == 0 || sum_negative == 0 || gcd == 0 {
            panic!("cannot resolve stoichiometry; did you provide both + and - ions?")
        }
        valencies
            .iter()
            .map(|valency| {
                ((match valency.is_positive() {
                    true => -sum_negative,
                    false => sum_positive,
                }) / gcd) as usize
            })
            .collect()
    }

    /// Calculate ionic strength from the salt molarity (mol/l), I = ½m∑(νᵢzᵢ²)
    pub fn ionic_strength(&self, molarity: f64) -> f64 {
        0.5 * molarity
            * std::iter::zip(self.valencies(), self.stoichiometry().iter().copied())
                .map(|(valency, nu)| (nu * valency.pow(2) as usize))
                .sum::<usize>() as f64
    }
}

impl Display for Salt {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "🧂Salt = ")?;
        match self {
            Salt::SodiumChloride => write!(f, "NaCl"),
            Salt::CalciumChloride => write!(f, "CaCl₂"),
            Salt::CalciumSulfate => write!(f, "CaSO₄"),
            Salt::PotassiumAlum => write!(f, "KAl(SO₄)₂"),
            Salt::SodiumSulfate => write!(f, "Na₂SO₄"),
            Salt::LanthanumChloride => write!(f, "LaCl₃"),
            Salt::Custom(valencies) => {
                write!(f, "Custom(")?;
                for (i, valency) in valencies.iter().enumerate() {
                    write!(f, "{}", valency)?;
                    if i < valencies.len() - 1 {
                        write!(f, ", ")?;
                    }
                }
                write!(f, ")")
            }
        }
    }
}

#[test]
fn test_salt() {
    let molarity = 0.15;

    // NaCl
    assert_eq!(Salt::SodiumChloride.valencies(), [1, -1]);
    assert_eq!(Salt::SodiumChloride.stoichiometry(), [1, 1]);
    approx::assert_abs_diff_eq!(Salt::SodiumChloride.ionic_strength(molarity), molarity);

    // CaSO₄
    assert_eq!(Salt::CalciumSulfate.valencies(), [2, -2]);
    assert_eq!(Salt::CalciumSulfate.stoichiometry(), [1, 1]);
    approx::assert_abs_diff_eq!(
        Salt::CalciumSulfate.ionic_strength(molarity),
        0.5 * (molarity * 4.0 + molarity * 4.0)
    );

    // CaCl₂
    assert_eq!(Salt::CalciumChloride.valencies(), [2, -1]);
    assert_eq!(Salt::CalciumChloride.stoichiometry(), [1, 2]);
    approx::assert_abs_diff_eq!(
        Salt::CalciumChloride.ionic_strength(molarity),
        0.5 * (molarity * 4.0 + 2.0 * molarity)
    );

    // KAl(SO₄)₂
    assert_eq!(Salt::PotassiumAlum.valencies(), [1, 3, -2]);
    assert_eq!(Salt::PotassiumAlum.stoichiometry(), [1, 1, 2]);
    approx::assert_abs_diff_eq!(
        Salt::PotassiumAlum.ionic_strength(molarity),
        0.5 * (molarity * 1.0 + molarity * 9.0 + 2.0 * molarity * 4.0)
    );
}
