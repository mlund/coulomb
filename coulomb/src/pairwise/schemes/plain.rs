// Copyright 2023 Björn Stenqvist and Mikael Lund
//
// Converted to Rust with modification from the C++ library "CoulombGalore":
// https://zenodo.org/doi/10.5281/zenodo.3522058
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

use crate::pairwise::ShortRangeFunction;

#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Scheme for vanilla Coulomb interactions, $S(q)=1$.
///
/// See _Premier mémoire sur l’électricité et le magnétisme_ by Charles-Augustin de Coulomb,
/// <https://doi.org/msxd>.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Plain {
    /// Cut-off distance
    cutoff: f64,
    /// Optional inverse Debye length
    #[cfg_attr(
        feature = "serde",
        serde(
            rename = "debye",
            alias = "debye_length",
            alias = "debyelength",
            serialize_with = "serialize_reciprocal",
            deserialize_with = "deserialize_reciprocal",
            default
        )
    )]
    kappa: Option<f64>,
}

/// Convert from kappa to debye when serializing.
#[cfg(feature = "serde")]
fn serialize_reciprocal<S>(x: &Option<f64>, s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    match x.as_ref() {
        Some(&value) => s.serialize_some(&f64::recip(value)),
        None => s.serialize_none(),
    }
}

/// Convert from debye to kappa when serializing.
#[cfg(feature = "serde")]
fn deserialize_reciprocal<'de, D>(deserializer: D) -> Result<Option<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(Option::deserialize(deserializer)?.map(f64::recip))
}

impl Default for Plain {
    /// The default is infinite cutoff radius and no screening
    fn default() -> Self {
        Self {
            cutoff: f64::INFINITY,
            kappa: None,
        }
    }
}

impl core::fmt::Display for Plain {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Plain Coulomb: 𝑟✂ = {:.1} Å", self.cutoff)?;
        if let Some(debye_length) = self.kappa.map(f64::recip) {
            write!(f, ", λᴰ = {:.1} Å", debye_length)?;
        }
        write!(f, " <{}>", Self::url())?;
        Ok(())
    }
}

impl Plain {
    pub fn without_cutoff() -> Self {
        Self::new(f64::INFINITY, None)
    }
    pub fn new(cutoff: f64, debye_length: Option<f64>) -> Self {
        Self {
            cutoff,
            kappa: debye_length.map(f64::recip),
        }
    }
    pub fn new_without_salt(cutoff: f64) -> Self {
        Self::new(cutoff, None)
    }
}

impl crate::Cutoff for Plain {
    #[inline]
    fn cutoff(&self) -> f64 {
        self.cutoff
    }
}

impl ShortRangeFunction for Plain {
    fn url() -> &'static str {
        "https://doi.org/msxd"
    }

    #[inline]
    fn kappa(&self) -> Option<f64> {
        self.kappa
    }
    #[inline]
    fn short_range_f0(&self, _q: f64) -> f64 {
        1.0
    }
    #[inline]
    fn short_range_f1(&self, _q: f64) -> f64 {
        0.0
    }
    #[inline]
    fn short_range_f2(&self, _q: f64) -> f64 {
        0.0
    }
    #[inline]
    fn short_range_f3(&self, _q: f64) -> f64 {
        0.0
    }
}

#[test]
fn test_coulomb() {
    use crate::{Matrix3, Vector3};
    use approx::assert_relative_eq;

    use crate::pairwise::{MultipoleEnergy, MultipoleField, MultipoleForce, MultipolePotential};
    let cutoff: f64 = 29.0; // cutoff distance
    let z1 = 2.0; // charge
    let z2 = 3.0; // charge
    let mu1 = Vector3::new(19.0, 7.0, 11.0); // dipole moment
    let mu2 = Vector3::new(13.0, 17.0, 5.0); // dipole moment
    let quad1 = Matrix3::new(3.0, 7.0, 8.0, 5.0, 9.0, 6.0, 2.0, 1.0, 4.0); // quadrupole moment
    let _quad2 = Matrix3::zeros(); // quadrupole moment
    let r = Vector3::new(23.0, 0.0, 0.0); // distance vector
    let rq = Vector3::new(
        5.75 * (6.0f64).sqrt(),
        5.75 * (2.0f64).sqrt(),
        11.5 * (2.0f64).sqrt(),
    ); // distance vector for quadrupole check
    let rh = Vector3::new(1.0, 0.0, 0.0); // normalized distance vector

    let pot = Plain::new(cutoff, None);
    let eps = 1e-9;

    assert_eq!(
        pot.to_string(),
        "Plain Coulomb: 𝑟✂ = 29.0 Å <https://doi.org/msxd>"
    );

    // Test short-ranged function
    assert_eq!(pot.short_range_f0(0.5), 1.0);
    assert_eq!(pot.short_range_f1(0.5), 0.0);
    assert_eq!(pot.short_range_f2(0.5), 0.0);
    assert_eq!(pot.short_range_f3(0.5), 0.0);

    // Test potentials
    assert_eq!(pot.ion_potential(z1, cutoff + 1.0), 0.0);
    assert_relative_eq!(
        pot.ion_potential(z1, r.norm()),
        0.08695652173913043,
        epsilon = eps
    );
    assert_relative_eq!(
        pot.dipole_potential(&mu1, &((cutoff + 1.0) * rh)),
        0.0,
        epsilon = eps
    );
    assert_relative_eq!(
        pot.dipole_potential(&mu1, &r),
        0.035916824196597356,
        epsilon = eps
    );
    assert_relative_eq!(
        pot.quadrupole_potential(&quad1, &rq),
        0.00093632817,
        epsilon = eps
    );

    // Test fields
    assert_relative_eq!(
        pot.ion_field(z1, &((cutoff + 1.0) * rh)).norm(),
        0.0,
        epsilon = eps
    );
    let ion_field = pot.ion_field(z1, &r);
    assert_relative_eq!(ion_field[0], 0.003780718336, epsilon = eps);
    assert_relative_eq!(ion_field.norm(), 0.003780718336, epsilon = eps);
    assert_relative_eq!(
        pot.dipole_field(&mu1, &((cutoff + 1.0) * rh)).norm(),
        0.0,
        epsilon = eps
    );
    let dip_field = pot.dipole_field(&mu1, &r);
    assert_relative_eq!(dip_field[0], 0.003123202104, epsilon = eps);
    assert_relative_eq!(dip_field[1], -0.0005753267034, epsilon = eps);
    assert_relative_eq!(dip_field[2], -0.0009040848196, epsilon = eps);
    let quad_field = pot.quadrupole_field(&quad1, &r);
    assert_relative_eq!(quad_field[0], -0.00003752130674, epsilon = eps);
    assert_relative_eq!(quad_field[1], -0.00006432224013, epsilon = eps);
    assert_relative_eq!(quad_field[2], -0.00005360186677, epsilon = eps);

    // Test energies
    approx::assert_relative_eq!(pot.ion_ion_energy(z1, z2, cutoff + 1.0), 0.0, epsilon = eps);
    approx::assert_relative_eq!(
        pot.ion_ion_energy(z1, z2, r.norm()),
        z1 * z2 / r.norm(),
        epsilon = eps
    );
    approx::assert_relative_eq!(
        pot.ion_dipole_energy(z1, &mu2, &((cutoff + 1.0) * rh)),
        -0.0,
        epsilon = eps
    );
    approx::assert_relative_eq!(
        pot.ion_dipole_energy(z1, &mu2, &r),
        -0.04914933837,
        epsilon = eps
    );
    approx::assert_relative_eq!(
        pot.dipole_dipole_energy(&mu1, &mu2, &((cutoff + 1.0) * rh)),
        -0.0,
        epsilon = eps
    );
    approx::assert_relative_eq!(
        pot.dipole_dipole_energy(&mu1, &mu2, &r),
        -0.02630064930,
        epsilon = eps
    );
    approx::assert_relative_eq!(
        pot.ion_quadrupole_energy(z2, &quad1, &rq),
        0.002808984511,
        epsilon = eps
    );

    // Test forces
    assert_relative_eq!(
        pot.ion_ion_force(z1, z2, &((cutoff + 1.0) * rh)).norm(),
        0.0,
        epsilon = eps
    );
    let force = pot.ion_ion_force(z1, z2, &r);
    assert_relative_eq!(force[0], 0.01134215501, epsilon = eps);
    assert_relative_eq!(force.norm(), 0.01134215501, epsilon = eps);
    assert_relative_eq!(
        pot.ion_dipole_force(z2, &mu1, &((cutoff + 1.0) * rh))
            .norm(),
        0.0,
        epsilon = eps
    );
    let force = pot.ion_dipole_force(z2, &mu1, &r);
    assert_relative_eq!(force[0], 0.009369606312, epsilon = eps);
    assert_relative_eq!(force[1], -0.001725980110, epsilon = eps);
    assert_relative_eq!(force[2], -0.002712254459, epsilon = eps);
    assert_relative_eq!(
        pot.dipole_dipole_force(&mu1, &mu2, &((cutoff + 1.0) * rh))
            .norm(),
        0.0,
        epsilon = eps
    );
    let force = pot.dipole_dipole_force(&mu1, &mu2, &r);
    assert_relative_eq!(force[0], 0.003430519474, epsilon = eps);
    assert_relative_eq!(force[1], -0.004438234569, epsilon = eps);
    assert_relative_eq!(force[2], -0.002551448858, epsilon = eps);

    // Now test with a non-zero kappa
    let pot = Plain::new(cutoff, Some(23.0));

    assert_eq!(
        pot.to_string(),
        "Plain Coulomb: 𝑟✂ = 29.0 Å, λᴰ = 23.0 Å <https://doi.org/msxd>"
    );

    assert_relative_eq!(pot.ion_potential(z1, cutoff + 1.0), 0.0, epsilon = eps);
    assert_relative_eq!(
        pot.ion_potential(z1, r.norm()),
        0.03198951663,
        epsilon = eps
    );
    assert_relative_eq!(
        pot.dipole_potential(&mu1, &((cutoff + 1.0) * rh)),
        0.0,
        epsilon = eps
    );
    assert_relative_eq!(pot.dipole_potential(&mu1, &r), 0.02642612243, epsilon = eps);

    // Test fields
    assert_relative_eq!(
        pot.ion_field(z1, &((cutoff + 1.0) * rh)).norm(),
        0.0,
        epsilon = eps
    );
    let field = pot.ion_field(z1, &r);
    assert_relative_eq!(field[0], 0.002781697098, epsilon = eps);
    assert_relative_eq!(field.norm(), 0.002781697098, epsilon = eps);
    assert_relative_eq!(
        pot.dipole_field(&mu1, &((cutoff + 1.0) * rh)).norm(),
        0.0,
        epsilon = eps
    );

    let field_scalar = pot.ion_field_scalar(z1, r.norm());
    assert_relative_eq!(field_scalar, field.norm(), epsilon = eps);

    let field = pot.dipole_field(&mu1, &r);
    assert_relative_eq!(field[0], 0.002872404612, epsilon = eps);
    assert_relative_eq!(field[1], -0.0004233017324, epsilon = eps);
    assert_relative_eq!(field[2], -0.0006651884364, epsilon = eps);
}

#[cfg(feature = "uom")]
#[test]
fn test_plain_si() {
    use crate::{
        pairwise::{MultipoleEnergySI, MultipoleFieldSI, MultipolePotentialSI},
        units::*,
    };
    use approx::assert_relative_eq;
    let eps = 1e-9; // Set epsilon for approximate equality
    let pot = Plain::without_cutoff();
    let z1 = ElectricCharge::new::<elementary_charge>(2.0);
    let z2 = ElectricCharge::new::<elementary_charge>(3.0);
    let r = Length::new::<nanometer>(2.3);
    let energy = pot.ion_ion_energy(z1, z2, r);
    assert_relative_eq!(
        energy.get::<kilojoule_per_mole>(),
        362.4403242896922,
        epsilon = eps
    );
    let potential = pot.ion_potential(z1, r);
    assert_relative_eq!(potential.get::<volt>(), 1.2521430850804929, epsilon = eps);

    let field = pot.ion_field(z1, r);
    assert_relative_eq!(
        field.get::<volt_per_micrometer>(),
        544.4100369915187,
        epsilon = eps
    );
}
