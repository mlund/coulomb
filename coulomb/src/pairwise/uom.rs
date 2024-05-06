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

//! Physical units for the `pairwise` module using the `uom` crate.

use crate::pairwise::{MultipoleEnergy, MultipoleField, MultipolePotential};
use crate::units::*;

unit! {
    system: uom::si;
    quantity: uom::si::electric_charge_areal_density; // charge per length^2
    @valence_per_angstrom_squared: 16.02176634; "e/Å2", "valence_per_angstrom_squared", "valence_per_angstrom_squared";
}

impl<T: MultipoleEnergy> MultipoleEnergySI for T {}
impl<T: MultipoleField> MultipoleFieldSI for T {}
impl<T: MultipolePotential> MultipolePotentialSI for T {}

/// Interaction energy between multipoles with compile time units.
pub trait MultipoleEnergySI: MultipolePotentialSI + MultipoleFieldSI {
    /// Molar interaction energy between two point charges with units
    fn ion_ion_energy(
        &self,
        charge1: crate::units::ElectricCharge,
        charge2: crate::units::ElectricCharge,
        distance: crate::units::Length,
    ) -> MolarEnergy {
        use crate::units::*;
        charge1 * MultipolePotentialSI::ion_potential(self, charge2, distance)
            / AmountOfSubstance::new::<mole>(1.0 / crate::AVOGADRO_CONSTANT)
    }
}

/// Field due to electric multipoles with compile time units.
pub trait MultipoleFieldSI: MultipoleField {
    /// Electrostatic field magnitude from point charge with units
    ///
    /// The magnitude of the electric field from a point charge is calculated
    /// a `distance` away from the charge.
    fn ion_field(
        &self,
        charge: crate::units::ElectricCharge,
        distance: crate::units::Length,
    ) -> crate::units::ElectricField {
        use crate::units::*;
        use num::Zero;
        use uom::si::electric_charge_areal_density::ElectricChargeArealDensity;

        let r = distance.get::<angstrom>();
        if r >= self.cutoff() {
            return ElectricField::zero();
        }
        let charge_areal_density = self.ion_field_scalar(charge.get::<elementary_charge>(), r);
        ElectricChargeArealDensity::new::<valence_per_angstrom_squared>(charge_areal_density)
            / (4.0
                * std::f64::consts::PI
                * ElectricPermittivity::new::<farad_per_meter>(crate::VACUUM_ELECTRIC_PERMITTIVITY))
    }
}

/// Electric potential from point multipoles with compile time units.
pub trait MultipolePotentialSI: MultipolePotential {
    /// Ion-ion energy with units
    ///
    /// # Note
    ///
    /// Assumes that the cutoff distance is in angstrom!
    fn ion_potential(
        &self,
        charge: crate::units::ElectricCharge,
        distance: crate::units::Length,
    ) -> crate::units::ElectricPotential {
        use crate::units::*;
        let z = charge.get::<elementary_charge>();
        let r = distance.get::<angstrom>();
        ElectricChargeLinearDensity::new::<valence_per_angstrom>(MultipolePotential::ion_potential(
            self, z, r,
        )) / (4.0
            * std::f64::consts::PI
            * ElectricPermittivity::new::<farad_per_meter>(crate::VACUUM_ELECTRIC_PERMITTIVITY))
    }
}
