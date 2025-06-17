#[allow(unused_imports)]
pub use uom::si::{
    amount_of_substance::mole,
    electric_charge::elementary_charge,
    electric_dipole_moment::{atomic_unit_of_charge_centimeter, debye},
    electric_field::{atomic_unit_of_electric_field, volt_per_micrometer},
    electric_permittivity::farad_per_meter,
    electric_potential::volt,
    energy::joule,
    f64::{
        AmountOfSubstance, ElectricCharge, ElectricChargeLinearDensity, ElectricDipoleMoment,
        ElectricField, ElectricPermittivity, ElectricPotential, Energy, Length, MolarEnergy,
        Volume,
    },
    length::{angstrom, nanometer},
    molar_energy::kilojoule_per_mole,
    volume::{cubic_centimeter, cubic_nanometer},
};

unit! {
    system: uom::si;
    quantity: uom::si::electric_charge_linear_density;
    @valence_per_angstrom: 1.602_176_633_999_999_8e-9; "e/Å", "valence_per_angstrom", "valence_per_angstroms";
}
