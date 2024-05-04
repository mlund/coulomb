#[allow(unused_imports)]
pub use uom::si::{
    amount_of_substance::mole,
    electric_charge::elementary_charge,
    electric_permittivity::farad_per_meter,
    f64::{
        AmountOfSubstance, ElectricCharge, ElectricChargeLinearDensity, ElectricField,
        ElectricPermittivity, ElectricPotential, Length, MolarEnergy,
    },
    length::{angstrom, nanometer},
    molar_energy::kilojoule_per_mole,
};

unit! {
    system: uom::si;
    quantity: uom::si::electric_charge_linear_density;
    @valence_per_angstrom: 1.602_176_633_999_999_8e-9; "e/Å", "valence_per_angstrom", "valence_per_angstroms";
}
