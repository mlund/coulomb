pub use uom::si::{
    electric_charge::elementary_charge,
    electric_permittivity::farad_per_meter,
    f64::{
        ElectricCharge, ElectricChargeLinearDensity, ElectricPermittivity, ElectricPotential,
        Energy, Length,
    },
    length::angstrom,
};

unit! {
    system: uom::si;
    quantity: uom::si::electric_charge_linear_density; // charge per length
    @valence_per_angstrom: 1.602_176_633_999_999_8e-9; "e/Å", "valence_per_angstrom", "valence_per_angstroms";
}

// unit! {
//     system: uom::si;
//     quantity: uom::si::electric_charge_areal_density; // charge per length^2
//     @valence_per_angstrom_squared: 102.0e-9; "e/Å", "valence_per_angstrom_squared", "valence_per_angstrom_squared";
// }
