use core::f64::consts::PI;

// Electric units in SI:
//
// - Dipole moment (Cm)
// - Field (V / m = J / C / m)
// - Permittivity (C² / J / m)
// - Polarizability (C²m² / J = Cm² / V)
// - Volt (J / C)

fn main() {
    use approx::assert_relative_eq;
    use coulomb::{
        pairwise::{MultipoleEnergySI, MultipoleFieldSI, MultipolePotentialSI, Plain},
        units::*,
    };
    use coulomb::{AVOGADRO_CONSTANT, VACUUM_ELECTRIC_PERMITTIVITY};

    let vacuum_permittivity =
        ElectricPermittivity::new::<farad_per_meter>(VACUUM_ELECTRIC_PERMITTIVITY);

    // moles per particle
    let moles = AmountOfSubstance::new::<mole>(1.0 / AVOGADRO_CONSTANT);

    let z1 = ElectricCharge::new::<elementary_charge>(1.0);
    let z2 = ElectricCharge::new::<elementary_charge>(2.0);
    let r = Length::new::<nanometer>(2.3);

    // test energy
    let scheme = Plain::without_cutoff();
    let energy = scheme.ion_ion_energy(z1, z2, r);
    assert_relative_eq!(energy.get::<kilojoule_per_mole>(), 120.81344142989738);

    // test potential
    let potential = scheme.ion_potential(z1, r);
    let molar_energy = potential * z2 / moles;
    assert_relative_eq!(molar_energy.get::<kilojoule_per_mole>(), 120.81344142989738);

    // test field
    let field = scheme.ion_field(z1, r);
    let molar_energy = field * z2 * r / moles;
    assert_relative_eq!(molar_energy.get::<kilojoule_per_mole>(), 120.81344142989738);

    // test ion-induced dipole energy
    let polarizability = 4.0 * PI * vacuum_permittivity * Volume::new::<cubic_nanometer>(1.0);
    let induced_dipole: ElectricDipoleMoment = field * polarizability;
    let molar_energy = -0.5 * field * induced_dipole / moles;

    assert_relative_eq!(induced_dipole.get::<debye>(), 9.079782065350217);
    assert_relative_eq!(molar_energy.get::<kilojoule_per_mole>(), -2.482399963629026);
}
