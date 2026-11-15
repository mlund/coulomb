use coulomb::reciprocal::EwaldReciprocal;

#[test]
fn strong_screening_needs_no_reciprocal_vectors() {
    let ewald = EwaldReciprocal::new([10.0; 3], 5.0, 1e-3, Some(0.1));
    assert_eq!(ewald.num_k_vectors(), 0);
}

#[test]
fn zero_splitting_real_space_is_bare_yukawa() {
    use coulomb::pairwise::{
        MultipoleEnergy, MultipolePotential, RealSpaceEwald, ShortRangeFunction,
    };

    for debye_length in [0.1, 0.001] {
        let scheme = RealSpaceEwald::new(5.0, 1e-3, Some(debye_length));
        for q in [0.0, 0.1, 0.8, 1.0] {
            assert_eq!(scheme.short_range_f0(q), 1.0);
            assert_eq!(scheme.short_range_f1(q), 0.0);
            assert_eq!(scheme.short_range_f2(q), 0.0);
            assert_eq!(scheme.short_range_f3(q), 0.0);
        }
        assert_eq!(scheme.self_energy(&[1.0, 2.0], &[3.0]), 0.0);
        let r = debye_length;
        approx::assert_relative_eq!(scheme.ion_potential(1.0, r), (-1.0_f64).exp() / r);
    }
}

#[test]
fn empty_reciprocal_sum_stays_zero_through_updates() {
    use coulomb::pairwise::{MultipoleEnergy, MultipoleForce};
    use coulomb::reciprocal::EwaldPolicy;

    // Non-neutral charges also exercise the screened k=0 correction.
    let charges = [1.0, 2.0];
    let x = [0.0, 0.1];
    let yz = [0.0; 2];
    for policy in [EwaldPolicy::PBC, EwaldPolicy::IPBC] {
        for optimize in [false, true] {
            let mut ewald = EwaldReciprocal::new([10.0; 3], 5.0, 1e-3, Some(0.1));
            ewald.set_policy(policy);
            ewald.update_all(&x, &yz, &yz, &charges, Some([12.0; 3]), optimize);
            assert_eq!(ewald.num_k_vectors(), 0);
            assert_eq!(ewald.energy(), 0.0);
            assert_eq!(ewald.self_energy(&charges), 0.0);
            assert_eq!(ewald.force([0.1, 0.0, 0.0], 2.0), [0.0; 3]);
            let old = [0.1, 0.0, 0.0];
            let new = [0.2, 0.0, 0.0];
            assert_eq!(ewald.energy_change(2.0, old, new), 0.0);
            ewald.update_particle(2.0, old, new);
            assert_eq!(ewald.energy(), 0.0);

            let real = ewald.real_space_scheme();
            let r = 0.1;
            let expected_energy = 2.0 * (-1.0_f64).exp() / r;
            approx::assert_relative_eq!(real.ion_ion_energy(1.0, 2.0, r), expected_energy);
            let force = real.ion_ion_force(1.0, 2.0, [r, 0.0, 0.0]);
            approx::assert_relative_eq!(force.x, 4.0 * (-1.0_f64).exp() / (r * r));
            assert_eq!([force.y, force.z], [0.0; 2]);
        }
    }
}
