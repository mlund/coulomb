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

use crate::pairwise::{SelfEnergyPrefactors, ShortRangeFunction};
use num::integer::binomial;
#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize};

/// # Scheme for the Poisson short-range function
///
/// This is a general scheme for the short-ranged part of the electrostatic interaction
/// which can be used to arbitrarily cancel derivatives at the origin and at the cut-off.
/// From the abstract of <https://doi.org/c5fr>:
///
/// _"Electrostatic pair-potentials within molecular simulations are often based on empirical data,
/// cancellation of derivatives or moments, or statistical distributions of image-particles.
/// In this work we start with the fundamental Poisson equation and show that no truncated Coulomb
/// pair-potential, unsurprisingly, can solve the Poisson equation. For any such pair-potential
/// the Poisson equation gives two incompatible constraints, yet we find a single unique expression
/// which, pending two physically connected smoothness parameters, can obey either one of these.
/// This expression has a general form which covers several recently published pair-potentials.
/// For sufficiently large degree of smoothness we find that the solution implies a Gaussian
/// distribution of the charge, a feature which is frequently assumed in pair-potential theory.
/// We end up by recommending a single pair-potential based both on theoretical arguments and
/// empirical evaluations of non-thermal lattice- and thermal water-systems.
/// The same derivations have also been made for the screened Poisson equation,
/// i.e. for Yukawa potentials, with a similar solution."_
///
/// The general short-range function is:
/// $$
/// S(q) = (1 - q)^{D + 1} \sum_{c = 0}^{C - 1} \frac{C - c}{C} \binom{D - 1 + c}{c} q^c
/// $$
///
/// where $C$ is the number of cancelled derivatives at origin -2 (starting from the second derivative),
/// and $D$ is the number of cancelled derivatives at the cut-off (starting from the zeroth derivative).
///
/// For infinite Debye-length, $\kappa=0$, the [`Poisson`] scheme captures several
/// other truncation schemes by setting $C$ and $D$ according to this table:
///
/// | Type          | $C$ | $D$ | Reference / Comment
/// |---------------|-----|-----|---------------------
/// | `plain`       | 1   | -1  | Scheme for a vanilla coulomb interaction using the Poisson framework. Same as `Coulomb`.
/// | `wolf`        | 1   | 0   | Scheme for [Undamped Wolf](https://doi.org/10.1063/1.478738)
/// | `fennell`     | 1   | 1   | Scheme for [Levitt/undamped Fennell](https://doi.org/10/fp959p). See also doi:10/bqgmv2.
/// | `kale`        | 1   | 2   | Scheme for [Kale](https://doi.org/10/csh8bg)
/// | `mccann`      | 1   | 3   | Scheme for [McCann](https://doi.org/10.1021/ct300961)
/// | `fukuda`      | 2   | 1   | Scheme for [Undamped Fukuda](https://doi.org/10.1063/1.3582791)
/// | `markland`    | 2   | 2   | Scheme for [Markland](https://doi.org/10.1016/j.cplett.2008.09.019)
/// | `stenqvist`   | 3   | 3   | Scheme for [Stenqvist](https://doi.org/10/c5fr)
/// | `fanourgakis` | 4   | 3   | Scheme for [Fanourgakis](https://doi.org/10.1063/1.3216520),
///

/// Helper struct to store salt screening parameters
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
struct Screening {
    /// Inverse Debye screening length
    #[cfg_attr(feature = "serde", serde(alias = "κ"))]
    pub kappa: f64,
    /// Reduced kappa = cutoff * kappa
    pub reduced_kappa: f64,
    pub reduced_kappa_squared: f64,
    pub yukawa_denom: f64,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct Poisson<const C: i32, const D: i32> {
    /// Cutoff radius
    cutoff: f64,
    /// Debye length
    #[cfg_attr(feature = "serde", serde(alias = "debyelength"))]
    debye_length: Option<f64>,
    /// Currently not in use
    #[cfg_attr(feature = "serde", serde(skip))]
    _has_dipolar_selfenergy: bool,
    #[cfg_attr(feature = "serde", serde(skip))]
    binom_cdc: f64,
    #[cfg_attr(feature = "serde", serde(skip))]
    screening: Option<Screening>,
}

#[cfg(feature = "serde")]
impl<'de, const C: i32, const D: i32> Deserialize<'de> for Poisson<C, D> {
    fn deserialize<DES>(deserializer: DES) -> Result<Self, DES::Error>
    where
        DES: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct PoissonData {
            cutoff: f64,
            debye_length: Option<f64>,
        }

        let PoissonData {
            cutoff,
            debye_length,
        } = PoissonData::deserialize(deserializer)?;
        Ok(Poisson::new(cutoff, debye_length))
    }
}

/// Scheme for a vanilla coulomb interaction using the Poisson framework. Same as `Coulomb`.
pub type _Plain = Poisson<1, -1>;

/// Energy and force shifted Yukawa potential [Levitt/undamped Fennell](https://doi.org/10/fp959p).
///
/// See also doi:10/bqgmv2.
pub type Yukawa = Poisson<1, 1>;

/// Scheme for [Undamped Wolf](https://doi.org/10.1063/1.478738)
pub type UndampedWolf = Poisson<1, 0>;

/// Scheme for [Kale](https://doi.org/10/csh8bg)
pub type Kale = Poisson<1, 2>;

/// Scheme for [McCann](https://doi.org/10.1021/ct300961)
pub type McCann = Poisson<1, 3>;

/// Scheme for [Undamped Fukuda](https://doi.org/10.1063/1.3582791)
pub type UndampedFukuda = Poisson<2, 1>;

/// Scheme for [Markland](https://doi.org/10.1016/j.cplett.2008.09.019)
pub type Markland = Poisson<2, 2>;

/// Scheme for [Stenqvist](https://doi.org/10/c5fr)
pub type Stenqvist = Poisson<3, 3>;

/// Scheme for [Fanourgakis](https://doi.org/10.1063/1.3216520)
pub type Fanourgakis = Poisson<4, 3>;

impl<const C: i32, const D: i32> Poisson<C, D> {
    pub fn new(cutoff: f64, debye_length: Option<f64>) -> Self {
        if C < 1 {
            panic!("`C` must be larger than zero");
        }
        if D < -1 && D != -C {
            panic!("If `D` is less than negative one, then it has to equal negative `C`");
        }
        if D == 0 && C != 1 {
            panic!("If `D` is zero, then `C` has to equal one ");
        }

        let _has_dipolar_selfenergy = C >= 2;

        let screening = debye_length.map(|debye_length| {
            let reduced_kappa = cutoff / debye_length;
            Screening {
                kappa: 1.0 / debye_length,
                reduced_kappa,
                reduced_kappa_squared: reduced_kappa.powi(2),
                yukawa_denom: 1.0 / (1.0 - (2.0 * reduced_kappa).exp()),
            }
        });

        let binom_cdc = if screening.is_some() || D != -C {
            f64::from(binomial(C + D, C) * D)
        } else {
            0.0
        };

        Self {
            cutoff,
            debye_length,
            _has_dipolar_selfenergy,
            binom_cdc,
            screening,
        }
    }
}

impl<const C: i32, const D: i32> crate::Cutoff for Poisson<C, D> {
    fn cutoff(&self) -> f64 {
        self.cutoff
    }
}

impl<const C: i32, const D: i32> ShortRangeFunction for Poisson<C, D> {
    fn url() -> &'static str {
        match (C, D) {
            (1, -1) => "https://doi.org/msxd",             // plain
            (1, 0) => "https://doi.org/10.1063/1.478738",  // wolf
            (1, 1) => "https://doi.org/10/fp959p",         // fennell
            (1, 2) => "https://doi.org/10/csh8bg",         // kale
            (1, 3) => "https://doi.org/10.1021/ct300961",  // mccann
            (2, 1) => "https://doi.org/10.1063/1.3582791", // fukuda
            (2, 2) => "https://doi.org/dbpbts",            // markland
            (3, 3) => "https://doi.org/10/c5fr",           // stenqvist
            (4, 3) => "https://doi.org/10.1063/1.3216520", // fanourgakis
            _ => "https://doi.org/c5fr",                   // generic poisson
        }
    }

    fn kappa(&self) -> Option<f64> {
        self.screening.as_ref().map(|s| s.kappa)
    }

    fn short_range_f0(&self, q: f64) -> f64 {
        // checks on C and D are evaluated at compile time
        if D == -C {
            return 1.0;
        }
        let qp: f64 = self.screening.as_ref().map_or(q, |s| {
            (1.0 - (2.0 * s.reduced_kappa * q).exp()) * s.yukawa_denom
        });

        if D == 0 && C == 1 {
            return 1.0 - qp;
        }

        // todo: could binomial coeffs be evaluated at compile time? E.g. with recursion.
        let sum: f64 = (0..C)
            .map(|c| (binomial(D - 1 + c, c) * (C - c)) as f64 / f64::from(C) * qp.powi(c))
            .sum();
        (1.0 - qp).powi(D + 1) * sum
    }
    fn short_range_f1(&self, q: f64) -> f64 {
        if D == -C {
            return 0.0;
        }
        if D == 0 && C == 1 {
            return 0.0;
        }
        let (qp, dqpdq) = if let Some(s) = &self.screening {
            let exp2kq = (2.0 * s.reduced_kappa * q).exp();
            let qp = (1.0 - exp2kq) * s.yukawa_denom;
            let dqpdq = -2.0 * s.reduced_kappa * exp2kq * s.yukawa_denom;
            (qp, dqpdq)
        } else {
            (q, 1.0)
        };
        let mut sum1 = 1.0;
        let mut sum2 = 0.0;

        for c in 1..C {
            let factor = (binomial(D - 1 + c, c) * (C - c)) as f64 / C as f64;
            sum1 += factor * qp.powi(c);
            sum2 += factor * c as f64 * qp.powi(c - 1);
        }
        let dsdqp = -f64::from(D + 1) * (1.0 - qp).powi(D) * sum1 + (1.0 - qp).powi(D + 1) * sum2;
        dsdqp * dqpdq
    }

    fn short_range_f2(&self, q: f64) -> f64 {
        if D == -C {
            return 0.0;
        }
        if D == 0 && C == 1 {
            return 0.0;
        }

        let (qp, dqpdq, d2qpdq2, dsdqp) = if let Some(s) = &self.screening {
            let qp = (1.0 - (2.0 * s.reduced_kappa * q).exp()) * s.yukawa_denom;
            let dqpdq = -2.0 * s.reduced_kappa * (2.0 * s.reduced_kappa * q).exp() * s.yukawa_denom;
            let d2qpdq2 =
                -4.0 * s.reduced_kappa_squared * (2.0 * s.reduced_kappa * q).exp() * s.yukawa_denom;
            let mut tmp1 = 1.0;
            let mut tmp2 = 0.0;
            for c in 1..C {
                let b = binomial(D - 1 + c, c) as f64 * (C - c) as f64;
                tmp1 += b / C as f64 * qp.powi(c);
                tmp2 += b * c as f64 / C as f64 * qp.powi(c - 1);
            }
            let dsdqp =
                -f64::from(D + 1) * (1.0 - qp).powi(D) * tmp1 + (1.0 - qp).powi(D + 1) * tmp2;
            (qp, dqpdq, d2qpdq2, dsdqp)
        } else {
            (q, 1.0, 0.0, 0.0)
        };
        let d2sdqp2 = self.binom_cdc * (1.0 - qp).powi(D - 1) * qp.powi(C - 1);
        d2sdqp2 * dqpdq * dqpdq + dsdqp * d2qpdq2
    }

    fn short_range_f3(&self, q: f64) -> f64 {
        if D == -C {
            return 0.0;
        }
        if D == 0 && C == 1 {
            return 0.0;
        }

        let (qp, dqpdq, d2qpdq2, d3qpdq3, d2sdqp2, dsdqp) = if let Some(s) = &self.screening {
            let qp = (1.0 - (2.0 * s.reduced_kappa * q).exp()) * s.yukawa_denom;
            let dqpdq = -2.0 * s.reduced_kappa * (2.0 * s.reduced_kappa * q).exp() * s.yukawa_denom;
            let d2qpdq2 =
                -4.0 * s.reduced_kappa_squared * (2.0 * s.reduced_kappa * q).exp() * s.yukawa_denom;
            let d3qpdq3 = -8.0
                * s.reduced_kappa_squared
                * s.reduced_kappa
                * (2.0 * s.reduced_kappa * q).exp()
                * s.yukawa_denom;
            let d2sdqp2 = self.binom_cdc * (1.0 - qp).powi(D - 1) * qp.powi(C - 1);
            let mut tmp1 = 1.0;
            let mut tmp2 = 0.0;
            for c in 1..C {
                tmp1 += (binomial(D - 1 + c, c) * (C - c)) as f64 / C as f64 * qp.powi(c);
                tmp2 += (binomial(D - 1 + c, c) * (C - c)) as f64 / C as f64
                    * c as f64
                    * qp.powi(c - 1);
            }
            let dsdqp =
                -f64::from(D + 1) * (1.0 - qp).powi(D) * tmp1 + (1.0 - qp).powi(D + 1) * tmp2;
            (qp, dqpdq, d2qpdq2, d3qpdq3, d2sdqp2, dsdqp)
        } else {
            (q, 1.0, 0.0, 0.0, 0.0, 0.0)
        };
        let d3sdqp3 = self.binom_cdc
            * (1.0 - qp).powi(D - 2)
            * qp.powi(C - 2)
            * ((2.0 - C as f64 - D as f64) * qp + C as f64 - 1.0);
        d3sdqp3 * dqpdq * dqpdq * dqpdq + 3.0 * d2sdqp2 * dqpdq * d2qpdq2 + dsdqp * d3qpdq3
    }

    fn self_energy_prefactors(&self) -> SelfEnergyPrefactors {
        let mut c1: f64 = -0.5 * (C + D) as f64 / C as f64;
        if let Some(s) = &self.screening {
            c1 = c1 * -2.0 * s.reduced_kappa * s.yukawa_denom;
        }
        SelfEnergyPrefactors {
            monopole: Some(c1),
            dipole: None,
        }
    }
}

impl<const C: i32, const D: i32> core::fmt::Display for Poisson<C, D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Poisson: 𝐶 = {}, 𝐷 = {}, 𝑟✂ = {:.1} Å",
            C, D, self.cutoff
        )?;
        if let Some(debye_length) = self.kappa().map(f64::recip) {
            write!(f, ", λᴰ = {:.1} Å", debye_length)?;
        }
        write!(f, " <{}>", Self::url())?;
        Ok(())
    }
}

#[test]
fn test_poisson() {
    use crate::{
        pairwise::{MultipoleEnergy, MultipoleField, MultipoleForce, MultipolePotential},
        Matrix3, Vector3,
    };
    use approx::assert_relative_eq;
    let cutoff = 29.0;
    let pot = Stenqvist::new(cutoff, None);
    let eps = 1e-9; // Set epsilon for approximate equality

    // Test Stenqvist short-range function
    assert_relative_eq!(pot.short_range_f0(0.5), 0.15625, epsilon = eps);
    assert_relative_eq!(pot.short_range_f1(0.5), -1.0, epsilon = eps);
    assert_relative_eq!(pot.short_range_f2(0.5), 3.75, epsilon = eps);
    assert_relative_eq!(pot.short_range_f3(0.5), 0.0, epsilon = eps);
    assert_relative_eq!(pot.short_range_f3(0.6), -5.76, epsilon = eps);
    assert_relative_eq!(pot.short_range_f0(1.0), 0.0, epsilon = eps);
    assert_relative_eq!(pot.short_range_f1(1.0), 0.0, epsilon = eps);
    assert_relative_eq!(pot.short_range_f2(1.0), 0.0, epsilon = eps);
    assert_relative_eq!(pot.short_range_f3(1.0), 0.0, epsilon = eps);
    assert_relative_eq!(pot.short_range_f0(0.0), 1.0, epsilon = eps);
    assert_relative_eq!(pot.short_range_f1(0.0), -2.0, epsilon = eps);
    assert_relative_eq!(pot.short_range_f2(0.0), 0.0, epsilon = eps);
    assert_relative_eq!(pot.short_range_f3(0.0), 0.0, epsilon = eps);

    let pot = Stenqvist::new(cutoff, Some(23.0));
    approx::assert_relative_eq!(
        pot.self_energy(&[2.0], &[0.0]),
        -0.03037721287,
        epsilon = eps
    );

    assert_eq!(
        pot.to_string(),
        "Poisson: 𝐶 = 3, 𝐷 = 3, 𝑟✂ = 29.0 Å, λᴰ = 23.0 Å <https://doi.org/10/c5fr>"
    );

    // Test Fanougarkis short-range function
    let pot = Fanourgakis::new(cutoff, None);
    assert_relative_eq!(pot.short_range_f0(0.5), 0.19921875, epsilon = eps);
    assert_relative_eq!(pot.short_range_f1(0.5), -1.1484375, epsilon = eps);
    assert_relative_eq!(pot.short_range_f2(0.5), 3.28125, epsilon = eps);
    assert_relative_eq!(pot.short_range_f3(0.5), 6.5625, epsilon = eps);

    assert_eq!(
        pot.to_string(),
        "Poisson: 𝐶 = 4, 𝐷 = 3, 𝑟✂ = 29.0 Å <https://doi.org/10.1063/1.3216520>"
    );

    let pot = Poisson::<4, 3>::new(cutoff, None);
    let z1 = 2.0;
    let z2 = 3.0;
    let r = Vector3::new(23.0, 0.0, 0.0); // distance vector
    let rq = Vector3::new(
        5.75 * 6.0_f64.sqrt(),
        5.75 * 2.0_f64.sqrt(),
        11.5 * 2.0_f64.sqrt(),
    );
    let rh = r.normalize();
    let mu1 = Vector3::new(19.0, 7.0, 11.0);
    let mu2 = Vector3::new(13.0, 17.0, 5.0);
    //      [3 7 8]
    //  Q = [5 9 6]
    //      [2 1 4]
    let quad1 = Matrix3::from_row_slice(&[3.0, 7.0, 8.0, 5.0, 9.0, 6.0, 2.0, 1.0, 4.0]);
    let quad2 = Matrix3::zeros();
    assert_relative_eq!(quad1.trace(), 16.0, epsilon = eps);

    // Test potentials
    assert_relative_eq!(pot.ion_potential(z1, cutoff), 0.0, epsilon = eps);
    assert_relative_eq!(
        pot.ion_potential(z1, r.norm()),
        0.0009430652121,
        epsilon = eps
    );
    assert_relative_eq!(
        pot.dipole_potential(&mu1, &rh.scale(cutoff)),
        0.0,
        epsilon = eps
    );
    assert_relative_eq!(
        pot.dipole_potential(&mu1, &r),
        0.005750206554,
        epsilon = eps
    );
    assert_relative_eq!(
        pot.quadrupole_potential(&quad1, &rq),
        0.000899228165,
        epsilon = eps
    );
    assert_relative_eq!(
        pot.quadrupole_potential(&quad1, &rq.scale(cutoff / 23.0)),
        0.0,
        epsilon = eps
    );

    // Test fields with no salt
    assert_relative_eq!(
        pot.ion_field(z1, &rh.scale(cutoff)).norm(),
        0.0,
        epsilon = eps
    );
    let e_ion = pot.ion_field(z1, &r);
    assert_relative_eq!(e_ion[0], 0.0006052849004, epsilon = eps);
    assert_relative_eq!(e_ion.norm(), 0.0006052849004, epsilon = eps);
    assert_relative_eq!(
        pot.dipole_field(&mu1, &rh.scale(cutoff)).norm(),
        0.0,
        epsilon = eps
    );
    let ion_field_scalar = pot.ion_field_scalar(z1, r.norm());
    assert_relative_eq!(ion_field_scalar, e_ion.norm(), epsilon = eps);

    let e_dipole = pot.dipole_field(&mu1, &r);
    assert_relative_eq!(e_dipole[0], 0.002702513754, epsilon = eps);
    assert_relative_eq!(e_dipole[1], -0.00009210857180, epsilon = eps);
    assert_relative_eq!(e_dipole[2], -0.0001447420414, epsilon = eps);

    let e_quadrupole = pot.quadrupole_field(&quad1, &r);
    assert_relative_eq!(e_quadrupole[0], 0.00001919309993, epsilon = eps);
    assert_relative_eq!(e_quadrupole[1], -0.00004053806958, epsilon = eps);
    assert_relative_eq!(e_quadrupole[2], -0.00003378172465, epsilon = eps);

    // Test energies
    assert_relative_eq!(pot.ion_ion_energy(z1, z2, cutoff), 0.0, epsilon = eps);
    assert_relative_eq!(
        pot.ion_ion_energy(z1, z2, r.norm()),
        0.002829195636,
        epsilon = eps
    );
    assert_relative_eq!(
        pot.ion_dipole_energy(z1, &mu2, &rh.scale(cutoff)),
        0.0,
        epsilon = eps
    );
    assert_relative_eq!(
        pot.ion_dipole_energy(z1, &mu2, &r),
        -0.007868703705,
        epsilon = eps
    );
    assert_relative_eq!(
        pot.ion_dipole_energy(z2, &mu1, &-r),
        0.01725061966,
        epsilon = eps
    );
    assert_relative_eq!(
        pot.dipole_dipole_energy(&mu1, &mu2, &rh.scale(cutoff)),
        0.0,
        epsilon = eps
    );
    assert_relative_eq!(
        pot.dipole_dipole_energy(&mu1, &mu2, &r),
        -0.03284312288,
        epsilon = eps
    );
    assert_relative_eq!(
        pot.ion_quadrupole_energy(z2, &quad1, &rq.scale(cutoff / 23.0 * 29.0)),
        0.0,
        epsilon = eps
    );
    assert_relative_eq!(
        pot.ion_quadrupole_energy(z2, &quad1, &rq),
        0.002697684495,
        epsilon = eps
    );
    assert_relative_eq!(
        pot.ion_quadrupole_energy(z1, &quad2, &-rq.scale(cutoff / 23.0 * 29.0)),
        0.0,
        epsilon = eps
    );
    assert_relative_eq!(
        pot.ion_quadrupole_energy(z1, &quad2, &-rq),
        0.0,
        epsilon = eps
    );

    // Test forces
    assert_relative_eq!(
        pot.ion_ion_force(z1, z2, &Vector3::new(cutoff, 0.0, 0.0))
            .norm(),
        0.0,
        epsilon = eps
    );
    let f_ionion = pot.ion_ion_force(z1, z2, &r);
    assert_relative_eq!(f_ionion[0], 0.001815854701, epsilon = eps);
    assert_relative_eq!(f_ionion.norm(), 0.001815854701, epsilon = eps);

    assert_relative_eq!(
        pot.ion_dipole_force(z2, &mu1, &rh.scale(cutoff)).norm(),
        0.0,
        epsilon = eps
    );

    let f_iondipole_ba = pot.ion_dipole_force(z2, &mu1, &r);
    assert_relative_eq!(f_iondipole_ba[0], 0.008107541263, epsilon = eps);
    assert_relative_eq!(f_iondipole_ba[1], -0.0002763257154, epsilon = eps);
    assert_relative_eq!(f_iondipole_ba[2], -0.0004342261242, epsilon = eps);

    let f_iondipole_ab = pot.ion_dipole_force(z1, &mu2, &-r);
    assert_relative_eq!(f_iondipole_ab[0], 0.003698176716, epsilon = eps);
    assert_relative_eq!(f_iondipole_ab[1], -0.0004473844916, epsilon = eps);
    assert_relative_eq!(f_iondipole_ab[2], -0.0001315836740, epsilon = eps);

    assert_relative_eq!(
        pot.dipole_dipole_force(&mu1, &mu2, &rh.scale(cutoff))
            .norm(),
        0.0,
        epsilon = eps
    );

    let f_dipoledipole = pot.dipole_dipole_force(&mu1, &mu2, &r);
    assert_relative_eq!(f_dipoledipole[0], 0.009216400961, epsilon = eps);
    assert_relative_eq!(f_dipoledipole[1], -0.002797126801, epsilon = eps);
    assert_relative_eq!(f_dipoledipole[2], -0.001608010094, epsilon = eps);

    // Test with screening
    let debye_length = 23.0;
    let pot = Poisson::<3, 3>::new(cutoff, Some(debye_length));

    // Test short-ranged function with screening
    assert_relative_eq!(pot.short_range_f0(0.5), 0.5673222086324718, epsilon = eps);
    assert_relative_eq!(pot.short_range_f1(0.5), -1.4373727619264975, epsilon = eps);
    assert_relative_eq!(pot.short_range_f2(0.5), -2.552012309527445, epsilon = eps);
    assert_relative_eq!(pot.short_range_f3(0.5), 4.384434366606605, epsilon = eps);

    // Test potentials with screening
    assert_relative_eq!(pot.ion_potential(z1, cutoff), 0.0, epsilon = eps);
    assert_relative_eq!(
        pot.ion_potential(z1, r.norm()),
        0.003344219306,
        epsilon = eps
    );
    assert_relative_eq!(
        pot.dipole_potential(&mu1, &rh.scale(cutoff)),
        0.0,
        epsilon = eps
    );
    assert_relative_eq!(pot.dipole_potential(&mu1, &r), 0.01614089171, epsilon = eps);
    assert_relative_eq!(
        pot.quadrupole_potential(&quad1, &rq),
        0.0016294707475,
        epsilon = eps
    );
    assert_relative_eq!(
        pot.quadrupole_potential(&quad1, &rq.scale(cutoff / 23.0 * 29.0)),
        0.0,
        epsilon = eps
    );

    // Test fields with screening
    assert_relative_eq!(
        pot.ion_field(z1, &r.scale(cutoff)).norm(),
        0.0,
        epsilon = eps
    );

    let ion_field = pot.ion_field(z1, &r);
    assert_relative_eq!(ion_field[0], 0.001699041230, epsilon = eps);
    assert_relative_eq!(ion_field.norm(), 0.001699041230, epsilon = eps);
    assert_relative_eq!(
        pot.dipole_field(&mu1, &r.scale(cutoff)).norm(),
        0.0,
        epsilon = eps
    );

    let ion_field_scalar = pot.ion_field_scalar(z1, r.norm());
    assert_relative_eq!(ion_field_scalar, ion_field.norm(), epsilon = eps);

    let dipole_field = pot.dipole_field(&mu1, &r);
    assert_relative_eq!(dipole_field[0], 0.004956265485, epsilon = eps);
    assert_relative_eq!(dipole_field[1], -0.0002585497523, epsilon = eps);
    assert_relative_eq!(dipole_field[2], -0.0004062924688, epsilon = eps);

    let quadrupole_field = pot.quadrupole_field(&quad1, &r);
    assert_relative_eq!(quadrupole_field[0], -0.00005233355205, epsilon = eps);
    assert_relative_eq!(quadrupole_field[1], -0.00007768480608, epsilon = eps);
    assert_relative_eq!(quadrupole_field[2], -0.00006473733856, epsilon = eps);

    // Test energies with screening
    assert_relative_eq!(pot.ion_ion_energy(z1, z2, cutoff), 0.0, epsilon = eps);
    assert_relative_eq!(
        pot.ion_ion_energy(z1, z2, r.norm()),
        0.01003265793,
        epsilon = eps
    );
    assert_relative_eq!(
        pot.ion_dipole_energy(z1, &mu2, &r.scale(cutoff)),
        0.0,
        epsilon = eps
    );
    assert_relative_eq!(
        pot.ion_dipole_energy(z1, &mu2, &r),
        -0.02208753604,
        epsilon = eps
    );
    assert_relative_eq!(
        pot.ion_dipole_energy(z2, &mu1, &-r),
        0.04842267505,
        epsilon = eps
    );
    assert_relative_eq!(
        pot.dipole_dipole_energy(&mu1, &mu2, &r.scale(cutoff)),
        0.0,
        epsilon = eps
    );
    assert_relative_eq!(
        pot.dipole_dipole_energy(&mu1, &mu2, &r),
        -0.05800464321,
        epsilon = eps
    );
    assert_relative_eq!(
        pot.ion_quadrupole_energy(z2, &quad1, &rq.scale(cutoff / 23.0 * 29.0)),
        0.0,
        epsilon = eps
    );
    assert_relative_eq!(
        pot.ion_quadrupole_energy(z2, &quad1, &rq),
        0.004888412229,
        epsilon = eps
    );

    // Test forces with screening
    assert_relative_eq!(
        pot.ion_ion_force(z1, z2, &r.scale(cutoff)).norm(),
        0.0,
        epsilon = eps
    );

    let ionion_force = pot.ion_ion_force(z1, z2, &r);
    assert_relative_eq!(ionion_force[0], 0.005097123689, epsilon = eps);
    assert_relative_eq!(ionion_force.norm(), 0.005097123689, epsilon = eps);

    assert_relative_eq!(
        pot.ion_dipole_force(z2, &mu1, &r.scale(cutoff)).norm(),
        0.0,
        epsilon = eps
    );

    let iondipole_force_21 = pot.ion_dipole_force(z2, &mu1, &r);
    assert_relative_eq!(iondipole_force_21[0], 0.01486879646, epsilon = eps);
    assert_relative_eq!(iondipole_force_21[1], -0.0007756492577, epsilon = eps);
    assert_relative_eq!(iondipole_force_21[2], -0.001218877402, epsilon = eps);

    let iondipole_force_12 = pot.ion_dipole_force(z1, &mu2, &-r);
    assert_relative_eq!(iondipole_force_12[0], 0.006782258035, epsilon = eps);
    assert_relative_eq!(iondipole_force_12[1], -0.001255813082, epsilon = eps);
    assert_relative_eq!(iondipole_force_12[2], -0.0003693567885, epsilon = eps);
    assert_relative_eq!(
        pot.dipole_dipole_force(&mu1, &mu2, &r.scale(cutoff)).norm(),
        0.0,
        epsilon = eps
    );

    let dipoledipole_force = pot.dipole_dipole_force(&mu1, &mu2, &r);
    assert_relative_eq!(dipoledipole_force[0], 0.002987655323, epsilon = eps);
    assert_relative_eq!(dipoledipole_force[1], -0.005360251624, epsilon = eps);
    assert_relative_eq!(dipoledipole_force[2], -0.003081497314, epsilon = eps);
}
