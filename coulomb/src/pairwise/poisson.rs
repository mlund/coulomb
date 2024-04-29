// Copyright 2023 Björn Stenqvist and Mikael Lund
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

use super::{
    MultipoleEnergy, MultipoleField, MultipoleForce, MultipolePotential, SelfEnergyPrefactors,
    ShortRangeFunction,
};
use num::integer::binomial;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

impl<const C: i32, const D: i32> MultipolePotential for Poisson<C, D> {}
impl<const C: i32, const D: i32> MultipoleField for Poisson<C, D> {}
impl<const C: i32, const D: i32> MultipoleForce for Poisson<C, D> {}
impl<const C: i32, const D: i32> MultipoleEnergy for Poisson<C, D> {}

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

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Poisson<const C: i32, const D: i32> {
    cutoff: f64,
    debye_length: f64,
    _has_dipolar_selfenergy: bool,
    #[cfg_attr(feature = "serde", serde(skip))]
    reduced_kappa: f64,
    #[cfg_attr(feature = "serde", serde(skip))]
    use_yukawa_screening: bool,
    #[cfg_attr(feature = "serde", serde(skip))]
    reduced_kappa_squared: f64,
    #[cfg_attr(feature = "serde", serde(skip))]
    yukawa_denom: f64,
    #[cfg_attr(feature = "serde", serde(skip))]
    binom_cdc: f64,
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
        let mut has_dipolar_selfenergy = true;
        if C < 2 {
            has_dipolar_selfenergy = false;
        }
        let mut reduced_kappa = 0.0;
        let mut use_yukawa_screening = false;
        let mut reduced_kappa_squared = 0.0;
        let mut yukawa_denom = 0.0;
        let mut binom_cdc = 0.0;

        if let Some(debye_length) = debye_length {
            reduced_kappa = cutoff / debye_length;
            if reduced_kappa.abs() > 1e-6 {
                use_yukawa_screening = true;
                reduced_kappa_squared = reduced_kappa * reduced_kappa;
                yukawa_denom = 1.0 / (1.0 - (2.0 * reduced_kappa).exp());
                let _a1 = -f64::from(C + D) / f64::from(C);
                binom_cdc = f64::from(binomial(C + D, C) * D);
            }
        }
        if D != -C {
            binom_cdc = f64::from(binomial(C + D, C) * D);
        }

        Poisson {
            cutoff,
            debye_length: debye_length.unwrap_or(f64::INFINITY),
            _has_dipolar_selfenergy: has_dipolar_selfenergy,
            reduced_kappa,
            use_yukawa_screening,
            reduced_kappa_squared,
            yukawa_denom,
            binom_cdc,
        }
    }
}

impl<const C: i32, const D: i32> crate::Cutoff for Poisson<C, D> {
    fn cutoff(&self) -> f64 {
        self.cutoff
    }
}

impl<const C: i32, const D: i32> ShortRangeFunction for Poisson<C, D> {
    const URL: &'static str = match (C, D) {
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
    };

    fn kappa(&self) -> Option<f64> {
        if self.debye_length.is_normal() {
            Some(1.0 / self.debye_length)
        } else {
            None
        }
    }
    fn short_range_f0(&self, q: f64) -> f64 {
        if D == -C {
            return 1.0;
        }
        let qp = if self.use_yukawa_screening {
            (1.0 - (2.0 * self.reduced_kappa * q).exp()) * self.yukawa_denom
        } else {
            q
        };

        if D == 0 && C == 1 {
            return 1.0 - qp;
        }

        let sum: f64 = (0..C)
            .map(|c| {
                (num::integer::binomial(D - 1 + c, c) * (C - c)) as f64 / f64::from(C) * qp.powi(c)
            })
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
        let mut qp = q;
        let mut dqpdq = 1.0;
        if self.use_yukawa_screening {
            let exp2kq = (2.0 * self.reduced_kappa * q).exp();
            qp = (1.0 - exp2kq) * self.yukawa_denom;
            dqpdq = -2.0 * self.reduced_kappa * exp2kq * self.yukawa_denom;
        }
        let mut tmp1 = 1.0;
        let mut tmp2 = 0.0;
        for c in 1..C {
            let factor = (binomial(D - 1 + c, c) * (C - c)) as f64 / C as f64;
            tmp1 += factor * qp.powi(c);
            tmp2 += factor * c as f64 * qp.powi(c - 1);
        }
        let dsdqp = -f64::from(D + 1) * (1.0 - qp).powi(D) * tmp1 + (1.0 - qp).powi(D + 1) * tmp2;
        dsdqp * dqpdq
    }

    fn short_range_f2(&self, q: f64) -> f64 {
        if D == -C {
            return 0.0;
        }
        if D == 0 && C == 1 {
            return 0.0;
        }
        let mut qp = q;
        let mut dqpdq = 1.0;
        let mut d2qpdq2 = 0.0;
        let mut dsdqp = 0.0;
        // todo: use Option<f64> for kappa
        if self.use_yukawa_screening {
            qp = (1.0 - (2.0 * self.reduced_kappa * q).exp()) * self.yukawa_denom;
            dqpdq = -2.0
                * self.reduced_kappa
                * (2.0 * self.reduced_kappa * q).exp()
                * self.yukawa_denom;
            d2qpdq2 = -4.0
                * self.reduced_kappa_squared
                * (2.0 * self.reduced_kappa * q).exp()
                * self.yukawa_denom;
            let mut tmp1 = 1.0;
            let mut tmp2 = 0.0;
            for c in 1..C {
                let b = binomial(D - 1 + c, c) as f64 * (C - c) as f64;
                tmp1 += b / C as f64 * qp.powi(c);
                tmp2 += b * c as f64 / C as f64 * qp.powi(c - 1);
            }
            dsdqp = -f64::from(D + 1) * (1.0 - qp).powi(D) * tmp1 + (1.0 - qp).powi(D + 1) * tmp2;
        }
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
        let mut qp = q;
        let mut dqpdq = 1.0;
        let mut d2qpdq2 = 0.0;
        let mut d3qpdq3 = 0.0;
        let mut d2sdqp2 = 0.0;
        let mut dsdqp = 0.0;
        // todo: use Option<f64> for kappa
        if self.use_yukawa_screening {
            qp = (1.0 - (2.0 * self.reduced_kappa * q).exp()) * self.yukawa_denom;
            dqpdq = -2.0
                * self.reduced_kappa
                * (2.0 * self.reduced_kappa * q).exp()
                * self.yukawa_denom;
            d2qpdq2 = -4.0
                * self.reduced_kappa_squared
                * (2.0 * self.reduced_kappa * q).exp()
                * self.yukawa_denom;
            d3qpdq3 = -8.0
                * self.reduced_kappa_squared
                * self.reduced_kappa
                * (2.0 * self.reduced_kappa * q).exp()
                * self.yukawa_denom;
            d2sdqp2 = self.binom_cdc * (1.0 - qp).powi(D - 1) * qp.powi(C - 1);
            let mut tmp1 = 1.0;
            let mut tmp2 = 0.0;
            for c in 1..C {
                tmp1 += (binomial(D - 1 + c, c) * (C - c)) as f64 / C as f64 * qp.powi(c);
                tmp2 += (binomial(D - 1 + c, c) * (C - c)) as f64 / C as f64
                    * c as f64
                    * qp.powi(c - 1);
            }
            dsdqp = -f64::from(D + 1) * (1.0 - qp).powi(D) * tmp1 + (1.0 - qp).powi(D + 1) * tmp2;
        }
        let d3sdqp3 = self.binom_cdc
            * (1.0 - qp).powi(D - 2)
            * qp.powi(C - 2)
            * ((2.0 - C as f64 - D as f64) * qp + C as f64 - 1.0);
        d3sdqp3 * dqpdq * dqpdq * dqpdq + 3.0 * d2sdqp2 * dqpdq * d2qpdq2 + dsdqp * d3qpdq3
    }

    fn self_energy_prefactors(&self) -> SelfEnergyPrefactors {
        let mut c1: f64 = -0.5 * (C + D) as f64 / C as f64;
        if self.use_yukawa_screening {
            c1 = c1 * -2.0 * self.reduced_kappa * self.yukawa_denom;
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
        write!(f, " <{}>", Self::URL)?;
        Ok(())
    }
}

#[test]
fn test_poisson() {
    let pot = Stenqvist::new(29.0, None);
    let eps = 1e-9; // Set epsilon for approximate equality

    // Test Stenqvist short-range function
    approx::assert_relative_eq!(pot.short_range_f0(0.5), 0.15625, epsilon = eps);
    approx::assert_relative_eq!(pot.short_range_f1(0.5), -1.0, epsilon = eps);
    approx::assert_relative_eq!(pot.short_range_f2(0.5), 3.75, epsilon = eps);
    approx::assert_relative_eq!(pot.short_range_f3(0.5), 0.0, epsilon = eps);
    approx::assert_relative_eq!(pot.short_range_f3(0.6), -5.76, epsilon = eps);
    approx::assert_relative_eq!(pot.short_range_f0(1.0), 0.0, epsilon = eps);
    approx::assert_relative_eq!(pot.short_range_f1(1.0), 0.0, epsilon = eps);
    approx::assert_relative_eq!(pot.short_range_f2(1.0), 0.0, epsilon = eps);
    approx::assert_relative_eq!(pot.short_range_f3(1.0), 0.0, epsilon = eps);
    approx::assert_relative_eq!(pot.short_range_f0(0.0), 1.0, epsilon = eps);
    approx::assert_relative_eq!(pot.short_range_f1(0.0), -2.0, epsilon = eps);
    approx::assert_relative_eq!(pot.short_range_f2(0.0), 0.0, epsilon = eps);
    approx::assert_relative_eq!(pot.short_range_f3(0.0), 0.0, epsilon = eps);

    let pot = Stenqvist::new(29.0, Some(23.0));
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
    let pot = Fanourgakis::new(29.0, None);
    approx::assert_relative_eq!(pot.short_range_f0(0.5), 0.19921875, epsilon = eps);
    approx::assert_relative_eq!(pot.short_range_f1(0.5), -1.1484375, epsilon = eps);
    approx::assert_relative_eq!(pot.short_range_f2(0.5), 3.28125, epsilon = eps);
    approx::assert_relative_eq!(pot.short_range_f3(0.5), 6.5625, epsilon = eps);

    assert_eq!(
        pot.to_string(),
        "Poisson: 𝐶 = 4, 𝐷 = 3, 𝑟✂ = 29.0 Å <https://doi.org/10.1063/1.3216520>"
    )

    // Test
}
