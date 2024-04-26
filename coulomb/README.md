<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/VFPt_charges_plus_minus_thumb.svg/440px-VFPt_charges_plus_minus_thumb.svg.png?20200314191255" alt="crates.io", height="200">
</p>
<p align="center">
    <a href="https://opensource.org/licenses/Apache-2.0">
        <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg">
    </a>
</p>

-----

<p align = "center">
<b>Coulomb: A support library for electrolyte solutions and electrostatic interactions</b>
</p>

-----

# Features

This is a library for working with electrolyte solutions and calculating electrostatic interactions
in and between molecules and particles.
The main purpose is to offer support for molecular simulation software.

- Temperature dependent dielectric permittivity models for common solvents.
- Handling of ionic strength and the Debye screening length.
- Automatic stoichiometry deduction for arbitrary salts.
- Extensive library of truncated electrostatic interaction schemes such as
  _Wolf_, _Reaction field_, _Real-space Ewald_, generalized through a short-range function `trait`.
- Ewald summation with and without implicit salt.
- Multipole expansion for _energies_, _forces_, _fields_ between ions, dipoles, and quadrupoles.
- Extensively unit tested and may serve as reference for other implementations or approximations.

# Example

## Electrolytes
~~~ rust
use coulomb::{Medium, Salt};
let molarity = 0.1;
let medium = Medium::salt_water(298.15, Salt::CalciumChloride, molarity);
assert_eq!(medium.permittivity()?, 78.35565171480539);
assert_eq!(medium.ionic_strength()?, 0.3);             // mol/l
assert_eq!(medium.debye_length()?, 5.548902662386284); // angstrom
~~~

# Electrostatic interactions 
~~~ rust
~~~