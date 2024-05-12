// Copyright 2023 Mikael Lund
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

//! Visualizes the electric potential at grid points in the XY-plane due to
//! two particles with charge, dipole moment, and radius.
//! The potential is calculated using the plain Coulomb scheme and saved as
//! a bitmap image where negative to positive potential is mapped to a color
//! gradient from red to blue.

extern crate image;

use coulomb::{pairwise::*, Vector3};
use image::RgbaImage;

const ASPECT_RATIO: f64 = 1.5;
const WIDTH: u32 = 400;
const HEIGHT: u32 = (WIDTH as f64 / ASPECT_RATIO) as u32;
const XMAX: f64 = 15.0;
const YMAX: f64 = XMAX / ASPECT_RATIO;

/// A particle with charge, dipole moment, position, and radius
#[derive(Clone)]
struct Particle {
    charge: f64,
    dipole: Vector3,
    pos: Vector3,
    radius: f64,
}

impl Particle {
    /// Sum of ion and dipole potential at `pos` or `None` if inside particle
    pub fn potential<T: MultipolePotential>(&self, pos: &Vector3, scheme: &T) -> Option<f64> {
        let r = pos - self.pos;
        let norm = r.norm();
        (norm > self.radius).then(|| {
            scheme.ion_potential(self.charge, norm) + scheme.dipole_potential(&self.dipole, &r)
        })
    }
}

fn main() {
    let particles = [
        Particle {
            charge: -3.0,
            dipole: Vector3::zeros(),
            pos: Vector3::new(XMAX / 2.0 - 3.0, YMAX / 2.0 - 1.0, 0.0),
            radius: 1.5,
        },
        Particle {
            charge: 0.0,
            dipole: Vector3::new(-5.0, 5.0, 0.0),
            pos: Vector3::new(XMAX / 2.0 + 3.0, YMAX / 2.0 + 1.0, 0.0),
            radius: 1.5,
        },
    ];

    let mut data = Vec::with_capacity((WIDTH * HEIGHT) as usize); // w, h, potential
    let scheme = Plain::default(); // vanilla Coulomb scheme, S(q)=1

    // Calculate potential at each pixel
    for w in 0..WIDTH {
        for h in 0..HEIGHT {
            let pos = Vector3::new(
                w as f64 * XMAX / WIDTH as f64,
                h as f64 * YMAX / HEIGHT as f64,
                0.0,
            );
            let pot: Vec<f64> = particles
                .iter()
                .filter_map(|p| p.potential(&pos, &scheme))
                .collect();
            if pot.len() == particles.len() {
                // only if outside *all* particles
                data.push((w, h, pot.iter().sum::<f64>()));
            }
        }
    }

    // Maximum absolute potential value
    let max = data
        .iter()
        .map(|(_, _, p)| p.abs())
        .max_by(f64::total_cmp)
        .unwrap();

    // New image and color gradient
    let mut img = RgbaImage::new(WIDTH, HEIGHT);
    let gradient = colorgrad::rd_bu();

    for &(w, h, p) in &data {
        let color = gradient.at(remap(p, -max, max, 0.0, 1.0)).to_rgba8();
        img.put_pixel(w, h, color.into());
    }
    img.save("potential.png").unwrap();
}

// Map t which is in range [a, b] to range [c, d]
fn remap(t: f64, a: f64, b: f64, c: f64, d: f64) -> f64 {
    (t - a) * ((d - c) / (b - a)) + c
}
