use coulomb::reciprocal::{EwaldPolicy, EwaldReciprocal};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

/// Generate a neutral NaCl-like system with `n_cells`³ unit cells (8 ions each).
fn nacl_system(n_cells: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, f64) {
    let a = 1.0;
    let l_cell = 2.0 * a;
    let l_box = l_cell * n_cells as f64;
    let n = 8 * n_cells * n_cells * n_cells;

    let na_basis: [[f64; 3]; 4] = [[0.0, 0.0, 0.0], [a, a, 0.0], [a, 0.0, a], [0.0, a, a]];
    let cl_basis: [[f64; 3]; 4] = [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a], [a, a, a]];

    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    let mut charges = Vec::with_capacity(n);

    for ix in 0..n_cells {
        for iy in 0..n_cells {
            for iz in 0..n_cells {
                let ox = ix as f64 * l_cell;
                let oy = iy as f64 * l_cell;
                let oz = iz as f64 * l_cell;
                for b in &na_basis {
                    x.push(b[0] + ox);
                    y.push(b[1] + oy);
                    z.push(b[2] + oz);
                    charges.push(1.0);
                }
                for b in &cl_basis {
                    x.push(b[0] + ox);
                    y.push(b[1] + oy);
                    z.push(b[2] + oz);
                    charges.push(-1.0);
                }
            }
        }
    }
    (x, y, z, charges, l_box)
}

fn bench_update_all(c: &mut Criterion) {
    let mut group = c.benchmark_group("update_all");
    for n_cells in [2, 3, 4] {
        let (x, y, z, charges, l_box) = nacl_system(n_cells);
        let n = x.len();
        let mut ewald = EwaldReciprocal::new([l_box; 3], l_box / 2.0, 1e-5, None);
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("N={n}")),
            &(),
            |b, _| {
                b.iter(|| {
                    ewald.update_all(
                        black_box(&x),
                        black_box(&y),
                        black_box(&z),
                        black_box(&charges),
                        None,
                    )
                });
            },
        );
    }
    group.finish();
}

fn bench_energy(c: &mut Criterion) {
    let (x, y, z, charges, l_box) = nacl_system(3);
    let mut ewald = EwaldReciprocal::new([l_box; 3], l_box / 2.0, 1e-5, None);
    ewald.update_all(&x, &y, &z, &charges, None);
    let n_k = ewald.num_k_vectors();

    c.bench_function(&format!("energy (N_k={n_k})"), |b| {
        b.iter(|| black_box(ewald.energy()));
    });
}

fn bench_energy_change(c: &mut Criterion) {
    let (x, y, z, charges, l_box) = nacl_system(3);
    let mut ewald = EwaldReciprocal::new([l_box; 3], l_box / 2.0, 1e-5, None);
    ewald.update_all(&x, &y, &z, &charges, None);
    let n_k = ewald.num_k_vectors();
    let old = [x[0], y[0], z[0]];
    let new = [x[0] + 0.1, y[0] + 0.05, z[0] - 0.1];

    c.bench_function(&format!("energy_change (N_k={n_k})"), |b| {
        b.iter(|| black_box(ewald.energy_change(charges[0], black_box(old), black_box(new))));
    });
}

fn bench_update_particle(c: &mut Criterion) {
    let (x, y, z, charges, l_box) = nacl_system(3);
    let mut ewald = EwaldReciprocal::new([l_box; 3], l_box / 2.0, 1e-5, None);
    ewald.update_all(&x, &y, &z, &charges, None);
    let n_k = ewald.num_k_vectors();
    let old = [x[0], y[0], z[0]];
    let new = [x[0] + 0.1, y[0] + 0.05, z[0] - 0.1];

    c.bench_function(&format!("update_particle (N_k={n_k})"), |b| {
        b.iter(|| {
            ewald.update_particle(charges[0], black_box(old), black_box(new));
            // Move back to keep state stable across iterations
            ewald.update_particle(charges[0], new, old);
        });
    });
}

fn bench_add_forces(c: &mut Criterion) {
    let mut group = c.benchmark_group("add_forces");
    for n_cells in [2, 3, 4] {
        let (x, y, z, charges, l_box) = nacl_system(n_cells);
        let n = x.len();
        let mut ewald = EwaldReciprocal::new([l_box; 3], l_box / 2.0, 1e-5, None);
        ewald.update_all(&x, &y, &z, &charges, None);
        let mut fx = vec![0.0; n];
        let mut fy = vec![0.0; n];
        let mut fz = vec![0.0; n];

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("N={n}")),
            &(),
            |b, _| {
                b.iter(|| {
                    fx.fill(0.0);
                    fy.fill(0.0);
                    fz.fill(0.0);
                    ewald.add_forces(
                        black_box(&x),
                        black_box(&y),
                        black_box(&z),
                        black_box(&charges),
                        &mut fx,
                        &mut fy,
                        &mut fz,
                    );
                    black_box(&fx);
                });
            },
        );
    }
    group.finish();
}

fn bench_self_energy(c: &mut Criterion) {
    let (_, _, _, charges, _) = nacl_system(3);
    let ewald = EwaldReciprocal::new([6.0; 3], 3.0, 1e-5, None);

    c.bench_function(&format!("self_energy (N={})", charges.len()), |b| {
        b.iter(|| black_box(ewald.self_energy(black_box(&charges))));
    });
}

fn bench_ipbc_update_all(c: &mut Criterion) {
    let mut group = c.benchmark_group("ipbc_update_all");
    for n_cells in [2, 3, 4] {
        let (x, y, z, charges, l_box) = nacl_system(n_cells);
        let n = x.len();
        let mut ewald = EwaldReciprocal::new([l_box; 3], l_box / 2.0, 1e-5, None)
            .with_policy(EwaldPolicy::IPBC);
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("N={n}")),
            &(),
            |b, _| {
                b.iter(|| {
                    ewald.update_all(
                        black_box(&x),
                        black_box(&y),
                        black_box(&z),
                        black_box(&charges),
                        None,
                    )
                });
            },
        );
    }
    group.finish();
}

fn bench_ipbc_energy_change(c: &mut Criterion) {
    let (x, y, z, charges, l_box) = nacl_system(3);
    let mut ewald =
        EwaldReciprocal::new([l_box; 3], l_box / 2.0, 1e-5, None).with_policy(EwaldPolicy::IPBC);
    ewald.update_all(&x, &y, &z, &charges, None);
    let n_k = ewald.num_k_vectors();
    let old = [x[0], y[0], z[0]];
    let new = [x[0] + 0.1, y[0] + 0.05, z[0] - 0.1];

    c.bench_function(&format!("ipbc_energy_change (N_k={n_k})"), |b| {
        b.iter(|| black_box(ewald.energy_change(charges[0], black_box(old), black_box(new))));
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(50)
        .warm_up_time(std::time::Duration::from_millis(500))
        .measurement_time(std::time::Duration::from_secs(2));
    targets =
        bench_update_all,
        bench_energy,
        bench_energy_change,
        bench_update_particle,
        bench_add_forces,
        bench_self_energy,
        bench_ipbc_update_all,
        bench_ipbc_energy_change,
}
criterion_main!(benches);
