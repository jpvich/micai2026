// Mamdani fuzzy inference system — Fuzzy Systems benchmark (MICAI).
// Two inputs -> 3 triangular sets each; 9-rule base; max-min aggregation;
// centroid defuzzification over a discretized output domain. Triangular MFs +
// centroid use only +,-,*,/,min,max, so the result is bit-exact across
// languages. Q input pairs drawn from the shared 64-bit LCG (seed 42).
// Checksum = sum of defuzzified outputs (6 dp).

const Q: usize = 2_000_000;
const NP: usize = 100;
const NSET: usize = 3;
const NRULE: usize = 9;

const SETP: [[f64; 3]; NSET] = [
    [-0.5, 0.0, 0.5],
    [0.0, 0.5, 1.0],
    [0.5, 1.0, 1.5],
];

fn tri(v: f64, a: f64, b: f64, c: f64) -> f64 {
    let left = (v - a) / (b - a);
    let right = (c - v) / (c - b);
    let m = if left < right { left } else { right };
    if m > 0.0 { m } else { 0.0 }
}

fn main() {
    let mut state: u64 = 42;
    let mut rnd = || -> f64 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as f64 / (1u64 << 31) as f64
    };

    let mut zval = [0.0f64; NP];
    let mut os = [[0.0f64; NP]; NSET];
    for j in 0..NP {
        let z = j as f64 / (NP - 1) as f64;
        zval[j] = z;
        for s in 0..NSET {
            os[s][j] = tri(z, SETP[s][0], SETP[s][1], SETP[s][2]);
        }
    }
    let mut outset = [0usize; NRULE];
    for xi in 0..NSET {
        for yi in 0..NSET {
            let sum = xi + yi;
            outset[xi * NSET + yi] = if sum <= 1 { 0 } else if sum == 2 { 1 } else { 2 };
        }
    }

    let mut checksum = 0.0f64;
    let mut mu_x = [0.0f64; NSET];
    let mut mu_y = [0.0f64; NSET];
    let mut fs = [0.0f64; NRULE];
    for _q in 0..Q {
        let x = rnd();
        let y = rnd();
        for s in 0..NSET {
            mu_x[s] = tri(x, SETP[s][0], SETP[s][1], SETP[s][2]);
            mu_y[s] = tri(y, SETP[s][0], SETP[s][1], SETP[s][2]);
        }
        for xi in 0..NSET {
            for yi in 0..NSET {
                let f = if mu_x[xi] < mu_y[yi] { mu_x[xi] } else { mu_y[yi] };
                fs[xi * NSET + yi] = f;
            }
        }
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for j in 0..NP {
            let mut agg = 0.0f64;
            for r in 0..NRULE {
                let osv = os[outset[r]][j];
                let m = if fs[r] < osv { fs[r] } else { osv };
                if m > agg {
                    agg = m;
                }
            }
            num += zval[j] * agg;
            den += agg;
        }
        let out = if den > 0.0 { num / den } else { 0.0 };
        checksum += out;
    }
    println!("{:.6}", checksum);
}
