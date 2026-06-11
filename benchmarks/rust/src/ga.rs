// Genetic Algorithm minimizing the Rosenbrock function —
// Computational Intelligence benchmark (MICAI). Tournament selection, uniform
// crossover, uniform mutation. Rosenbrock + uniform mutation use only +,-,*,/,
// so the search is bit-exact across languages. Every random decision draws from
// the shared 64-bit LCG (seed 42) in identical order in all six languages.
// Checksum = best fitness found + sum of the best individual's genes (6 dp).

const D: usize = 30;
const P: usize = 5000;
const G: usize = 1200;
const T: usize = 3;
const MUT_RATE: f64 = 0.1;
const MUT_STEP: f64 = 0.1;

fn rosenbrock(x: &[f64; D]) -> f64 {
    let mut f = 0.0;
    for i in 0..D - 1 {
        let a = x[i + 1] - x[i] * x[i];
        let b = 1.0 - x[i];
        f += 100.0 * a * a + b * b;
    }
    f
}

fn main() {
    let mut state: u64 = 42;
    let mut rnd = || -> f64 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as f64 / (1u64 << 31) as f64
    };

    let mut pop = vec![[0.0f64; D]; P];
    let mut newpop = vec![[0.0f64; D]; P];
    let mut fitness = vec![0.0f64; P];
    let mut best_genes = [0.0f64; D];

    for p in 0..P {
        for d in 0..D {
            pop[p][d] = (rnd() * 2.0 - 1.0) * 5.0;
        }
    }

    let mut best_fit = 1e300f64;
    for _g in 0..G {
        for p in 0..P {
            let f = rosenbrock(&pop[p]);
            fitness[p] = f;
            if f < best_fit {
                best_fit = f;
                best_genes = pop[p];
            }
        }
        newpop[0] = best_genes;
        for i in 1..P {
            let mut a = (rnd() * P as f64) as usize;
            for _t in 1..T {
                let idx = (rnd() * P as f64) as usize;
                if fitness[idx] < fitness[a] {
                    a = idx;
                }
            }
            let mut b = (rnd() * P as f64) as usize;
            for _t in 1..T {
                let idx = (rnd() * P as f64) as usize;
                if fitness[idx] < fitness[b] {
                    b = idx;
                }
            }
            for d in 0..D {
                newpop[i][d] = if rnd() < 0.5 { pop[a][d] } else { pop[b][d] };
            }
            for d in 0..D {
                if rnd() < MUT_RATE {
                    newpop[i][d] += (rnd() * 2.0 - 1.0) * MUT_STEP;
                }
            }
        }
        for p in 0..P {
            pop[p] = newpop[p];
        }
    }

    let mut gsum = 0.0;
    for d in 0..D {
        gsum += best_genes[d];
    }
    println!("{:.6} {:.6}", best_fit, gsum);
}
