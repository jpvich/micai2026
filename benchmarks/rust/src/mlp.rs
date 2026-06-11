// MLP training (forward + backprop, full-batch gradient descent) —
// Neural Network benchmark (MICAI). D->H->O, ReLU hidden, linear output, MSE.
// ReLU + MSE use only +,-,*,/ and max (no exp/softmax), so the result is
// bit-exact across languages. Shared 64-bit LCG (seed 42) for identical init.
// Checksum = final-epoch loss + sum of all weights (both 6 dp).

const N: usize = 10_000;
const D: usize = 16;
const H: usize = 64;
const O: usize = 4;
const E: usize = 150;
const LR: f64 = 0.01;

fn main() {
    let mut state: u64 = 42;
    let mut next_double = || -> f64 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as f64 / (1u64 << 31) as f64
    };

    let mut w1 = vec![[0.0f64; D]; H];
    let mut b1 = [0.0f64; H];
    let mut w2 = vec![[0.0f64; H]; O];
    let mut b2 = [0.0f64; O];
    for h in 0..H {
        for d in 0..D {
            w1[h][d] = (next_double() * 2.0 - 1.0) * 0.1;
        }
    }
    for o in 0..O {
        for h in 0..H {
            w2[o][h] = (next_double() * 2.0 - 1.0) * 0.1;
        }
    }
    let mut x = vec![[0.0f64; D]; N];
    let mut target = vec![[0.0f64; O]; N];
    for n in 0..N {
        for d in 0..D {
            x[n][d] = next_double();
        }
        for o in 0..O {
            target[n][o] = next_double();
        }
    }

    let mut gw1 = vec![[0.0f64; D]; H];
    let mut gb1 = [0.0f64; H];
    let mut gw2 = vec![[0.0f64; H]; O];
    let mut gb2 = [0.0f64; O];
    let mut z1 = [0.0f64; H];
    let mut a1 = [0.0f64; H];
    let mut y = [0.0f64; O];
    let mut dy = [0.0f64; O];

    let scale = LR / N as f64;
    let mut final_loss = 0.0f64;
    for _ in 0..E {
        for h in 0..H {
            gb1[h] = 0.0;
            for d in 0..D {
                gw1[h][d] = 0.0;
            }
        }
        for o in 0..O {
            gb2[o] = 0.0;
            for h in 0..H {
                gw2[o][h] = 0.0;
            }
        }
        let mut epoch_loss = 0.0f64;
        for n in 0..N {
            for h in 0..H {
                let mut s = b1[h];
                for d in 0..D {
                    s += w1[h][d] * x[n][d];
                }
                z1[h] = s;
                a1[h] = if s > 0.0 { s } else { 0.0 };
            }
            for o in 0..O {
                let mut s = b2[o];
                for h in 0..H {
                    s += w2[o][h] * a1[h];
                }
                y[o] = s;
            }
            for o in 0..O {
                let diff = y[o] - target[n][o];
                epoch_loss += diff * diff;
                dy[o] = 2.0 * diff;
            }
            for o in 0..O {
                gb2[o] += dy[o];
                for h in 0..H {
                    gw2[o][h] += dy[o] * a1[h];
                }
            }
            for h in 0..H {
                let mut da = 0.0f64;
                for o in 0..O {
                    da += w2[o][h] * dy[o];
                }
                let dz = if z1[h] > 0.0 { da } else { 0.0 };
                gb1[h] += dz;
                for d in 0..D {
                    gw1[h][d] += dz * x[n][d];
                }
            }
        }
        for h in 0..H {
            b1[h] -= scale * gb1[h];
            for d in 0..D {
                w1[h][d] -= scale * gw1[h][d];
            }
        }
        for o in 0..O {
            b2[o] -= scale * gb2[o];
            for h in 0..H {
                w2[o][h] -= scale * gw2[o][h];
            }
        }
        final_loss = epoch_loss / (N * O) as f64;
    }

    let mut wsum = 0.0f64;
    for h in 0..H {
        wsum += b1[h];
        for d in 0..D {
            wsum += w1[h][d];
        }
    }
    for o in 0..O {
        wsum += b2[o];
        for h in 0..H {
            wsum += w2[o][h];
        }
    }
    println!("{:.6} {:.6}", final_loss, wsum);
}
