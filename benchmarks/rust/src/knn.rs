// k-NN classification — Machine Learning benchmark (MICAI).
// Same algorithm as the other five languages; shared 64-bit LCG (seed 42);
// SQUARED Euclidean distance (no sqrt) for bit-exactness; vote ties break to
// the lowest class label. Checksum = sum of predicted labels (integer).

const M: usize = 50_000;
const Q: usize = 10_000;
const D: usize = 8;
const K: usize = 15;
const C: usize = 3;

fn main() {
    let mut state: u64 = 42;
    let mut next_double = || -> f64 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as f64 / (1u64 << 31) as f64
    };

    let mut train = vec![[0.0f64; D]; M];
    let mut label = vec![0usize; M];
    let mut query = vec![[0.0f64; D]; Q];
    for t in 0..M {
        for d in 0..D {
            train[t][d] = next_double();
        }
        label[t] = (next_double() * C as f64) as usize;
    }
    for q in 0..Q {
        for d in 0..D {
            query[q][d] = next_double();
        }
    }

    let mut checksum: i64 = 0;
    let mut best_d = [0.0f64; K];
    let mut best_l = [0usize; K];
    for q in 0..Q {
        for j in 0..K {
            best_d[j] = 1e300;
            best_l[j] = 0;
        }
        for t in 0..M {
            let mut dist = 0.0f64;
            for d in 0..D {
                let diff = query[q][d] - train[t][d];
                dist += diff * diff;
            }
            if dist < best_d[K - 1] {
                let mut p = K - 1;
                while p > 0 && dist < best_d[p - 1] {
                    best_d[p] = best_d[p - 1];
                    best_l[p] = best_l[p - 1];
                    p -= 1;
                }
                best_d[p] = dist;
                best_l[p] = label[t];
            }
        }
        let mut votes = [0i32; C];
        for j in 0..K {
            votes[best_l[j]] += 1;
        }
        let mut pred = 0usize;
        for c in 1..C {
            if votes[c] > votes[pred] {
                pred = c;
            }
        }
        checksum += pred as i64;
    }
    println!("{}", checksum);
}
