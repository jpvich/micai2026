// K-Means clustering (Lloyd's algorithm) — Data Mining benchmark (MICAI).
// Same algorithm as the other five languages; shared 64-bit LCG (seed 42);
// SQUARED Euclidean distance (no sqrt) keeps the inner loop bit-exact.
// Assignment ties break to the lowest centroid index (strict `<`).

const N: usize = 100_000;
const D: usize = 4;
const K: usize = 10;
const ITERS: usize = 1000; // sized so C ~0.8s

fn main() {
    let mut state: u64 = 42;
    let mut next_double = || -> f64 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as f64 / (1u64 << 31) as f64
    };

    let mut points = vec![[0.0f64; D]; N];
    for p in points.iter_mut() {
        for d in 0..D {
            p[d] = next_double();
        }
    }

    let mut centroids = vec![[0.0f64; D]; K];
    for k in 0..K {
        centroids[k] = points[k];
    }

    let mut assign = vec![0usize; N];
    let mut counts = vec![0i64; K];
    let mut sums = vec![[0.0f64; D]; K];

    for _ in 0..ITERS {
        for i in 0..N {
            let mut best = 1e300f64;
            let mut bestk = 0usize;
            for k in 0..K {
                let mut dist = 0.0f64;
                for d in 0..D {
                    let diff = points[i][d] - centroids[k][d];
                    dist += diff * diff;
                }
                if dist < best {
                    best = dist;
                    bestk = k;
                }
            }
            assign[i] = bestk;
        }
        for k in 0..K {
            counts[k] = 0;
            sums[k] = [0.0; D];
        }
        for i in 0..N {
            let k = assign[i];
            counts[k] += 1;
            for d in 0..D {
                sums[k][d] += points[i][d];
            }
        }
        for k in 0..K {
            if counts[k] > 0 {
                for d in 0..D {
                    centroids[k][d] = sums[k][d] / counts[k] as f64;
                }
            }
        }
    }

    let mut fingerprint: i64 = 0;
    let mut centroid_sum = 0.0f64;
    for k in 0..K {
        fingerprint += counts[k] * (k as i64 + 1);
        for d in 0..D {
            centroid_sum += centroids[k][d];
        }
    }
    println!("{} {:.6}", fingerprint, centroid_sum);
}
