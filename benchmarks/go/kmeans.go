// K-Means clustering (Lloyd's algorithm) — Data Mining benchmark (MICAI).
// Same algorithm as the other five languages; shared 64-bit LCG (seed 42);
// SQUARED Euclidean distance (no sqrt) keeps the inner loop bit-exact.
// Assignment ties break to the lowest centroid index (strict `<`).
package main

import "fmt"

const (
	N     = 100000
	D     = 4
	K     = 10
	ITERS = 1000 // sized so C ~0.8s
)

type SimpleRNG struct{ state uint64 }

func (r *SimpleRNG) next() float64 {
	r.state = r.state*6364136223846793005 + 1442695040888963407
	return float64(r.state>>33) / float64(uint64(1)<<31)
}

func main() {
	rng := SimpleRNG{42}
	points := make([][D]float64, N)
	for i := 0; i < N; i++ {
		for d := 0; d < D; d++ {
			points[i][d] = rng.next()
		}
	}

	centroids := make([][D]float64, K)
	for k := 0; k < K; k++ {
		centroids[k] = points[k]
	}

	assign := make([]int, N)
	counts := make([]int64, K)
	sums := make([][D]float64, K)

	for it := 0; it < ITERS; it++ {
		for i := 0; i < N; i++ {
			best := 1e300
			bestk := 0
			for k := 0; k < K; k++ {
				dist := 0.0
				for d := 0; d < D; d++ {
					diff := points[i][d] - centroids[k][d]
					dist += diff * diff
				}
				if dist < best {
					best = dist
					bestk = k
				}
			}
			assign[i] = bestk
		}
		for k := 0; k < K; k++ {
			counts[k] = 0
			sums[k] = [D]float64{}
		}
		for i := 0; i < N; i++ {
			k := assign[i]
			counts[k]++
			for d := 0; d < D; d++ {
				sums[k][d] += points[i][d]
			}
		}
		for k := 0; k < K; k++ {
			if counts[k] > 0 {
				for d := 0; d < D; d++ {
					centroids[k][d] = sums[k][d] / float64(counts[k])
				}
			}
		}
	}

	var fingerprint int64 = 0
	centroidSum := 0.0
	for k := 0; k < K; k++ {
		fingerprint += counts[k] * int64(k+1)
		for d := 0; d < D; d++ {
			centroidSum += centroids[k][d]
		}
	}
	fmt.Printf("%d %.6f\n", fingerprint, centroidSum)
}
