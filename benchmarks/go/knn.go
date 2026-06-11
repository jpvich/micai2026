// k-NN classification — Machine Learning benchmark (MICAI).
// Same algorithm as the other five languages; shared 64-bit LCG (seed 42);
// SQUARED Euclidean distance (no sqrt) for bit-exactness; vote ties break to
// the lowest class label. Checksum = sum of predicted labels (integer).
package main

import "fmt"

const (
	M = 50000
	Q = 10000
	D = 8
	K = 15
	C = 3
)

type SimpleRNG struct{ state uint64 }

func (r *SimpleRNG) next() float64 {
	r.state = r.state*6364136223846793005 + 1442695040888963407
	return float64(r.state>>33) / float64(uint64(1)<<31)
}

func main() {
	rng := SimpleRNG{42}
	train := make([][D]float64, M)
	label := make([]int, M)
	query := make([][D]float64, Q)
	for t := 0; t < M; t++ {
		for d := 0; d < D; d++ {
			train[t][d] = rng.next()
		}
		label[t] = int(rng.next() * C)
	}
	for q := 0; q < Q; q++ {
		for d := 0; d < D; d++ {
			query[q][d] = rng.next()
		}
	}

	var checksum int64 = 0
	var bestD [K]float64
	var bestL [K]int
	for q := 0; q < Q; q++ {
		for j := 0; j < K; j++ {
			bestD[j] = 1e300
			bestL[j] = 0
		}
		for t := 0; t < M; t++ {
			dist := 0.0
			for d := 0; d < D; d++ {
				diff := query[q][d] - train[t][d]
				dist += diff * diff
			}
			if dist < bestD[K-1] {
				p := K - 1
				for p > 0 && dist < bestD[p-1] {
					bestD[p] = bestD[p-1]
					bestL[p] = bestL[p-1]
					p--
				}
				bestD[p] = dist
				bestL[p] = label[t]
			}
		}
		var votes [C]int
		for j := 0; j < K; j++ {
			votes[bestL[j]]++
		}
		pred := 0
		for c := 1; c < C; c++ {
			if votes[c] > votes[pred] {
				pred = c
			}
		}
		checksum += int64(pred)
	}
	fmt.Printf("%d\n", checksum)
}
