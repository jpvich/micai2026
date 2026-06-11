// MLP training (forward + backprop, full-batch gradient descent) —
// Neural Network benchmark (MICAI). D->H->O, ReLU hidden, linear output, MSE.
// ReLU + MSE use only +,-,*,/ and max (no exp/softmax), so the result is
// bit-exact across languages. Shared 64-bit LCG (seed 42) for identical init.
// Checksum = final-epoch loss + sum of all weights (both 6 dp).
package main

import "fmt"

const (
	N  = 10000
	D  = 16
	H  = 64
	O  = 4
	E  = 150
	LR = 0.01
)

type SimpleRNG struct{ state uint64 }

func (r *SimpleRNG) next() float64 {
	r.state = r.state*6364136223846793005 + 1442695040888963407
	return float64(r.state>>33) / float64(uint64(1)<<31)
}

var (
	x      [N][D]float64
	target [N][O]float64
	W1     [H][D]float64
	b1     [H]float64
	W2     [O][H]float64
	b2     [O]float64
	gW1    [H][D]float64
	gb1    [H]float64
	gW2    [O][H]float64
	gb2    [O]float64
	z1     [H]float64
	a1     [H]float64
	y      [O]float64
	dy     [O]float64
)

func main() {
	rng := SimpleRNG{42}
	for h := 0; h < H; h++ {
		for d := 0; d < D; d++ {
			W1[h][d] = (rng.next()*2.0 - 1.0) * 0.1
		}
		b1[h] = 0.0
	}
	for o := 0; o < O; o++ {
		for h := 0; h < H; h++ {
			W2[o][h] = (rng.next()*2.0 - 1.0) * 0.1
		}
		b2[o] = 0.0
	}
	for n := 0; n < N; n++ {
		for d := 0; d < D; d++ {
			x[n][d] = rng.next()
		}
		for o := 0; o < O; o++ {
			target[n][o] = rng.next()
		}
	}

	scale := LR / float64(N)
	finalLoss := 0.0
	for e := 0; e < E; e++ {
		for h := 0; h < H; h++ {
			gb1[h] = 0.0
			for d := 0; d < D; d++ {
				gW1[h][d] = 0.0
			}
		}
		for o := 0; o < O; o++ {
			gb2[o] = 0.0
			for h := 0; h < H; h++ {
				gW2[o][h] = 0.0
			}
		}
		epochLoss := 0.0
		for n := 0; n < N; n++ {
			for h := 0; h < H; h++ {
				s := b1[h]
				for d := 0; d < D; d++ {
					s += W1[h][d] * x[n][d]
				}
				z1[h] = s
				if s > 0.0 {
					a1[h] = s
				} else {
					a1[h] = 0.0
				}
			}
			for o := 0; o < O; o++ {
				s := b2[o]
				for h := 0; h < H; h++ {
					s += W2[o][h] * a1[h]
				}
				y[o] = s
			}
			for o := 0; o < O; o++ {
				diff := y[o] - target[n][o]
				epochLoss += diff * diff
				dy[o] = 2.0 * diff
			}
			for o := 0; o < O; o++ {
				gb2[o] += dy[o]
				for h := 0; h < H; h++ {
					gW2[o][h] += dy[o] * a1[h]
				}
			}
			for h := 0; h < H; h++ {
				da := 0.0
				for o := 0; o < O; o++ {
					da += W2[o][h] * dy[o]
				}
				dz := 0.0
				if z1[h] > 0.0 {
					dz = da
				}
				gb1[h] += dz
				for d := 0; d < D; d++ {
					gW1[h][d] += dz * x[n][d]
				}
			}
		}
		for h := 0; h < H; h++ {
			b1[h] -= scale * gb1[h]
			for d := 0; d < D; d++ {
				W1[h][d] -= scale * gW1[h][d]
			}
		}
		for o := 0; o < O; o++ {
			b2[o] -= scale * gb2[o]
			for h := 0; h < H; h++ {
				W2[o][h] -= scale * gW2[o][h]
			}
		}
		finalLoss = epochLoss / float64(N*O)
	}

	wsum := 0.0
	for h := 0; h < H; h++ {
		wsum += b1[h]
		for d := 0; d < D; d++ {
			wsum += W1[h][d]
		}
	}
	for o := 0; o < O; o++ {
		wsum += b2[o]
		for h := 0; h < H; h++ {
			wsum += W2[o][h]
		}
	}
	fmt.Printf("%.6f %.6f\n", finalLoss, wsum)
}
