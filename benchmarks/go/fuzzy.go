// Mamdani fuzzy inference system — Fuzzy Systems benchmark (MICAI).
// Two inputs -> 3 triangular sets each; 9-rule base; max-min aggregation;
// centroid defuzzification over a discretized output domain. Triangular MFs +
// centroid use only +,-,*,/,min,max, so the result is bit-exact across
// languages. Q input pairs drawn from the shared 64-bit LCG (seed 42).
// Checksum = sum of defuzzified outputs (6 dp).
package main

import "fmt"

const (
	Q     = 2000000
	NP    = 100
	NSET  = 3
	NRULE = 9
)

type SimpleRNG struct{ state uint64 }

func (r *SimpleRNG) next() float64 {
	r.state = r.state*6364136223846793005 + 1442695040888963407
	return float64(r.state>>33) / float64(uint64(1)<<31)
}

func tri(v, a, b, c float64) float64 {
	left := (v - a) / (b - a)
	right := (c - v) / (c - b)
	m := left
	if right < left {
		m = right
	}
	if m > 0.0 {
		return m
	}
	return 0.0
}

var setp = [NSET][3]float64{
	{-0.5, 0.0, 0.5},
	{0.0, 0.5, 1.0},
	{0.5, 1.0, 1.5},
}

func main() {
	var zval [NP]float64
	var os [NSET][NP]float64
	for j := 0; j < NP; j++ {
		z := float64(j) / float64(NP-1)
		zval[j] = z
		for s := 0; s < NSET; s++ {
			os[s][j] = tri(z, setp[s][0], setp[s][1], setp[s][2])
		}
	}
	var outset [NRULE]int
	for xi := 0; xi < NSET; xi++ {
		for yi := 0; yi < NSET; yi++ {
			sum := xi + yi
			if sum <= 1 {
				outset[xi*NSET+yi] = 0
			} else if sum == 2 {
				outset[xi*NSET+yi] = 1
			} else {
				outset[xi*NSET+yi] = 2
			}
		}
	}

	rng := SimpleRNG{42}
	checksum := 0.0
	var muX, muY [NSET]float64
	var fs [NRULE]float64
	for q := 0; q < Q; q++ {
		x := rng.next()
		y := rng.next()
		for s := 0; s < NSET; s++ {
			muX[s] = tri(x, setp[s][0], setp[s][1], setp[s][2])
			muY[s] = tri(y, setp[s][0], setp[s][1], setp[s][2])
		}
		for xi := 0; xi < NSET; xi++ {
			for yi := 0; yi < NSET; yi++ {
				f := muX[xi]
				if muY[yi] < f {
					f = muY[yi]
				}
				fs[xi*NSET+yi] = f
			}
		}
		num, den := 0.0, 0.0
		for j := 0; j < NP; j++ {
			agg := 0.0
			for r := 0; r < NRULE; r++ {
				osv := os[outset[r]][j]
				m := fs[r]
				if osv < m {
					m = osv
				}
				if m > agg {
					agg = m
				}
			}
			num += zval[j] * agg
			den += agg
		}
		out := 0.0
		if den > 0.0 {
			out = num / den
		}
		checksum += out
	}
	fmt.Printf("%.6f\n", checksum)
}
