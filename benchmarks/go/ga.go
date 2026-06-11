// Genetic Algorithm minimizing the Rosenbrock function —
// Computational Intelligence benchmark (MICAI). Tournament selection, uniform
// crossover, uniform mutation. Rosenbrock + uniform mutation use only +,-,*,/,
// so the search is bit-exact across languages. Every random decision draws from
// the shared 64-bit LCG (seed 42) in identical order in all six languages.
// Checksum = best fitness found + sum of the best individual's genes (6 dp).
package main

import "fmt"

const (
	D        = 30
	P        = 5000
	G        = 1200
	T        = 3
	MUT_RATE = 0.1
	MUT_STEP = 0.1
)

type SimpleRNG struct{ state uint64 }

func (r *SimpleRNG) next() float64 {
	r.state = r.state*6364136223846793005 + 1442695040888963407
	return float64(r.state>>33) / float64(uint64(1)<<31)
}

var (
	pop       [P][D]float64
	newpop    [P][D]float64
	fitness   [P]float64
	bestGenes [D]float64
)

func rosenbrock(x *[D]float64) float64 {
	f := 0.0
	for i := 0; i < D-1; i++ {
		a := x[i+1] - x[i]*x[i]
		b := 1.0 - x[i]
		f += 100.0*a*a + b*b
	}
	return f
}

func main() {
	rng := SimpleRNG{42}
	for p := 0; p < P; p++ {
		for d := 0; d < D; d++ {
			pop[p][d] = (rng.next()*2.0 - 1.0) * 5.0
		}
	}

	bestFit := 1e300
	for g := 0; g < G; g++ {
		for p := 0; p < P; p++ {
			f := rosenbrock(&pop[p])
			fitness[p] = f
			if f < bestFit {
				bestFit = f
				for d := 0; d < D; d++ {
					bestGenes[d] = pop[p][d]
				}
			}
		}
		for d := 0; d < D; d++ {
			newpop[0][d] = bestGenes[d]
		}
		for i := 1; i < P; i++ {
			a := int(rng.next() * P)
			for t := 1; t < T; t++ {
				idx := int(rng.next() * P)
				if fitness[idx] < fitness[a] {
					a = idx
				}
			}
			b := int(rng.next() * P)
			for t := 1; t < T; t++ {
				idx := int(rng.next() * P)
				if fitness[idx] < fitness[b] {
					b = idx
				}
			}
			for d := 0; d < D; d++ {
				if rng.next() < 0.5 {
					newpop[i][d] = pop[a][d]
				} else {
					newpop[i][d] = pop[b][d]
				}
			}
			for d := 0; d < D; d++ {
				if rng.next() < MUT_RATE {
					newpop[i][d] += (rng.next()*2.0 - 1.0) * MUT_STEP
				}
			}
		}
		for p := 0; p < P; p++ {
			for d := 0; d < D; d++ {
				pop[p][d] = newpop[p][d]
			}
		}
	}

	gsum := 0.0
	for d := 0; d < D; d++ {
		gsum += bestGenes[d]
	}
	fmt.Printf("%.6f %.6f\n", bestFit, gsum)
}
