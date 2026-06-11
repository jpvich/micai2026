# Genetic Algorithm minimizing the Rosenbrock function —
# Computational Intelligence benchmark (MICAI). Tournament selection, uniform
# crossover, uniform mutation. Rosenbrock + uniform mutation use only +,-,*,/,
# so the search is bit-exact across languages. Every random decision draws from
# the shared 64-bit LCG (seed 42) in identical order in all six languages.
# Checksum = best fitness found + sum of the best individual's genes (6 dp).
D = 30
P = 5000
G = 1200
T = 3
MUT_RATE = 0.1
MUT_STEP = 0.1


class SimpleRNG:
    def __init__(self, seed=42):
        self.state = seed

    def next_double(self):
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return (self.state >> 33) / (1 << 31)


def rosenbrock(x):
    f = 0.0
    for i in range(D - 1):
        a = x[i + 1] - x[i] * x[i]
        b = 1.0 - x[i]
        f += 100.0 * a * a + b * b
    return f


def main():
    rng = SimpleRNG(42)
    pop = [[(rng.next_double() * 2.0 - 1.0) * 5.0 for _ in range(D)] for _ in range(P)]
    newpop = [[0.0] * D for _ in range(P)]
    fitness = [0.0] * P
    best_genes = [0.0] * D

    best_fit = 1e300
    for g in range(G):
        for p in range(P):
            f = rosenbrock(pop[p])
            fitness[p] = f
            if f < best_fit:
                best_fit = f
                best_genes = pop[p][:]
        newpop[0] = best_genes[:]
        for i in range(1, P):
            a = int(rng.next_double() * P)
            for t in range(1, T):
                idx = int(rng.next_double() * P)
                if fitness[idx] < fitness[a]:
                    a = idx
            b = int(rng.next_double() * P)
            for t in range(1, T):
                idx = int(rng.next_double() * P)
                if fitness[idx] < fitness[b]:
                    b = idx
            ni = newpop[i]
            pa = pop[a]
            pb = pop[b]
            for d in range(D):
                ni[d] = pa[d] if rng.next_double() < 0.5 else pb[d]
            for d in range(D):
                if rng.next_double() < MUT_RATE:
                    ni[d] += (rng.next_double() * 2.0 - 1.0) * MUT_STEP
        # swap populations (copy values to keep pop/newpop distinct buffers)
        for p in range(P):
            pop[p] = newpop[p][:]

    gsum = 0.0
    for d in range(D):
        gsum += best_genes[d]
    print(f"{best_fit:.6f} {gsum:.6f}")


main()
