// Genetic Algorithm minimizing the Rosenbrock function —
// Computational Intelligence benchmark (MICAI). Tournament selection, uniform
// crossover, uniform mutation. Rosenbrock + uniform mutation use only +,-,*,/,
// so the search is bit-exact across languages. Every random decision draws from
// the shared 64-bit LCG (seed 42) in identical order in all six languages.
// Checksum = best fitness found + sum of the best individual's genes (6 dp).
#include <cstdio>

constexpr int D = 30;
constexpr int P = 5000;
constexpr int G = 1200;
constexpr int T = 3;
constexpr double MUT_RATE = 0.1;
constexpr double MUT_STEP = 0.1;

struct SimpleRNG {
    unsigned long long state;
    double next() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>(state >> 33) / static_cast<double>(1ULL << 31);
    }
};

static double pop[P][D];
static double newpop[P][D];
static double fitness[P];
static double best_genes[D];

static double rosenbrock(const double *x) {
    double f = 0.0;
    for (int i = 0; i < D - 1; i++) {
        double a = x[i + 1] - x[i] * x[i];
        double b = 1.0 - x[i];
        f += 100.0 * a * a + b * b;
    }
    return f;
}

int main() {
    SimpleRNG rng{42};
    for (int p = 0; p < P; p++)
        for (int d = 0; d < D; d++)
            pop[p][d] = (rng.next() * 2.0 - 1.0) * 5.0;

    double best_fit = 1e300;
    for (int g = 0; g < G; g++) {
        for (int p = 0; p < P; p++) {
            double f = rosenbrock(pop[p]);
            fitness[p] = f;
            if (f < best_fit) {
                best_fit = f;
                for (int d = 0; d < D; d++) best_genes[d] = pop[p][d];
            }
        }
        for (int d = 0; d < D; d++) newpop[0][d] = best_genes[d];
        for (int i = 1; i < P; i++) {
            int a = static_cast<int>(rng.next() * P);
            for (int t = 1; t < T; t++) {
                int idx = static_cast<int>(rng.next() * P);
                if (fitness[idx] < fitness[a]) a = idx;
            }
            int b = static_cast<int>(rng.next() * P);
            for (int t = 1; t < T; t++) {
                int idx = static_cast<int>(rng.next() * P);
                if (fitness[idx] < fitness[b]) b = idx;
            }
            for (int d = 0; d < D; d++)
                newpop[i][d] = (rng.next() < 0.5) ? pop[a][d] : pop[b][d];
            for (int d = 0; d < D; d++)
                if (rng.next() < MUT_RATE)
                    newpop[i][d] += (rng.next() * 2.0 - 1.0) * MUT_STEP;
        }
        for (int p = 0; p < P; p++)
            for (int d = 0; d < D; d++) pop[p][d] = newpop[p][d];
    }

    double gsum = 0.0;
    for (int d = 0; d < D; d++) gsum += best_genes[d];
    printf("%.6f %.6f\n", best_fit, gsum);
    return 0;
}
