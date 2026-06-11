/* Genetic Algorithm minimizing the Rosenbrock function —
 * Computational Intelligence benchmark (MICAI). Tournament selection, uniform
 * crossover, uniform mutation. Rosenbrock uses only +,-,* (no cos as in
 * Rastrigin) and mutation is uniform (no Gaussian sqrt/log), so the entire
 * search is bit-exact across IEEE-754 implementations. Every random decision
 * draws from the shared 64-bit LCG (seed 42) in an identical order in all six
 * languages, making the whole evolution deterministic.
 * Checksum = best fitness found + sum of the best individual's genes (6 dp). */
#include <stdio.h>

#define D 30          /* genes per individual */
#define P 5000        /* population size */
#define G 1200        /* generations (sized so C ~1s) */
#define T 3           /* tournament size */
#define MUT_RATE 0.1  /* per-gene mutation probability */
#define MUT_STEP 0.1  /* uniform mutation half-range */

typedef struct { unsigned long long state; } SimpleRNG;

static double next_double(SimpleRNG *rng) {
    rng->state = rng->state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(rng->state >> 33) / (double)(1ULL << 31);
}

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

int main(void) {
    SimpleRNG rng = { .state = 42 };
    for (int p = 0; p < P; p++)
        for (int d = 0; d < D; d++)
            pop[p][d] = (next_double(&rng) * 2.0 - 1.0) * 5.0;  /* genes in [-5, 5] */

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
        for (int d = 0; d < D; d++) newpop[0][d] = best_genes[d];  /* elitism */
        for (int i = 1; i < P; i++) {
            int a = (int)(next_double(&rng) * P);
            for (int t = 1; t < T; t++) {
                int idx = (int)(next_double(&rng) * P);
                if (fitness[idx] < fitness[a]) a = idx;
            }
            int b = (int)(next_double(&rng) * P);
            for (int t = 1; t < T; t++) {
                int idx = (int)(next_double(&rng) * P);
                if (fitness[idx] < fitness[b]) b = idx;
            }
            for (int d = 0; d < D; d++)
                newpop[i][d] = (next_double(&rng) < 0.5) ? pop[a][d] : pop[b][d];
            for (int d = 0; d < D; d++)
                if (next_double(&rng) < MUT_RATE)
                    newpop[i][d] += (next_double(&rng) * 2.0 - 1.0) * MUT_STEP;
        }
        for (int p = 0; p < P; p++)
            for (int d = 0; d < D; d++) pop[p][d] = newpop[p][d];
    }

    double gsum = 0.0;
    for (int d = 0; d < D; d++) gsum += best_genes[d];
    printf("%.6f %.6f\n", best_fit, gsum);
    return 0;
}
