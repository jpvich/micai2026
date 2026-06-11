/* K-Means clustering (Lloyd's algorithm) — Data Mining benchmark (MICAI).
 * Deterministic across languages: a shared 64-bit LCG (seed 42) generates the
 * same N points everywhere. Nearest-centroid uses SQUARED Euclidean distance
 * (no sqrt) so the inner loop is bit-exact across IEEE-754 implementations.
 * Ties in assignment break to the lowest centroid index (strict `<`). */
#include <stdio.h>

#define N 100000   /* points */
#define D 4        /* dimensions */
#define K 10       /* clusters */
#define ITERS 1000 /* fixed iterations (no early-exit, for determinism); sized so C ~0.8s */

typedef struct { unsigned long long state; } SimpleRNG;

static double next_double(SimpleRNG *rng) {
    rng->state = rng->state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(rng->state >> 33) / (double)(1ULL << 31);
}

static double points[N][D];
static double centroids[K][D];
static double sums[K][D];
static long long counts[K];
static int assign[N];

int main(void) {
    SimpleRNG rng = { .state = 42 };
    for (int i = 0; i < N; i++)
        for (int d = 0; d < D; d++)
            points[i][d] = next_double(&rng);

    /* initialize centroids to the first K points */
    for (int k = 0; k < K; k++)
        for (int d = 0; d < D; d++)
            centroids[k][d] = points[k][d];

    for (int it = 0; it < ITERS; it++) {
        /* assignment step */
        for (int i = 0; i < N; i++) {
            double best = 1e300;
            int bestk = 0;
            for (int k = 0; k < K; k++) {
                double dist = 0.0;
                for (int d = 0; d < D; d++) {
                    double diff = points[i][d] - centroids[k][d];
                    dist += diff * diff;
                }
                if (dist < best) { best = dist; bestk = k; }
            }
            assign[i] = bestk;
        }
        /* update step */
        for (int k = 0; k < K; k++) {
            counts[k] = 0;
            for (int d = 0; d < D; d++) sums[k][d] = 0.0;
        }
        for (int i = 0; i < N; i++) {
            int k = assign[i];
            counts[k]++;
            for (int d = 0; d < D; d++) sums[k][d] += points[i][d];
        }
        for (int k = 0; k < K; k++) {
            if (counts[k] > 0)
                for (int d = 0; d < D; d++)
                    centroids[k][d] = sums[k][d] / (double)counts[k];
        }
    }

    long long fingerprint = 0;
    double centroid_sum = 0.0;
    for (int k = 0; k < K; k++) {
        fingerprint += counts[k] * (long long)(k + 1);
        for (int d = 0; d < D; d++) centroid_sum += centroids[k][d];
    }
    printf("%lld %.6f\n", fingerprint, centroid_sum);
    return 0;
}
