/* k-NN classification — Machine Learning benchmark (MICAI).
 * Deterministic across languages: shared 64-bit LCG (seed 42) generates the
 * same M labeled training points and Q query points everywhere. Nearest
 * neighbours use SQUARED Euclidean distance (no sqrt) for bit-exactness.
 * Majority-vote ties break to the lowest class label (strict `>`).
 * Checksum = sum of predicted labels over all queries (integer, bit-exact). */
#include <stdio.h>

#define M 50000   /* training points */
#define Q 10000   /* query points */
#define D 8       /* dimensions */
#define K 15      /* neighbours */
#define C 3       /* classes */

typedef struct { unsigned long long state; } SimpleRNG;

static double next_double(SimpleRNG *rng) {
    rng->state = rng->state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(rng->state >> 33) / (double)(1ULL << 31);
}

static double train[M][D];
static int label[M];
static double query[Q][D];

int main(void) {
    SimpleRNG rng = { .state = 42 };
    for (int t = 0; t < M; t++) {
        for (int d = 0; d < D; d++) train[t][d] = next_double(&rng);
        label[t] = (int)(next_double(&rng) * C);
    }
    for (int q = 0; q < Q; q++)
        for (int d = 0; d < D; d++) query[q][d] = next_double(&rng);

    long long checksum = 0;
    double best_d[K];
    int best_l[K];
    for (int q = 0; q < Q; q++) {
        for (int j = 0; j < K; j++) { best_d[j] = 1e300; best_l[j] = 0; }
        for (int t = 0; t < M; t++) {
            double dist = 0.0;
            for (int d = 0; d < D; d++) {
                double diff = query[q][d] - train[t][d];
                dist += diff * diff;
            }
            if (dist < best_d[K - 1]) {  /* sorted-insert into k-best (ascending) */
                int p = K - 1;
                while (p > 0 && dist < best_d[p - 1]) {
                    best_d[p] = best_d[p - 1];
                    best_l[p] = best_l[p - 1];
                    p--;
                }
                best_d[p] = dist;
                best_l[p] = label[t];
            }
        }
        int votes[C];
        for (int c = 0; c < C; c++) votes[c] = 0;
        for (int j = 0; j < K; j++) votes[best_l[j]]++;
        int pred = 0;
        for (int c = 1; c < C; c++) if (votes[c] > votes[pred]) pred = c;
        checksum += pred;
    }
    printf("%lld\n", checksum);
    return 0;
}
