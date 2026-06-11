// K-Means clustering (Lloyd's algorithm) — Data Mining benchmark (MICAI).
// Same algorithm as the other five languages; shared 64-bit LCG (seed 42) for
// identical points; SQUARED Euclidean distance (no sqrt) keeps it bit-exact.
// Assignment ties break to the lowest centroid index (strict `<`).
#include <cstdio>
#include <vector>
#include <array>

constexpr int N = 100000;
constexpr int D = 4;
constexpr int K = 10;
constexpr int ITERS = 1000;  // sized so C ~0.8s

struct SimpleRNG {
    unsigned long long state;
    double next() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>(state >> 33) / static_cast<double>(1ULL << 31);
    }
};

int main() {
    SimpleRNG rng{42};
    std::vector<std::array<double, D>> points(N);
    for (int i = 0; i < N; i++)
        for (int d = 0; d < D; d++)
            points[i][d] = rng.next();

    std::vector<std::array<double, D>> centroids(K);
    for (int k = 0; k < K; k++) centroids[k] = points[k];

    std::vector<int> assign(N);
    std::vector<long long> counts(K);
    std::vector<std::array<double, D>> sums(K);

    for (int it = 0; it < ITERS; it++) {
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
        for (int k = 0; k < K; k++) {
            counts[k] = 0;
            sums[k].fill(0.0);
        }
        for (int i = 0; i < N; i++) {
            int k = assign[i];
            counts[k]++;
            for (int d = 0; d < D; d++) sums[k][d] += points[i][d];
        }
        for (int k = 0; k < K; k++) {
            if (counts[k] > 0)
                for (int d = 0; d < D; d++)
                    centroids[k][d] = sums[k][d] / static_cast<double>(counts[k]);
        }
    }

    long long fingerprint = 0;
    double centroid_sum = 0.0;
    for (int k = 0; k < K; k++) {
        fingerprint += counts[k] * static_cast<long long>(k + 1);
        for (int d = 0; d < D; d++) centroid_sum += centroids[k][d];
    }
    printf("%lld %.6f\n", fingerprint, centroid_sum);
    return 0;
}
