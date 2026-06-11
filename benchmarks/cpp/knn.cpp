// k-NN classification — Machine Learning benchmark (MICAI).
// Same algorithm as the other five languages; shared 64-bit LCG (seed 42);
// SQUARED Euclidean distance (no sqrt) for bit-exactness; vote ties break to
// the lowest class label. Checksum = sum of predicted labels (integer).
#include <cstdio>
#include <vector>
#include <array>

constexpr int M = 50000;
constexpr int Q = 10000;
constexpr int D = 8;
constexpr int K = 15;
constexpr int C = 3;

struct SimpleRNG {
    unsigned long long state;
    double next() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>(state >> 33) / static_cast<double>(1ULL << 31);
    }
};

int main() {
    SimpleRNG rng{42};
    std::vector<std::array<double, D>> train(M);
    std::vector<int> label(M);
    std::vector<std::array<double, D>> query(Q);
    for (int t = 0; t < M; t++) {
        for (int d = 0; d < D; d++) train[t][d] = rng.next();
        label[t] = static_cast<int>(rng.next() * C);
    }
    for (int q = 0; q < Q; q++)
        for (int d = 0; d < D; d++) query[q][d] = rng.next();

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
            if (dist < best_d[K - 1]) {
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
