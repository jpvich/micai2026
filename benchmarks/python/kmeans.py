# K-Means clustering (Lloyd's algorithm) — Data Mining benchmark (MICAI).
# Same algorithm as the other five languages; shared 64-bit LCG (seed 42);
# SQUARED Euclidean distance (no sqrt) keeps the inner loop bit-exact.
# Assignment ties break to the lowest centroid index (strict `<`).
N = 100000
D = 4
K = 10
ITERS = 1000  # sized so C ~0.8s


class SimpleRNG:
    def __init__(self, seed=42):
        self.state = seed

    def next_double(self):
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return (self.state >> 33) / (1 << 31)


def main():
    rng = SimpleRNG(42)
    points = [[rng.next_double() for _ in range(D)] for _ in range(N)]
    centroids = [points[k][:] for k in range(K)]
    assign = [0] * N
    counts = [0] * K

    for _ in range(ITERS):
        for i in range(N):
            pi = points[i]
            best = 1e300
            bestk = 0
            for k in range(K):
                ck = centroids[k]
                dist = 0.0
                for d in range(D):
                    diff = pi[d] - ck[d]
                    dist += diff * diff
                if dist < best:
                    best = dist
                    bestk = k
            assign[i] = bestk
        for k in range(K):
            counts[k] = 0
        sums = [[0.0] * D for _ in range(K)]
        for i in range(N):
            k = assign[i]
            counts[k] += 1
            pi = points[i]
            sk = sums[k]
            for d in range(D):
                sk[d] += pi[d]
        for k in range(K):
            if counts[k] > 0:
                ck = centroids[k]
                sk = sums[k]
                c = counts[k]
                for d in range(D):
                    ck[d] = sk[d] / c

    fingerprint = 0
    centroid_sum = 0.0
    for k in range(K):
        fingerprint += counts[k] * (k + 1)
        for d in range(D):
            centroid_sum += centroids[k][d]
    print(f"{fingerprint} {centroid_sum:.6f}")


main()
