# k-NN classification — Machine Learning benchmark (MICAI).
# Same algorithm as the other five languages; shared 64-bit LCG (seed 42);
# SQUARED Euclidean distance (no sqrt) for bit-exactness; vote ties break to
# the lowest class label. Checksum = sum of predicted labels (integer).
M = 50000
Q = 10000
D = 8
K = 15
C = 3


class SimpleRNG:
    def __init__(self, seed=42):
        self.state = seed

    def next_double(self):
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return (self.state >> 33) / (1 << 31)


def main():
    rng = SimpleRNG(42)
    train = []
    label = []
    for _ in range(M):
        train.append([rng.next_double() for _ in range(D)])
        label.append(int(rng.next_double() * C))
    query = [[rng.next_double() for _ in range(D)] for _ in range(Q)]

    checksum = 0
    for q in range(Q):
        qv = query[q]
        best_d = [1e300] * K
        best_l = [0] * K
        for t in range(M):
            tv = train[t]
            dist = 0.0
            for d in range(D):
                diff = qv[d] - tv[d]
                dist += diff * diff
            if dist < best_d[K - 1]:
                p = K - 1
                while p > 0 and dist < best_d[p - 1]:
                    best_d[p] = best_d[p - 1]
                    best_l[p] = best_l[p - 1]
                    p -= 1
                best_d[p] = dist
                best_l[p] = label[t]
        votes = [0] * C
        for j in range(K):
            votes[best_l[j]] += 1
        pred = 0
        for c in range(1, C):
            if votes[c] > votes[pred]:
                pred = c
        checksum += pred
    print(checksum)


main()
