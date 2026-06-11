# MLP training (forward + backprop, full-batch gradient descent) —
# Neural Network benchmark (MICAI). D->H->O, ReLU hidden, linear output, MSE.
# ReLU + MSE use only +,-,*,/ and max (no exp/softmax), so the result is
# bit-exact across languages. Shared 64-bit LCG (seed 42) for identical init.
# Checksum = final-epoch loss + sum of all weights (both 6 dp).
N = 10000
D = 16
H = 64
O = 4
E = 150
LR = 0.01


class SimpleRNG:
    def __init__(self, seed=42):
        self.state = seed

    def next_double(self):
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return (self.state >> 33) / (1 << 31)


def main():
    rng = SimpleRNG(42)
    W1 = [[0.0] * D for _ in range(H)]
    b1 = [0.0] * H
    W2 = [[0.0] * H for _ in range(O)]
    b2 = [0.0] * O
    for h in range(H):
        for d in range(D):
            W1[h][d] = (rng.next_double() * 2.0 - 1.0) * 0.1
    for o in range(O):
        for h in range(H):
            W2[o][h] = (rng.next_double() * 2.0 - 1.0) * 0.1
    x = [[0.0] * D for _ in range(N)]
    target = [[0.0] * O for _ in range(N)]
    for n in range(N):
        for d in range(D):
            x[n][d] = rng.next_double()
        for o in range(O):
            target[n][o] = rng.next_double()

    z1 = [0.0] * H
    a1 = [0.0] * H
    y = [0.0] * O
    dy = [0.0] * O

    scale = LR / N
    final_loss = 0.0
    for _ in range(E):
        gW1 = [[0.0] * D for _ in range(H)]
        gb1 = [0.0] * H
        gW2 = [[0.0] * H for _ in range(O)]
        gb2 = [0.0] * O
        epoch_loss = 0.0
        for n in range(N):
            xn = x[n]
            for h in range(H):
                s = b1[h]
                W1h = W1[h]
                for d in range(D):
                    s += W1h[d] * xn[d]
                z1[h] = s
                a1[h] = s if s > 0.0 else 0.0
            for o in range(O):
                s = b2[o]
                W2o = W2[o]
                for h in range(H):
                    s += W2o[h] * a1[h]
                y[o] = s
            tn = target[n]
            for o in range(O):
                diff = y[o] - tn[o]
                epoch_loss += diff * diff
                dy[o] = 2.0 * diff
            for o in range(O):
                gb2[o] += dy[o]
                gW2o = gW2[o]
                dyo = dy[o]
                for h in range(H):
                    gW2o[h] += dyo * a1[h]
            for h in range(H):
                da = 0.0
                for o in range(O):
                    da += W2[o][h] * dy[o]
                dz = da if z1[h] > 0.0 else 0.0
                gb1[h] += dz
                gW1h = gW1[h]
                for d in range(D):
                    gW1h[d] += dz * xn[d]
        for h in range(H):
            b1[h] -= scale * gb1[h]
            W1h = W1[h]
            gW1h = gW1[h]
            for d in range(D):
                W1h[d] -= scale * gW1h[d]
        for o in range(O):
            b2[o] -= scale * gb2[o]
            W2o = W2[o]
            gW2o = gW2[o]
            for h in range(H):
                W2o[h] -= scale * gW2o[h]
        final_loss = epoch_loss / (N * O)

    wsum = 0.0
    for h in range(H):
        wsum += b1[h]
        for d in range(D):
            wsum += W1[h][d]
    for o in range(O):
        wsum += b2[o]
        for h in range(H):
            wsum += W2[o][h]
    print(f"{final_loss:.6f} {wsum:.6f}")


main()
