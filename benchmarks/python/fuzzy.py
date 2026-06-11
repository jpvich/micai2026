# Mamdani fuzzy inference system — Fuzzy Systems benchmark (MICAI).
# Two inputs -> 3 triangular sets each; 9-rule base; max-min aggregation;
# centroid defuzzification over a discretized output domain. Triangular MFs +
# centroid use only +,-,*,/,min,max, so the result is bit-exact across
# languages. Q input pairs drawn from the shared 64-bit LCG (seed 42).
# Checksum = sum of defuzzified outputs (6 dp).
Q = 2000000
NP = 100
NSET = 3
NRULE = 9

SETP = [
    [-0.5, 0.0, 0.5],
    [0.0, 0.5, 1.0],
    [0.5, 1.0, 1.5],
]


class SimpleRNG:
    def __init__(self, seed=42):
        self.state = seed

    def next_double(self):
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return (self.state >> 33) / (1 << 31)


def tri(v, a, b, c):
    left = (v - a) / (b - a)
    right = (c - v) / (c - b)
    m = left if left < right else right
    return m if m > 0.0 else 0.0


def main():
    zval = [0.0] * NP
    os = [[0.0] * NP for _ in range(NSET)]
    for j in range(NP):
        z = j / (NP - 1)
        zval[j] = z
        for s in range(NSET):
            os[s][j] = tri(z, SETP[s][0], SETP[s][1], SETP[s][2])
    outset = [0] * NRULE
    for xi in range(NSET):
        for yi in range(NSET):
            sm = xi + yi
            outset[xi * NSET + yi] = 0 if sm <= 1 else (1 if sm == 2 else 2)

    rng = SimpleRNG(42)
    checksum = 0.0
    mu_x = [0.0] * NSET
    mu_y = [0.0] * NSET
    fs = [0.0] * NRULE
    for q in range(Q):
        x = rng.next_double()
        y = rng.next_double()
        for s in range(NSET):
            mu_x[s] = tri(x, SETP[s][0], SETP[s][1], SETP[s][2])
            mu_y[s] = tri(y, SETP[s][0], SETP[s][1], SETP[s][2])
        for xi in range(NSET):
            for yi in range(NSET):
                a = mu_x[xi]
                b = mu_y[yi]
                fs[xi * NSET + yi] = a if a < b else b
        num = 0.0
        den = 0.0
        for j in range(NP):
            agg = 0.0
            for r in range(NRULE):
                osv = os[outset[r]][j]
                fr = fs[r]
                m = fr if fr < osv else osv
                if m > agg:
                    agg = m
            num += zval[j] * agg
            den += agg
        out = num / den if den > 0.0 else 0.0
        checksum += out
    print(f"{checksum:.6f}")


main()
