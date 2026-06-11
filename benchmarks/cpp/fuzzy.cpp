// Mamdani fuzzy inference system — Fuzzy Systems benchmark (MICAI).
// Two inputs -> 3 triangular sets each; 9-rule base; max-min aggregation;
// centroid defuzzification over a discretized output domain. Triangular MFs +
// centroid use only +,-,*,/,min,max, so the result is bit-exact across
// languages. Q input pairs drawn from the shared 64-bit LCG (seed 42).
// Checksum = sum of defuzzified outputs (6 dp).
#include <cstdio>

constexpr int Q = 2000000;
constexpr int NP = 100;
constexpr int NSET = 3;
constexpr int NRULE = 9;

struct SimpleRNG {
    unsigned long long state;
    double next() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>(state >> 33) / static_cast<double>(1ULL << 31);
    }
};

static double tri(double v, double a, double b, double c) {
    double left = (v - a) / (b - a);
    double right = (c - v) / (c - b);
    double m = left < right ? left : right;
    return m > 0.0 ? m : 0.0;
}

static const double setp[NSET][3] = {
    { -0.5, 0.0, 0.5 },
    {  0.0, 0.5, 1.0 },
    {  0.5, 1.0, 1.5 },
};

int main() {
    double zval[NP];
    double os[NSET][NP];
    for (int j = 0; j < NP; j++) {
        double z = static_cast<double>(j) / static_cast<double>(NP - 1);
        zval[j] = z;
        for (int s = 0; s < NSET; s++)
            os[s][j] = tri(z, setp[s][0], setp[s][1], setp[s][2]);
    }
    int outset[NRULE];
    for (int xi = 0; xi < NSET; xi++)
        for (int yi = 0; yi < NSET; yi++) {
            int sum = xi + yi;
            outset[xi * NSET + yi] = sum <= 1 ? 0 : (sum == 2 ? 1 : 2);
        }

    SimpleRNG rng{42};
    double checksum = 0.0;
    double mu_x[NSET], mu_y[NSET], fs[NRULE];
    for (int q = 0; q < Q; q++) {
        double x = rng.next();
        double y = rng.next();
        for (int s = 0; s < NSET; s++) {
            mu_x[s] = tri(x, setp[s][0], setp[s][1], setp[s][2]);
            mu_y[s] = tri(y, setp[s][0], setp[s][1], setp[s][2]);
        }
        for (int xi = 0; xi < NSET; xi++)
            for (int yi = 0; yi < NSET; yi++) {
                double f = mu_x[xi] < mu_y[yi] ? mu_x[xi] : mu_y[yi];
                fs[xi * NSET + yi] = f;
            }
        double num = 0.0, den = 0.0;
        for (int j = 0; j < NP; j++) {
            double agg = 0.0;
            for (int r = 0; r < NRULE; r++) {
                double osv = os[outset[r]][j];
                double m = fs[r] < osv ? fs[r] : osv;
                if (m > agg) agg = m;
            }
            num += zval[j] * agg;
            den += agg;
        }
        double out = den > 0.0 ? num / den : 0.0;
        checksum += out;
    }
    printf("%.6f\n", checksum);
    return 0;
}
