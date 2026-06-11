/* Mamdani fuzzy inference system — Fuzzy Systems benchmark (MICAI).
 * Two inputs, each fuzzified into 3 triangular sets (Low/Med/High); a 9-rule
 * base (one per input-set pair); max-min aggregation; centroid defuzzification
 * over a discretized output domain. Triangular membership functions and the
 * centroid use only +,-,*,/,min,max (no transcendental functions), so the
 * result is bit-exact across IEEE-754 implementations. Q input pairs are drawn
 * from the shared 64-bit LCG (seed 42).
 * Checksum = sum of defuzzified outputs over all input pairs (6 dp). */
#include <stdio.h>

#define Q 2000000  /* input pairs (sized to balance C signal vs Python cost) */
#define NP 100     /* output-domain discretization points */
#define NSET 3     /* fuzzy sets per variable */
#define NRULE 9    /* 3 x 3 rule base */

typedef struct { unsigned long long state; } SimpleRNG;

static double next_double(SimpleRNG *rng) {
    rng->state = rng->state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(rng->state >> 33) / (double)(1ULL << 31);
}

/* triangular membership: peak at b, zero outside (a, c) */
static double tri(double v, double a, double b, double c) {
    double left = (v - a) / (b - a);
    double right = (c - v) / (c - b);
    double m = left < right ? left : right;
    return m > 0.0 ? m : 0.0;
}

/* set parameters (a, b, c) for Low, Med, High over [0,1] with shoulders */
static const double setp[NSET][3] = {
    { -0.5, 0.0, 0.5 },
    {  0.0, 0.5, 1.0 },
    {  0.5, 1.0, 1.5 },
};

int main(void) {
    double zval[NP];
    double os[NSET][NP];   /* precomputed output-set memberships */
    for (int j = 0; j < NP; j++) {
        double z = (double)j / (double)(NP - 1);
        zval[j] = z;
        for (int s = 0; s < NSET; s++)
            os[s][j] = tri(z, setp[s][0], setp[s][1], setp[s][2]);
    }
    int outset[NRULE];     /* rule consequent: "fuzzy average" of the two inputs */
    for (int xi = 0; xi < NSET; xi++)
        for (int yi = 0; yi < NSET; yi++) {
            int sum = xi + yi;
            outset[xi * NSET + yi] = sum <= 1 ? 0 : (sum == 2 ? 1 : 2);
        }

    SimpleRNG rng = { .state = 42 };
    double checksum = 0.0;
    double mu_x[NSET], mu_y[NSET], fs[NRULE];
    for (int q = 0; q < Q; q++) {
        double x = next_double(&rng);
        double y = next_double(&rng);
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
