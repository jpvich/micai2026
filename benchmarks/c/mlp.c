/* MLP training (forward + backprop, full-batch gradient descent) —
 * Neural Network benchmark (MICAI). Architecture D->H->O with ReLU hidden
 * activation, linear output, and mean-squared-error loss. ReLU + MSE use only
 * +,-,*,/ and max, so there are no transcendental functions and the result is
 * bit-exact across IEEE-754 implementations (cf. softmax/sigmoid which would
 * call divergent per-language exp()). Shared 64-bit LCG (seed 42) initializes
 * identical weights and data everywhere.
 * Checksum = final-epoch loss + sum of all weights (both 6 dp). */
#include <stdio.h>

#define N 10000   /* samples */
#define D 16      /* input features */
#define H 64      /* hidden units */
#define O 4       /* outputs */
#define E 150     /* epochs */
#define LR 0.01   /* learning rate */

typedef struct { unsigned long long state; } SimpleRNG;

static double next_double(SimpleRNG *rng) {
    rng->state = rng->state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(rng->state >> 33) / (double)(1ULL << 31);
}

static double x[N][D], target[N][O];
static double W1[H][D], b1[H], W2[O][H], b2[O];
static double gW1[H][D], gb1[H], gW2[O][H], gb2[O];
static double z1[H], a1[H], y[O], dy[O];

int main(void) {
    SimpleRNG rng = { .state = 42 };
    for (int h = 0; h < H; h++) {
        for (int d = 0; d < D; d++) W1[h][d] = (next_double(&rng) * 2.0 - 1.0) * 0.1;
        b1[h] = 0.0;
    }
    for (int o = 0; o < O; o++) {
        for (int h = 0; h < H; h++) W2[o][h] = (next_double(&rng) * 2.0 - 1.0) * 0.1;
        b2[o] = 0.0;
    }
    for (int n = 0; n < N; n++) {
        for (int d = 0; d < D; d++) x[n][d] = next_double(&rng);
        for (int o = 0; o < O; o++) target[n][o] = next_double(&rng);
    }

    double scale = LR / (double)N;
    double final_loss = 0.0;
    for (int e = 0; e < E; e++) {
        for (int h = 0; h < H; h++) { gb1[h] = 0.0; for (int d = 0; d < D; d++) gW1[h][d] = 0.0; }
        for (int o = 0; o < O; o++) { gb2[o] = 0.0; for (int h = 0; h < H; h++) gW2[o][h] = 0.0; }
        double epoch_loss = 0.0;
        for (int n = 0; n < N; n++) {
            for (int h = 0; h < H; h++) {
                double s = b1[h];
                for (int d = 0; d < D; d++) s += W1[h][d] * x[n][d];
                z1[h] = s;
                a1[h] = s > 0.0 ? s : 0.0;
            }
            for (int o = 0; o < O; o++) {
                double s = b2[o];
                for (int h = 0; h < H; h++) s += W2[o][h] * a1[h];
                y[o] = s;
            }
            for (int o = 0; o < O; o++) {
                double diff = y[o] - target[n][o];
                epoch_loss += diff * diff;
                dy[o] = 2.0 * diff;
            }
            for (int o = 0; o < O; o++) {
                gb2[o] += dy[o];
                for (int h = 0; h < H; h++) gW2[o][h] += dy[o] * a1[h];
            }
            for (int h = 0; h < H; h++) {
                double da = 0.0;
                for (int o = 0; o < O; o++) da += W2[o][h] * dy[o];
                double dz = z1[h] > 0.0 ? da : 0.0;
                gb1[h] += dz;
                for (int d = 0; d < D; d++) gW1[h][d] += dz * x[n][d];
            }
        }
        for (int h = 0; h < H; h++) {
            b1[h] -= scale * gb1[h];
            for (int d = 0; d < D; d++) W1[h][d] -= scale * gW1[h][d];
        }
        for (int o = 0; o < O; o++) {
            b2[o] -= scale * gb2[o];
            for (int h = 0; h < H; h++) W2[o][h] -= scale * gW2[o][h];
        }
        final_loss = epoch_loss / (double)(N * O);
    }

    double wsum = 0.0;
    for (int h = 0; h < H; h++) { wsum += b1[h]; for (int d = 0; d < D; d++) wsum += W1[h][d]; }
    for (int o = 0; o < O; o++) { wsum += b2[o]; for (int h = 0; h < H; h++) wsum += W2[o][h]; }
    printf("%.6f %.6f\n", final_loss, wsum);
    return 0;
}
