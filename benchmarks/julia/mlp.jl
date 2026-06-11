# MLP training (forward + backprop, full-batch gradient descent) —
# Neural Network benchmark (MICAI). D->H->O, ReLU hidden, linear output, MSE.
# ReLU + MSE use only +,-,*,/ and max (no exp/softmax), so the result is
# bit-exact across languages. Shared 64-bit LCG (seed 42) for identical init.
# Flat 1-D arrays with row-major indexing mirror the C layout exactly (same
# access pattern + same accumulation order). Hot path wrapped in main().
using Printf

const N = 10_000
const D = 16
const H = 64
const O = 4
const E = 150
const LR = 0.01

mutable struct SimpleRNG
    state::UInt64
end

function next_double!(rng::SimpleRNG)::Float64
    rng.state = rng.state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
    return Float64(rng.state >> 33) / Float64(UInt64(1) << 31)
end

function main()
    rng = SimpleRNG(UInt64(42))
    W1 = zeros(Float64, H * D)   # row-major: W1[(h-1)*D + d]
    b1 = zeros(Float64, H)
    W2 = zeros(Float64, O * H)   # W2[(o-1)*H + h]
    b2 = zeros(Float64, O)
    for h in 1:H
        for d in 1:D
            W1[(h - 1) * D + d] = (next_double!(rng) * 2.0 - 1.0) * 0.1
        end
    end
    for o in 1:O
        for h in 1:H
            W2[(o - 1) * H + h] = (next_double!(rng) * 2.0 - 1.0) * 0.1
        end
    end
    x = zeros(Float64, N * D)        # x[(n-1)*D + d]
    target = zeros(Float64, N * O)   # target[(n-1)*O + o]
    for n in 1:N
        for d in 1:D
            x[(n - 1) * D + d] = next_double!(rng)
        end
        for o in 1:O
            target[(n - 1) * O + o] = next_double!(rng)
        end
    end

    gW1 = zeros(Float64, H * D)
    gb1 = zeros(Float64, H)
    gW2 = zeros(Float64, O * H)
    gb2 = zeros(Float64, O)
    z1 = zeros(Float64, H)
    a1 = zeros(Float64, H)
    y = zeros(Float64, O)
    dy = zeros(Float64, O)

    scale = LR / Float64(N)
    final_loss = 0.0
    for e in 1:E
        fill!(gW1, 0.0); fill!(gb1, 0.0); fill!(gW2, 0.0); fill!(gb2, 0.0)
        epoch_loss = 0.0
        for n in 1:N
            nbaseD = (n - 1) * D
            for h in 1:H
                s = b1[h]
                wbase = (h - 1) * D
                for d in 1:D
                    s += W1[wbase + d] * x[nbaseD + d]
                end
                z1[h] = s
                a1[h] = s > 0.0 ? s : 0.0
            end
            for o in 1:O
                s = b2[o]
                wbase = (o - 1) * H
                for h in 1:H
                    s += W2[wbase + h] * a1[h]
                end
                y[o] = s
            end
            nbaseO = (n - 1) * O
            for o in 1:O
                diff = y[o] - target[nbaseO + o]
                epoch_loss += diff * diff
                dy[o] = 2.0 * diff
            end
            for o in 1:O
                gb2[o] += dy[o]
                wbase = (o - 1) * H
                for h in 1:H
                    gW2[wbase + h] += dy[o] * a1[h]
                end
            end
            for h in 1:H
                da = 0.0
                for o in 1:O
                    da += W2[(o - 1) * H + h] * dy[o]
                end
                dz = z1[h] > 0.0 ? da : 0.0
                gb1[h] += dz
                wbase = (h - 1) * D
                for d in 1:D
                    gW1[wbase + d] += dz * x[nbaseD + d]
                end
            end
        end
        for h in 1:H
            b1[h] -= scale * gb1[h]
            wbase = (h - 1) * D
            for d in 1:D
                W1[wbase + d] -= scale * gW1[wbase + d]
            end
        end
        for o in 1:O
            b2[o] -= scale * gb2[o]
            wbase = (o - 1) * H
            for h in 1:H
                W2[wbase + h] -= scale * gW2[wbase + h]
            end
        end
        final_loss = epoch_loss / Float64(N * O)
    end

    wsum = 0.0
    for h in 1:H
        wsum += b1[h]
        wbase = (h - 1) * D
        for d in 1:D
            wsum += W1[wbase + d]
        end
    end
    for o in 1:O
        wsum += b2[o]
        wbase = (o - 1) * H
        for h in 1:H
            wsum += W2[wbase + h]
        end
    end
    @printf("%.6f %.6f\n", final_loss, wsum)
end

main()
