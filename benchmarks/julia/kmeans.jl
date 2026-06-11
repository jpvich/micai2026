# K-Means clustering (Lloyd's algorithm) — Data Mining benchmark (MICAI).
# Same algorithm as the other five languages; shared 64-bit LCG (seed 42);
# SQUARED Euclidean distance (no sqrt) keeps the inner loop bit-exact.
# Assignment ties break to the lowest centroid index (strict `<`).
# Hot path wrapped in main() — Julia compiles global non-const accesses as
# dynamic dispatch, which silently makes large loops 10-50x slower.
using Printf

const N = 100_000
const D = 4
const K = 10
const ITERS = 1000  # sized so C ~0.8s

mutable struct SimpleRNG
    state::UInt64
end

function next_double!(rng::SimpleRNG)::Float64
    rng.state = rng.state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
    return Float64(rng.state >> 33) / Float64(UInt64(1) << 31)
end

function main()
    rng = SimpleRNG(UInt64(42))
    # column-major: points[d, i] is contiguous per point; LCG consumed in
    # (point i, dim d) order to match the row-major languages exactly.
    points = zeros(Float64, D, N)
    for i in 1:N
        for d in 1:D
            points[d, i] = next_double!(rng)
        end
    end

    centroids = zeros(Float64, D, K)
    for k in 1:K
        for d in 1:D
            centroids[d, k] = points[d, k]
        end
    end

    assign = zeros(Int, N)
    counts = zeros(Int64, K)
    sums = zeros(Float64, D, K)

    for it in 1:ITERS
        for i in 1:N
            best = 1e300
            bestk = 1
            for k in 1:K
                dist = 0.0
                for d in 1:D
                    diff = points[d, i] - centroids[d, k]
                    dist += diff * diff
                end
                if dist < best
                    best = dist
                    bestk = k
                end
            end
            assign[i] = bestk
        end
        for k in 1:K
            counts[k] = 0
            for d in 1:D
                sums[d, k] = 0.0
            end
        end
        for i in 1:N
            k = assign[i]
            counts[k] += 1
            for d in 1:D
                sums[d, k] += points[d, i]
            end
        end
        for k in 1:K
            if counts[k] > 0
                for d in 1:D
                    centroids[d, k] = sums[d, k] / Float64(counts[k])
                end
            end
        end
    end

    fingerprint = Int64(0)
    centroid_sum = 0.0
    for k in 1:K
        fingerprint += counts[k] * Int64(k)  # 1-based k == (k+1) weight in 0-based langs
        for d in 1:D
            centroid_sum += centroids[d, k]
        end
    end
    @printf("%d %.6f\n", fingerprint, centroid_sum)
end

main()
