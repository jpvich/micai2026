# k-NN classification — Machine Learning benchmark (MICAI).
# Same algorithm as the other five languages; shared 64-bit LCG (seed 42);
# SQUARED Euclidean distance (no sqrt) for bit-exactness; vote ties break to
# the lowest class label. Checksum = sum of predicted labels (integer).
# Hot path wrapped in main() so Julia type-specializes the loops.

const M = 50_000
const Q = 10_000
const D = 8
const K = 15
const C = 3

mutable struct SimpleRNG
    state::UInt64
end

function next_double!(rng::SimpleRNG)::Float64
    rng.state = rng.state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
    return Float64(rng.state >> 33) / Float64(UInt64(1) << 31)
end

function main()
    rng = SimpleRNG(UInt64(42))
    train = zeros(Float64, D, M)   # column-major: contiguous per training point
    label = zeros(Int, M)
    query = zeros(Float64, D, Q)
    for t in 1:M
        for d in 1:D
            train[d, t] = next_double!(rng)
        end
        label[t] = Int(floor(next_double!(rng) * C))  # 0..C-1
    end
    for q in 1:Q
        for d in 1:D
            query[d, q] = next_double!(rng)
        end
    end

    checksum = Int64(0)
    best_d = fill(1e300, K)
    best_l = zeros(Int, K)
    for q in 1:Q
        for j in 1:K
            best_d[j] = 1e300
            best_l[j] = 0
        end
        for t in 1:M
            dist = 0.0
            for d in 1:D
                diff = query[d, q] - train[d, t]
                dist += diff * diff
            end
            if dist < best_d[K]
                p = K
                while p > 1 && dist < best_d[p - 1]
                    best_d[p] = best_d[p - 1]
                    best_l[p] = best_l[p - 1]
                    p -= 1
                end
                best_d[p] = dist
                best_l[p] = label[t]
            end
        end
        votes = zeros(Int, C)   # index c+1 holds class c
        for j in 1:K
            votes[best_l[j] + 1] += 1
        end
        pred = 0
        for c in 1:(C - 1)
            if votes[c + 1] > votes[pred + 1]
                pred = c
            end
        end
        checksum += pred
    end
    println(checksum)
end

main()
