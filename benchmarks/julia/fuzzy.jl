# Mamdani fuzzy inference system — Fuzzy Systems benchmark (MICAI).
# Two inputs -> 3 triangular sets each; 9-rule base; max-min aggregation;
# centroid defuzzification over a discretized output domain. Triangular MFs +
# centroid use only +,-,*,/,min,max, so the result is bit-exact across
# languages. Q input pairs drawn from the shared 64-bit LCG (seed 42).
# Checksum = sum of defuzzified outputs (6 dp). Hot path wrapped in main().
using Printf

const Q = 2_000_000
const NP = 100
const NSET = 3
const NRULE = 9

const SETP = [
    -0.5 0.0 0.5;
    0.0 0.5 1.0;
    0.5 1.0 1.5
]  # row s = set s, columns are (a, b, c)

mutable struct SimpleRNG
    state::UInt64
end

function next_double!(rng::SimpleRNG)::Float64
    rng.state = rng.state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
    return Float64(rng.state >> 33) / Float64(UInt64(1) << 31)
end

@inline function tri(v::Float64, a::Float64, b::Float64, c::Float64)::Float64
    left = (v - a) / (b - a)
    right = (c - v) / (c - b)
    m = left < right ? left : right
    return m > 0.0 ? m : 0.0
end

function main()
    zval = zeros(Float64, NP)
    os = zeros(Float64, NSET, NP)   # os[s, j]
    for j in 1:NP
        z = Float64(j - 1) / Float64(NP - 1)
        zval[j] = z
        for s in 1:NSET
            os[s, j] = tri(z, SETP[s, 1], SETP[s, 2], SETP[s, 3])
        end
    end
    outset = zeros(Int, NRULE)      # 0-based set index stored; +1 when indexing os
    for xi in 0:(NSET - 1)
        for yi in 0:(NSET - 1)
            sm = xi + yi
            outset[xi * NSET + yi + 1] = sm <= 1 ? 0 : (sm == 2 ? 1 : 2)
        end
    end

    rng = SimpleRNG(UInt64(42))
    checksum = 0.0
    mu_x = zeros(Float64, NSET)
    mu_y = zeros(Float64, NSET)
    fs = zeros(Float64, NRULE)
    for q in 1:Q
        x = next_double!(rng)
        y = next_double!(rng)
        for s in 1:NSET
            mu_x[s] = tri(x, SETP[s, 1], SETP[s, 2], SETP[s, 3])
            mu_y[s] = tri(y, SETP[s, 1], SETP[s, 2], SETP[s, 3])
        end
        for xi in 0:(NSET - 1)
            for yi in 0:(NSET - 1)
                f = mu_x[xi + 1] < mu_y[yi + 1] ? mu_x[xi + 1] : mu_y[yi + 1]
                fs[xi * NSET + yi + 1] = f
            end
        end
        num = 0.0
        den = 0.0
        for j in 1:NP
            agg = 0.0
            for r in 1:NRULE
                osv = os[outset[r] + 1, j]
                m = fs[r] < osv ? fs[r] : osv
                if m > agg
                    agg = m
                end
            end
            num += zval[j] * agg
            den += agg
        end
        out = den > 0.0 ? num / den : 0.0
        checksum += out
    end
    @printf("%.6f\n", checksum)
end

main()
