# Genetic Algorithm minimizing the Rosenbrock function —
# Computational Intelligence benchmark (MICAI). Tournament selection, uniform
# crossover, uniform mutation. Rosenbrock + uniform mutation use only +,-,*,/,
# so the search is bit-exact across languages. Every random decision draws from
# the shared 64-bit LCG (seed 42) in identical order in all six languages.
# Checksum = best fitness found + sum of the best individual's genes (6 dp).
using Printf

const D = 30
const P = 5000
const G = 1200
const T = 3
const MUT_RATE = 0.1
const MUT_STEP = 0.1

mutable struct SimpleRNG
    state::UInt64
end

function next_double!(rng::SimpleRNG)::Float64
    rng.state = rng.state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
    return Float64(rng.state >> 33) / Float64(UInt64(1) << 31)
end

# column-major: pop[d, p] is contiguous per individual.
function rosenbrock(pop::Matrix{Float64}, p::Int)::Float64
    f = 0.0
    for i in 1:(D - 1)
        a = pop[i + 1, p] - pop[i, p] * pop[i, p]
        b = 1.0 - pop[i, p]
        f += 100.0 * a * a + b * b
    end
    return f
end

function main()
    rng = SimpleRNG(UInt64(42))
    pop = zeros(Float64, D, P)
    newpop = zeros(Float64, D, P)
    fitness = zeros(Float64, P)
    best_genes = zeros(Float64, D)

    for p in 1:P
        for d in 1:D
            pop[d, p] = (next_double!(rng) * 2.0 - 1.0) * 5.0
        end
    end

    best_fit = 1e300
    for g in 1:G
        for p in 1:P
            f = rosenbrock(pop, p)
            fitness[p] = f
            if f < best_fit
                best_fit = f
                for d in 1:D
                    best_genes[d] = pop[d, p]
                end
            end
        end
        for d in 1:D
            newpop[d, 1] = best_genes[d]
        end
        for i in 2:P
            a = Int(floor(next_double!(rng) * P)) + 1
            for t in 2:T
                idx = Int(floor(next_double!(rng) * P)) + 1
                if fitness[idx] < fitness[a]
                    a = idx
                end
            end
            b = Int(floor(next_double!(rng) * P)) + 1
            for t in 2:T
                idx = Int(floor(next_double!(rng) * P)) + 1
                if fitness[idx] < fitness[b]
                    b = idx
                end
            end
            for d in 1:D
                newpop[d, i] = next_double!(rng) < 0.5 ? pop[d, a] : pop[d, b]
            end
            for d in 1:D
                if next_double!(rng) < MUT_RATE
                    newpop[d, i] += (next_double!(rng) * 2.0 - 1.0) * MUT_STEP
                end
            end
        end
        for p in 1:P
            for d in 1:D
                pop[d, p] = newpop[d, p]
            end
        end
    end

    gsum = 0.0
    for d in 1:D
        gsum += best_genes[d]
    end
    @printf("%.6f %.6f\n", best_fit, gsum)
end

main()
