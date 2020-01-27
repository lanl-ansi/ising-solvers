#!/usr/bin/env julia

### Note ###
#
# code from the blog post at, https://invenia.github.io/blog/2019/09/11/memristors/
#
# tested on Julia v1.3
# test with: include("ls_mem.jl"); main(Dict("random-seed" => nothing, "max-steps"=>100, "show-solution" => true, "runtime-limit"=>1.0, "input-file" => "---"))
#

using ArgParse
using JSON

using Random
using LinearAlgebra
using DifferentialEquations

function main(parsed_args)
    if !isnothing(parsed_args["random-seed"])
        Random.seed!(parsed_args["random-seed"])
    end

    #read the problem
    file = open(parsed_args["input-file"])
    data = JSON.parse(file)

    if data["variable_domain"] != "spin"
        error("only boolean domains are supported. Given $(data["variable_domain"])")
    end

    data = spin_to_bool(data)

    idx_to_var = data["variable_ids"]
    var_to_idx = Dict(var => idx for (idx,var) in enumerate(data["variable_ids"]))
    n = length(idx_to_var)

    #println(idx_to_var)
    println("problem size: $(n), $(length(data["quadratic_terms"]))")

    Q = zeros(n, n);
    for qt in data["quadratic_terms"]
        i = var_to_idx[qt["id_head"]]
        j = var_to_idx[qt["id_tail"]]
        c = qt["coeff"]
        Q[i, j] = c/2
        Q[j, i] = c/2
    end

    h = [0.0 for i in 1:n]
    for lt in data["linear_terms"]
        i = var_to_idx[lt["id"]]
        c = lt["coeff"]

        h[i] = c
    end

    #optional regularisation term
    for i in 1:n
        Q[i,i] = 1e-4
    end

    # complication run
    #memristive_opt(h, Q, 0.1,  [0.5 for i in 1:n], total_time=1)
    memristive_opt_2(h, Q, [0.5 for i in 1:n], total_time=1)

    #=
    v = [0 for i in 1:n]
    check_encoding(h, Q, v, data)

    v = [1 for i in 1:n]
    check_encoding(h, Q, v, data)

    v = [mod(i,2) for i in 1:n]
    check_encoding(h, Q, v, data)
    =#

    # a tunable parameter
    p = 0.1

    restarts = 0
    best_assignment = Dict(idx_to_var[i] => 0.0 for i in 1:n)
    best_energy = calc_energy(data, best_assignment)
    print("($(best_energy))")

    time_start = time()
    while time() - time_start < parsed_args["runtime-limit"]
        v = [Random.rand() for i in 1:n]

        #weights_final, energies = memristive_opt(h, Q, p, v, total_time=parsed_args["max-steps"])
        weights_final, energies = memristive_opt_2(h, Q, v, total_time=parsed_args["max-steps"])

        #println("energies trace: $energies")
        #println("weights: $weights_final")

        assignment = Dict(idx_to_var[i] => weights_final[i] for i in 1:n)
        energy = calc_energy(data, assignment)

        if energy < best_energy
            best_energy = energy
            best_assignment = assignment
            print("($(best_energy))")
        end

        restarts += 1
        print("R")
    end
    println()

    time_elapsed = time() - time_start

    println("restarts: $restarts")
    println("final energy eval: $best_energy")

    nodes = length(data["variable_ids"])
    edges = length(data["quadratic_terms"])

    scale = data["scale"]
    offset = data["offset"]
    lt_lb = -sum(abs(lt["coeff"]) for lt in data["linear_terms"])/scale
    qt_lb = -sum(abs(qt["coeff"]) for qt in data["quadratic_terms"])/scale
    lower_bound = lt_lb+qt_lb

    best_objective = best_energy
    best_nodes = restarts
    best_runtime = time_elapsed
    scaled_objective = scale*(best_objective+offset)
    scaled_lower_bound = scale*(lower_bound+offset)

    if parsed_args["show-solution"]
        best_solution_string = join([(best_assignment[vid] < 0.5 ? "-1" : "1") for vid in data["variable_ids"]], ", ")
        println("BQP_SOLUTION, $(nodes), $(edges), $(scaled_objective), $(best_runtime), $(best_solution_string)")
    end

    println("BQP_DATA, $(nodes), $(edges), $(scaled_objective), $(scaled_lower_bound), $(best_objective), $(lower_bound), $(best_runtime), $(0), $(best_nodes)")
end


"""
    memristive_opt(
        expected_returns::Vector{Float64},
        Σ::Matrix{Float64},
        p::Float64,
        weights_init::Vector{<:Real};
        α=0.1,
        β=10,
        δt=0.1,
        total_time=3000,
    )

Execute optimisation via the heuristic "memristive" equation in order to find the
optimal portfolio composition, considering an asset covariance matrix `Σ` and a
risk parameter `p`. `reg` represents the regularisation constant for `Σ`, `α` and
`β` parametrise the memristor state (see Equation (2)),
`δt` is the size of the time step for the dynamical updates and `total_time` is
the number of time steps for which the dynamics will be run.
"""
function memristive_opt(
    expected_returns::Vector{Float64},
    Σ::Matrix{Float64},
    p::Float64,
    weights_init::Vector{<:Real};
    α=0.1,
    β=10,
    δt=0.1,
    total_time=3000,
)
    n = size(weights_init, 1)

    weights_series = Matrix{Float64}(undef, n, total_time)
    weights_series[:, 1] = weights_init
    energies = Vector{Float64}(undef, total_time)
    energies[1] = energy(expected_returns, Σ, weights_series[:, 1], p=p)

    # Compute resistance change ratio
    ξ = p / 2α

    #println(p, " ", α, " ", β, " ", ξ)
    #display(Σ)
    #S = β * inv(Σ) * (α/2 * ones(n, 1) + (p/2 + α * ξ/3) * diag(Σ) - expected_returns)
    #println(S)

    # Compute Σ times applied voltages matrix
    ΣS = β * (α/2 * ones(n, 1) + (p/2 + α * ξ/3) * diag(Σ) - expected_returns)

    #println(inv(Σ)*ΣS)
    prev_update = zeros(n)
    t = 1
    while t < total_time
    #for t in 1:total_time-1
        update = δt * (α * weights_series[:, t] - 1/β * (I(n) + ξ *
            Σ * Diagonal(weights_series[:, t])) \ ΣS)

        #println(norm(update .- prev_update))
        if norm(update .- prev_update) <= 1e-6
            break # solver converged
        else
            #println(norm(update .- prev_update), " ", update)
        end
        prev_update = update

        weights_series[:, t+1] = weights_series[:, t] + update

        weights_series[weights_series[:, t+1] .> 1, t+1] .= 1.0
        weights_series[weights_series[:, t+1] .< 1, t+1] .= 0.0

        energies[t + 1] = energy(expected_returns, Σ, weights_series[:, t+1], p=p)
        #println(weights_series[:, t+1])
        #println(energies[t + 1])
        t += 1
    end

    #println(t)
    weights_final = weights_series[:, t]

    #println(weights_final)
    #println(energies)

    return weights_final, energies
end

"""
    memristive_opt_2(
        h::Vector{Float64},
        Q::Matrix{Float64},
        weights_init::Vector{<:Real};
        α=0.1,
        β=10,
        total_time=3000,
    )

Heuristic optimization via a memristive network.  Changes to the above include
a windowed memristive dynamics equation, and a changed mapping that ensures
Ω is PSD.
"""
function memristive_opt_2(
    h::Vector{Float64},
    Q::Matrix{Float64},
    weights_init::Vector{<:Real};
    α=0.2,
    β=0.2,
    ξ=1.0,
    total_time=3000.,
)

    n = length(h)

    v = [Random.rand() for i in 1:n]

    v = reshape(v, n, 1)

    # I don't understand at the moment why this minus sign is required
    Ω, Ωs = convert_QUBO_to_MEMNET(-Q, h, α, β, ξ)

    dwdt = caravelli_eqn_windowed(Ω, Ωs, α, β, ξ)

    function condition(u, t, integrator)
        w = integrator.u
        maximum(w .* (1.0 .- w)) < 1e-3
    end

    function affect!(integrator)
        terminate!(integrator)
    end

    cb = DiscreteCallback(condition, affect!, save_positions=(true, true))

    tspan = (0.0, total_time)
    prob = ODEProblem(dwdt, v, tspan)

    sol = solve(prob, reltol=1e-6, callback=cb)

    t = sol.t
    w_traj = sol.u

    tsteps = length(t)
    energies = zeros(tsteps)

    for idx in 1:tsteps
        energies[idx] = energy(h, Q, vec(w_traj[idx]))
    end

    return w_traj[tsteps], energies
end


function convert_QUBO_to_MEMNET(Q, h, α, ξ, β)

    n = length(h)
    # Q should have zero diagonal
    Q[diagind(Q)] .= 0
    Ω = -Q/(α*ξ)
    mineval = minimum(eigvals(Ω))

    Ω[diagind(Ω)] .= -mineval
    Ωs = β * (α/2 * ones(n, 1) + (α * ξ/3) * diag(Ω) + h)

    return Ω, Ωs
end


"""
    caravelli_eqn_windowed(
        Ω::Matrix{Float64},
        Ωs::Matrix{Float64},
        α::Float64,
        β::Float64,
        ξ
    )

A windowed version of the equations of motion for a memristive network.
Ω is the network projection matrix
Ωs is the source vector which is assumed to live in the space projected
by Ω
α is the memristive decay constant
β is the growth timescale
ξ is (R_off - R_on)/R_on
"""
function caravelli_eqn_windowed(
    Ω::Matrix{Float64},
    Ωs::Matrix{Float64},
    α::Float64,
    β::Float64,
    ξ::Float64
    )
    n = length(Ωs)

    function dwdt(w, p, t)
        return w .* (1 .- w).*(α * w - 1/β * (I(n) + ξ*Ω*Diagonal(w)) \ Ωs)
    end

    return dwdt
end



function energy(
    h::Vector{Float64},
    Q::Matrix{Float64},
    weights::Vector{<:Real};
    p=2.0
)
    dot(h, weights) +  p/2 * weights' * Q * weights
end


"checks that qubo evalution matches energy function evaluation"
function check_encoding(h, Q, assignment, data)
    eval_q = assignment' * Q * assignment - dot(h, assignment)
    eval_energy = energy(h, Q, assignment)
    #println(eval_q, " ", eval_energy)

    #eval_data = calc_energy(data, Dict(data["variable_ids"][i] => v for (i,v) in enumerate(assignment)))
    #println(eval_q, " ", eval_energy, " ", eval_data)
    @assert(isapprox(eval_q, eval_energy))
    #@assert(isapprox(eval_energy, eval_data))
end


"evaluate the enery function given a variable assignment"
function calc_energy(data, assignment)::Float64
    energy = 0.0
    for qt in data["quadratic_terms"]
        i = qt["id_head"]
        j = qt["id_tail"]
        c = qt["coeff"]
        energy += c * assignment[i] * assignment[j]
    end

    for lt in data["linear_terms"]
        i = lt["id"]
        c = lt["coeff"]
        energy += c * assignment[i]
    end

    #return data["scale"]*(energy+data["offset"])
    return energy
end


function spin_to_bool(ising_data)
    @assert ising_data["variable_domain"] == "spin"

    offset = ising_data["offset"]
    coefficients = Dict()

    for v_id in ising_data["variable_ids"]
        coefficients[(v_id, v_id)] = 0.0
    end

    for linear_term in ising_data["linear_terms"]
        v_id = linear_term["id"]
        coeff = linear_term["coeff"]
        #assert(coeff != 0.0)

        coefficients[(v_id, v_id)] = 2.0*coeff
        offset += -coeff
    end

    for quadratic_term in ising_data["quadratic_terms"]
        v_id1 = quadratic_term["id_tail"]
        v_id2 = quadratic_term["id_head"]
        coeff = quadratic_term["coeff"]
        #assert(coeff != 0.0)

        if !haskey(coefficients, (v_id1, v_id2))
            coefficients[(v_id1, v_id2)] = 0.0
        end

        coefficients[(v_id1, v_id2)] = coefficients[(v_id1, v_id2)] + 4.0*coeff
        coefficients[(v_id1, v_id1)] = coefficients[(v_id1, v_id1)] - 2.0*coeff
        coefficients[(v_id2, v_id2)] = coefficients[(v_id2, v_id2)] - 2.0*coeff
        offset += coeff
    end

    linear_terms = []
    quadratic_terms = []

    for (i,j) in sort(collect(keys(coefficients)))
        v = coefficients[(i,j)]
        if v != 0.0
            if i == j
                push!(linear_terms, Dict("id" => i, "coeff" =>v))
            else
                push!(quadratic_terms, Dict("id_tail" => i, "id_head" => j, "coeff" => v))
            end
        end
    end

    bool_data = deepcopy(ising_data)
    bool_data["variable_domain"] = "boolean"
    bool_data["offset"] = offset
    bool_data["linear_terms"] = linear_terms
    bool_data["quadratic_terms"] = quadratic_terms

    if haskey(bool_data, "solutions")
        for solution in bool_data["solutions"]
            for assign in solution["assignment"]
                if assign["value"] == -1
                    assign["value"] = 0
                end
            end
        end
    end

    return bool_data
end


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--input-file", "-f"
            help = "the data file to operate on (.json)"
            required = true
        "--runtime-limit", "-t"
            help = "puts a time limit on the solver"
            arg_type = Float64
            default = 10.0
        "--show-solution", "-s"
            help = "print the solution"
            action = :store_true
        "--random-seed"
            help = "fixes the random number generator seed"
            arg_type = Int
        "--max-steps"
            help = "set the maximum number of steps in the memristor evolution"
            arg_type = Int
            default = 100
    end

    return parse_args(s)
end

if isinteractive() == false
    main(parse_commandline())
end
