#!/usr/bin/env julia

### Note ###
#
# test with: include("ls_const.jl"); main(Dict("random-seed" => nothing, "show-solution" => true, "runtime-limit"=>1.0, "input-file" => "---"))
#

using ArgParse
using JSON

using Random

function main(parsed_args)
    if !isnothing(parsed_args["random-seed"])
        Random.seed!(parsed_args["random-seed"])
    end

    #read the problem
    file = open(parsed_args["input-file"])
    data = JSON.parse(file)

    if data["variable_domain"] != "spin"
        error("only spin domains are supported. Given $(data["variable_domain"])")
    end

    time_limit = parsed_args["runtime-limit"]

    idx_to_var = data["variable_ids"]
    var_to_idx = Dict(var => idx for (idx,var) in enumerate(data["variable_ids"]))
    n = length(idx_to_var)

    variable_linear_term = [0.0 for i in 1:n]
    linear_terms = Tuple{Int64,Float64}[]
    for lt in data["linear_terms"]
        i = var_to_idx[lt["id"]]
        c = lt["coeff"]
        push!(linear_terms, (i, c))
        variable_linear_term[i] = c
    end

    neighbors = [Tuple{Int64,Float64}[] for i in 1:n]
    quadratic_terms = Tuple{Int64,Int64,Float64}[]
    for qt in data["quadratic_terms"]
        i = var_to_idx[qt["id_head"]]
        j = var_to_idx[qt["id_tail"]]
        c = qt["coeff"]
        push!(quadratic_terms, (i, j, c))
        push!(neighbors[i], (j, c))
        push!(neighbors[j], (i, c))
    end

    #println(idx_to_var)
    println("problem size: $(n), $(length(data["quadratic_terms"]))")

    restarts = 0
    iterations = 0
    best_assignment = [0 for i in 1:n]
    best_energy = Inf

    variable_order = [i for i in 1:n]
    assignment = [2*rand(Bool) - 1 for i in 1:n]
    energy = calc_energy(assignment, linear_terms, quadratic_terms)

    # compilatio runs, does not seem to be required
    #calc_energy(assignment, linear_terms, quadratic_terms)
    #evaluate_neighbors(1, variable_linear_term[1], neighbors[1], assignment)

    time_start = time()
    while time() - time_start < time_limit
        Random.shuffle!(variable_order)

        improved = false
        for i in variable_order
            iterations += 1

            if assignment[i] == 1
                assignment[i] = -1
            else
                assignment[i] = 1
            end
            eval_flip = evaluate_neighbors(i, variable_linear_term[i], neighbors[i], assignment)

            if eval_flip < 0
                energy = energy + 2*eval_flip
                improved = true
            else
                if eval_flip > 0 || (isapprox(eval_flip, 0) && rand() < 0.5)
                    if assignment[i] == 1
                        assignment[i] = -1
                    else
                        assignment[i] = 1
                    end
                end
            end

            if time() - time_start > time_limit
                println()
                println("termination at $(i) of $(n)")
                break
            end
        end

        #print(assignment)

        #energy_test = calc_energy(assignment, linear_terms, quadratic_terms)
        #println(energy, " ", energy_test)
        #@assert(isapprox(energy, energy_test))

        if improved
            if energy < best_energy
                best_energy = energy
                for i in 1:n
                    best_assignment[i] = assignment[i]
                end
                #println(best_assignment)
                print("($(best_energy), $(iterations), $(restarts))")
            end
        else
            for i in 1:n
                assignment[i] = 2*rand(Bool) - 1
            end
            energy = calc_energy(assignment, linear_terms, quadratic_terms)

            restarts += 1
            print("R")
        end
    end
    println()
    time_elapsed = time() - time_start

    println("restarts: $restarts")
    println("final energy: $best_energy")

    best_solution = Dict(idx_to_var[i] => best_assignment[i]  for i in 1:n)
    sol_energy = calc_energy(best_solution, data)
    @assert isapprox(best_energy, sol_energy)

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
        best_solution_string = join([string(best_solution[vid]) for vid in data["variable_ids"]], ", ")
        println("BQP_SOLUTION, $(nodes), $(edges), $(scaled_objective), $(best_runtime), $(best_solution_string)")
    end

    println("BQP_DATA, $(nodes), $(edges), $(scaled_objective), $(scaled_lower_bound), $(best_objective), $(lower_bound), $(best_runtime), $(iterations), $(best_nodes)")
end


function evaluate_neighbors(vid, linear_coeff, neighbors, assignment)::Float64
    energy = 0.0
    energy += linear_coeff * assignment[vid]
    for (j,c) in neighbors
        energy += c * assignment[vid] * assignment[j]
    end
    return energy
end


function calc_energy(assignment, linear_terms, quadratic_terms)
    energy = 0.0
    for (i,c) in linear_terms
        energy += c * assignment[i]
    end

    for (i,j,c) in quadratic_terms
        energy += c * assignment[i] * assignment[j]
    end

    return energy
end


"evaluate the enery function given a variable assignment"
function calc_energy(assignment, data)::Float64
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

    return energy
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
    end

    return parse_args(s)
end


if isinteractive() == false
    main(parse_commandline())
end
