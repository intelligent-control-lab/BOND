
function get_bounds_convdual_original(nnet::Network, input::Vector{Float64}, ϵ::Float64)
    layers  = nnet.layers

    l = Vector{Vector{Float64}}() # Lower bound
    u = Vector{Vector{Float64}}() # Upper bound
    b = Vector{Vector{Float64}}() # bias
    μ = Vector{Vector{Vector{Float64}}}() # Dual variables
    input_ReLU = Vector{Vector{Float64}}()

    v1 = layers[1].weights'
    push!(b, layers[1].bias)
    # Bounds for the first layer
    l1, u1 = input_layer_bounds(layers[1], input, ϵ)
    push!(l, l1)
    push!(u, u1)

    for i in 2:length(layers)
        n_input  = length(layers[i-1].bias)
        n_output = length(layers[i].bias)


        last_input_ReLU = NeuralVerification.relaxed_relu_gradient.(last(l), last(u))
        push!(input_ReLU, last_input_ReLU)
        D = Diagonal(last_input_ReLU)   # a matrix whose diagonal values are the relaxed_ReLU values (maybe should be sparse?)

        # Propagate existing terms by right multiplication of D*W' or left multiplication of W*D
        WD = layers[i].weights*D
        v1 = v1 * WD' # propagate V_1^{i-1} to V_1^{i}
        map!(g -> WD*g,   b, b) # propagate bias
        for V in μ
            map!(m -> WD*m,   V, V) # Updating ν_j for all previous layers
        end

        # New terms
        push!(b, layers[i].bias)
        push!(μ, new_μ(n_input, n_output, last_input_ReLU, WD))

        # Compute bounds
        ψ = v1' * input + sum(b)
        eps_v1_sum = ϵ * vec(sum(abs, v1, dims = 1))
        neg, pos = residual(input_ReLU, l, μ, n_output)
        push!(l,  ψ - eps_v1_sum - neg )
        push!(u,  ψ + eps_v1_sum - pos )
    end

    return l, u
end

# This step is similar to reachability method
function get_bounds_convdual(nnet::Network, hyperrectangle_input::Hyperrectangle)

    new_matrix, new_vector, input_uniform = normalize_rectangle_input(hyperrectangle_input)
    layers  = nnet.layers

    l = Vector{Vector{Float64}}() # Lower bound
    u = Vector{Vector{Float64}}() # Upper bound
    b = Vector{Vector{Float64}}() # bias
    μ = Vector{Vector{Vector{Float64}}}() # Dual variables
    input_ReLU = Vector{Vector{Float64}}()

    v1 = layers[1].weights'
    push!(b, layers[1].bias)
    l1, u1 = IA_bounds(layers[1], hyperrectangle_input)
    push!(l, l1)
    push!(u, u1)

    for i in 2:length(layers)
        n_input  = length(layers[i-1].bias)
        n_output = length(layers[i].bias)


        last_input_ReLU = NeuralVerification.relaxed_relu_gradient.(last(l), last(u))
        push!(input_ReLU, last_input_ReLU)
        D = Diagonal(last_input_ReLU)   # a matrix whose diagonal values are the relaxed_ReLU values (maybe should be sparse?)

        # Propagate existing terms by right multiplication of D*W' or left multiplication of W*D
        WD = layers[i].weights*D
        v1 = v1 * WD' # propagate V_1^{i-1} to V_1^{i}
        map!(g -> WD*g,   b, b) # propagate bias
        for V in μ
            map!(m -> WD*m,   V, V) # Updating ν_j for all previous layers
        end
        
        # New terms
        push!(b, layers[i].bias)
        push!(μ, new_μ(n_input, n_output, last_input_ReLU, WD))

        # Compute bounds
        ψ = v1' * new_matrix * input_uniform.center + v1' * new_vector + sum(b)
        eps_v1_sum = input_uniform.radius[1] * vec(sum(abs,  new_matrix' * v1, dims = 1))# TODO check bug? 
        neg, pos = residual(input_ReLU, l, μ, n_output)
        push!(l,  ψ - eps_v1_sum - neg )
        push!(u,  ψ + eps_v1_sum - pos )
    end

    return l, u
end

function IA_bounds(layer, input) 
    bounds = NeuralVerification.approximate_affine_map(layer, overapproximate(input))
    return low(bounds), high(bounds)
end




function input_layer_bounds(input_layer, input, ϵ)
    W, b = input_layer.weights, input_layer.bias

    out1 = vec(W * input + b)
    Δ    = ϵ * vec(sum(abs, W, dims = 2))

    l = out1 - Δ
    u = out1 + Δ
    return l, u
end
function residual(slopes, l, μ, n_output)
    neg = zeros(n_output)
    pos = zeros(n_output)
    for (i, ℓ) in enumerate(l)                # ℓ::Vector{Float64}
        for (j, V) in enumerate(μ[i])         # M::Vector{Float64}
            if 0 < slopes[i][j] < 1              # if in the triangle region of relaxed ReLU
                #posind = M .> 0
                neg .+= ℓ[j] * min.(V, 0) #-M .* !posind  # multiply by boolean to set the undesired values to 0.0
                pos .+= ℓ[j] * max.(V, 0) #M .* posind
            end
        end
    end
    return neg, pos
end



function new_μ(n_input, n_output, input_ReLU, WD)
    sub_μ = Vector{Vector{Float64}}(undef, n_input)
    for j in 1:n_input
        if 0 < input_ReLU[j] < 1 # negative region 
            sub_μ[j] = WD[:, j] 
        else
            sub_μ[j] = zeros(n_output)
        end
    end
    return sub_μ
end


function normalize_rectangle_input(input::Hyperrectangle)
    input_uniform = Hyperrectangle(copy(input.center),copy(input.radius))
    fill!(input_uniform.radius, input_uniform.radius[1])
    new_matrix = zeros((size(input_uniform.center)[1], size(input_uniform.center)[1]))
    new_vector = zeros(size(input_uniform.center)[1])
    for i in 1:size(input_uniform.center)[1]
        new_matrix[i,i] = input.radius[i] / input_uniform.radius[i]
        new_vector[i] = (1 - new_matrix[i,i]) * input.center[i]
    end
    return new_matrix, new_vector, input_uniform
end

function normalize_rectangle_input(net, input::Hyperrectangle)
    input_uniform = Hyperrectangle(copy(input.center),copy(input.radius))
    fill!(input_uniform.radius, input_uniform.radius[1])

    network_with_uniform = net
    new_matrix = zeros((size(input_uniform.center)[1], size(input_uniform.center)[1]))
    new_vector = zeros(size(input_uniform.center)[1])
    for i in 1:size(input_uniform.center)[1]
        new_matrix[i,i] = input.radius[i] / input_uniform.radius[i]
        new_vector[i] = (1 - new_matrix[i,i]) * input.center[i]
    end
    new_layer = NeuralVerification.Layer(new_matrix, new_vector, Id())
    pushfirst!(network_with_uniform.layers, new_layer)
    return network_with_uniform, input_uniform
end

function _ẑᵢ₊₁_bound(m, i)
    layer = m[:network].layers[i]
    # let's assume we're post-activation if the parameter isn't set,
    # since that's the default for get_bounds
    if get(object_dictionary(m), :before_act, false)
        ẑ_bound = m[:bounds][i+1]
    else
        ẑ_bound = NeuralVerification.approximate_affine_map(layer, m[:bounds][i])
    end
    low(ẑ_bound), high(ẑ_bound)
end

function solve_BPO_worstcase(solver::NNDynTrack, problem::TrackingProblem, start_values=nothing; u_ref = nothing, xu_init = nothing, IA_bounds=false, BPO_degree=1)
    # split input rec into branches, solve for last z (exist last z, not in safe output set), return hold once exists some first branch, forall last z holds
    num_branch = [10,10]
    lows = low(problem.input)
    highs = high(problem.input)
    @assert length(lows) == 6
    u_range = highs[5:6] - lows[5:6]
    shuffled = shuffle(Vector(1:num_branch[1]*num_branch[2]))
    for index in shuffled
        i = floor(Int, index / num_branch[2]) + 1
        j = index % num_branch[2] == 0 ? num_branch[2] : index % num_branch[2]

            
        sub_input = Hyperrectangle(low=[lows[1:4]; lows[5:6]+ [(i-1) * u_range[1] / num_branch[1], (j-1) * u_range[2] / num_branch[2]]], high=[highs[1:4]; lows[5:6]+[i * u_range[1] / num_branch[1], j * u_range[2] / num_branch[2]]])

        model = Model(solver)
        set_silent(model)

        set_optimizer_attribute(model, "CPX_PARAM_EPOPT", 1e-8)
        set_optimizer_attribute(model, "CPX_PARAM_EPAGAP", 1e-6)

        if IA_bounds
            # get pre-activationlooser bounds from IBP/IA
            model[:bounds] = bounds = NeuralVerification.get_bounds(problem.network, sub_input, false)
        else 
            # using get_bounds in FastLin and convDual
            L, U  = get_bounds_convdual(problem.network, sub_input)

            pushfirst!(L, low(sub_input))
            pushfirst!(U, high(sub_input))
            bounds = Vector{Hyperrectangle}(undef,size(L)[1])
            for (i, _) in enumerate(L)
                bounds[i] = Hyperrectangle(low=L[i], high=U[i])
            end
            model[:bounds] = bounds
        end
        model[:before_act] = true

        z = NeuralVerification.init_vars(model, problem.network, :z, with_input=true)

        NeuralVerification.add_set_constraint!(model, sub_input, first(z))
        NeuralVerification.add_complementary_set_constraint!(model, problem.output, last(z))
        if BPO_degree == 1
            NeuralVerification.encode_network!(model, problem.network, NeuralVerification.TriangularRelaxedLP())
        else
            NeuralVerification.encode_network!(model, problem.network, NeuralVerification.BernsteinPolynomial2LP())
        end

        d = (last(z) - problem.output_ref)
        o = isnothing(u_ref) ? dot(d, d.*problem.output_cost) : dot(first(z)[5:6] - u_ref, first(z)[5:6] - u_ref)

        @objective(model, Min, o)

        isnothing(start_values) || set_start_value.(all_variables(model), start_values)

        if !isnothing(xu_init)
            set_start_value.(first(z), xu_init)
        end

        optimize!(model)

        if termination_status(model) == OPTIMAL
            continue
        end
        return NeuralVerification.TrackingResult(:holds), nothing
        
    end
    return NeuralVerification.TrackingResult(:violated), nothing
end

function solve_BPO_worstcase_new(solver::NNDynTrack, problem::TrackingProblem, start_values=nothing; u_ref = nothing, xu_init = nothing, IA_bounds=false, BPO_degree=1)
    i = 0
    infeasible_input_vec = []#[EmptySet(length(low(problem.input)))]
    while i < 5
        model = Model(solver)
        set_silent(model)
    
        set_optimizer_attribute(model, "CPX_PARAM_EPOPT", 1e-8)
        set_optimizer_attribute(model, "CPX_PARAM_EPAGAP", 1e-6)
    
        if IA_bounds
            # get pre-activationlooser bounds from IBP/IA
            model[:bounds] = bounds = NeuralVerification.get_bounds(problem.network, problem.input, false)
        else 
            # using get_bounds in FastLin and convDual
            L, U  = get_bounds_convdual(problem.network, problem.input)
            pushfirst!(L, low(problem.input))
            pushfirst!(U, high(problem.input))
            bounds = Vector{Hyperrectangle}(undef,size(L)[1])
            for (i, _) in enumerate(L)
                bounds[i] = Hyperrectangle(low=L[i], high=U[i])
            end
            model[:bounds] = bounds
        end
        model[:before_act] = true
    
        z = NeuralVerification.init_vars(model, problem.network, :z, with_input=true)
        
        NeuralVerification.add_set_constraint!(model, problem.input, first(z))
        for infeasible_input in infeasible_input_vec
            infeasible_halfs = constraints_list(infeasible_input)
            for infeasible_half in infeasible_halfs
                NeuralVerification.add_complementary_set_constraint!(model, infeasible_half, first(z))
            end
        end
        NeuralVerification.add_set_constraint!(model, problem.output, last(z))
        if BPO_degree == 1
            NeuralVerification.encode_network!(model, problem.network, NeuralVerification.TriangularRelaxedLP())
        else
            NeuralVerification.encode_network!(model, problem.network, NeuralVerification.BernsteinPolynomial2LP())
        end
    
    
        d = (last(z) - problem.output_ref)
        o = isnothing(u_ref) ? dot(d, d.*problem.output_cost) : dot(first(z)[5:6] - u_ref, first(z)[5:6] - u_ref)
    
        @objective(model, Min, o)
    
        isnothing(start_values) || set_start_value.(all_variables(model), start_values)
    
        if !isnothing(xu_init)
            set_start_value.(first(z), xu_init)
        end
    
        optimize!(model)
        if termination_status(model) == OPTIMAL
            new_input_center = NeuralVerification.value(first(z))
            sub_input = Hyperrectangle(new_input_center,[1e-8 for _ in 1:length(new_input_center)])
            new_model = Model(solver)
            set_silent(new_model)
    
            set_optimizer_attribute(new_model, "CPX_PARAM_EPOPT", 1e-8)
            set_optimizer_attribute(new_model, "CPX_PARAM_EPAGAP", 1e-6)
    
            if IA_bounds
                # get pre-activationlooser bounds from IBP/IA
                new_model[:bounds] = bounds = NeuralVerification.get_bounds(problem.network, sub_input, false)
            else 
                # using get_bounds in FastLin and convDual
                L, U  = get_bounds_convdual(problem.network, sub_input)
                pushfirst!(L, low(sub_input))
                pushfirst!(U, high(sub_input))
                bounds = Vector{Hyperrectangle}(undef,size(L)[1])
                for (i, _) in enumerate(L)
                    bounds[i] = Hyperrectangle(low=L[i], high=U[i])
                end
                new_model[:bounds] = bounds
            end
            new_model[:before_act] = true
    
            _z = NeuralVerification.init_vars(new_model, problem.network, :z, with_input=true)
    
            NeuralVerification.add_set_constraint!(new_model, sub_input, first(_z))
            NeuralVerification.add_complementary_set_constraint!(new_model, problem.output, last(_z))
            if BPO_degree == 1
                NeuralVerification.encode_network!(new_model, problem.network, NeuralVerification.TriangularRelaxedLP())
            else
                NeuralVerification.encode_network!(new_model, problem.network, NeuralVerification.BernsteinPolynomial2LP())
            end

    
            d = (last(_z) - problem.output_ref)
            o = isnothing(u_ref) ? dot(d, d.*problem.output_cost) : dot(first(_z)[5:6] - u_ref, first(_z)[5:6] - u_ref)
    
            @objective(new_model, Min, o)
    
            isnothing(start_values) || set_start_value.(all_variables(new_model), start_values)
    
            if !isnothing(xu_init)
                set_start_value.(first(_z), xu_init)
            end
    
            optimize!(new_model)
    
            if termination_status(new_model) == OPTIMAL
                push!(infeasible_input_vec, Hyperrectangle(new_input_center,[1e-9,1e-9,1e-9,1e-9,1e-2, 1e-2]))
                i += 1
                continue
            end
            return NeuralVerification.TrackingResult(:holds), nothing
        else
            return NeuralVerification.TrackingResult(:violated), start_values
        end
        
        
    end
    println("TIMEOUT...")
    return NeuralVerification.TrackingResult(:violated), start_values
    
end

function solve_BPO_new(solver::NNDynTrack, problem::TrackingProblem, start_values=nothing; u_ref = nothing, xu_init = nothing, IA_bounds=false, BPO_degree=1, p=2)
    i = 0
    infeasible_input_vec = []#[EmptySet(length(low(problem.input)))]
    while i < 50
    
        model = Model(solver)
        set_silent(model)

        set_optimizer_attribute(model, "CPX_PARAM_EPOPT", 1e-8)
        set_optimizer_attribute(model, "CPX_PARAM_EPAGAP", 1e-6)


        if IA_bounds
            # get pre-activationlooser bounds from IBP/IA
            model[:bounds] = bounds = NeuralVerification.get_bounds(problem.network, problem.input, false)
        else 
            # using get_bounds in FastLin and convDual
            L, U  = get_bounds_convdual(problem.network, problem.input)
            pushfirst!(L, low(problem.input))
            pushfirst!(U, high(problem.input))
            bounds = Vector{Hyperrectangle}(undef,size(L)[1])
            for (i, _) in enumerate(L)
                bounds[i] = Hyperrectangle(low=L[i], high=U[i])
            end
            model[:bounds] = bounds
        end
        model[:before_act] = true

        z = NeuralVerification.init_vars(model, problem.network, :z, with_input=true)

        NeuralVerification.add_set_constraint!(model, problem.input, first(z))
        for infeasible_input in infeasible_input_vec
            infeasible_halfs = constraints_list(infeasible_input)
            for infeasible_half in infeasible_halfs
                NeuralVerification.add_complementary_set_constraint!(model, infeasible_half, first(z))
            end
        end
        NeuralVerification.add_set_constraint!(model, problem.output, last(z))
        if BPO_degree == 1
            NeuralVerification.encode_network!(model, problem.network, NeuralVerification.TriangularRelaxedLP())
        else
            NeuralVerification.encode_network!(model, problem.network, NeuralVerification.BernsteinPolynomial2LP())
        end

        if p==1
            o = sum(NeuralVerification.symbolic_abs.(last(z) - problem.output_ref) .* problem.output_cost)
        else
            d = (last(z) - problem.output_ref)
            o = isnothing(u_ref) ? dot(d, d.*problem.output_cost) : dot(first(z)[5:6] - u_ref, first(z)[5:6] - u_ref)
        end
        @objective(model, Min, o)

        isnothing(start_values) || set_start_value.(all_variables(model), start_values)

        if !isnothing(xu_init)
            set_start_value.(first(z), xu_init)
        end

        optimize!(model)
        if termination_status(model) == OPTIMAL
            
            new_input_center = NeuralVerification.value(first(z))
            sub_input = Hyperrectangle(new_input_center,[1e-8 for _ in 1:length(new_input_center)])
            new_model = Model(solver)
            set_silent(new_model)
            set_optimizer_attribute(new_model, "CPX_PARAM_EPOPT", 1e-8)
            set_optimizer_attribute(new_model, "CPX_PARAM_EPAGAP", 1e-6)
    
            if IA_bounds
                # get pre-activationlooser bounds from IBP/IA
                new_model[:bounds] = bounds = NeuralVerification.get_bounds(problem.network, sub_input, false)
            else 
                # using get_bounds in FastLin and convDual
                L, U  = get_bounds_convdual(problem.network, sub_input)
                pushfirst!(L, low(sub_input))
                pushfirst!(U, high(sub_input))
                bounds = Vector{Hyperrectangle}(undef,size(L)[1])
                for (i, _) in enumerate(L)
                    bounds[i] = Hyperrectangle(low=L[i], high=U[i])
                end
                new_model[:bounds] = bounds
            end
            new_model[:before_act] = true
    
            _z = NeuralVerification.init_vars(new_model, problem.network, :z, with_input=true)
    
            NeuralVerification.add_set_constraint!(new_model, sub_input, first(_z))
            NeuralVerification.add_complementary_set_constraint!(new_model, problem.output, last(_z))
            if BPO_degree == 1
                NeuralVerification.encode_network!(new_model, problem.network, NeuralVerification.TriangularRelaxedLP())
            else
                NeuralVerification.encode_network!(new_model, problem.network, NeuralVerification.BernsteinPolynomial2LP())
            end
            if p==1
                o = sum(NeuralVerification.symbolic_abs.(last(_z) - problem.output_ref) .* problem.output_cost)
            else
                d = (last(_z) - problem.output_ref)
                o = isnothing(u_ref) ? dot(d, d.*problem.output_cost) : dot(first(_z)[5:6] - u_ref, first(_z)[5:6] - u_ref)
            end
            
            @objective(new_model, Min, o)
    
            isnothing(start_values) || set_start_value.(all_variables(new_model), start_values)
    
            if !isnothing(xu_init)
                set_start_value.(first(_z), xu_init)
            end
    
            optimize!(new_model)
    
            if termination_status(new_model) == OPTIMAL
                push!(infeasible_input_vec, Hyperrectangle(new_input_center,[1e-9,1e-9,1e-9,1e-9,5e-8, 5e-8]))
                i += 1
                
                continue
            end
            @show i
            return NeuralVerification.TrackingResult(:holds, NeuralVerification.value(first(z)), objective_value(model)), NeuralVerification.value.(all_variables(model)), NeuralVerification.value(last(z))

        else
            @show termination_status(model)
            return NeuralVerification.TrackingResult(:violated), start_values, nothing
        end
    end
end


function solve_BPO_new(solver::NNDynTrackGurobi, problem::TrackingProblem, start_values=nothing; u_ref=nothing, xu_init=nothing, IA_bounds=false, BPO_degree=1, p=2)
    i = 0
    infeasible_input_vec = []
    while i < 10000
        model = Model(solver)
        set_silent(model)
        set_optimizer_attribute(model, "TimeLimit", 100000)
        set_optimizer_attribute(model, "Presolve", 0) 
        set_optimizer_attribute(model, "NonConvex", 2) 

        if IA_bounds
            model[:bounds] = bounds = NeuralVerification.get_bounds(problem.network, problem.input, false)
        else 
            L, U  = get_bounds_convdual(problem.network, problem.input)
            pushfirst!(L, low(problem.input))
            pushfirst!(U, high(problem.input))
            bounds = Vector{Hyperrectangle}(undef, size(L)[1])
            for (i, _) in enumerate(L)
                bounds[i] = Hyperrectangle(low=L[i], high=U[i])
            end
            model[:bounds] = bounds
        end
        model[:before_act] = true
        z = NeuralVerification.init_vars(model, problem.network, :z, with_input=true)
        NeuralVerification.add_set_constraint!(model, problem.input, first(z))
        for infeasible_input in infeasible_input_vec
            infeasible_halfs = constraints_list(infeasible_input)
            for infeasible_half in infeasible_halfs
                NeuralVerification.add_complementary_set_constraint!(model, infeasible_half, first(z))
            end
        end
        NeuralVerification.add_set_constraint!(model, problem.output, last(z))
        if BPO_degree == 1
            NeuralVerification.encode_network!(model, problem.network, NeuralVerification.TriangularRelaxedLP())
        else
            NeuralVerification.encode_network!(model, problem.network, NeuralVerification.BernsteinPolynomial2LP())
        end

        if p == 1
            o = sum(NeuralVerification.symbolic_abs.(last(z) - problem.output_ref) .* problem.output_cost)
        else
            d = (last(z) - problem.output_ref)
            o = isnothing(u_ref) ? dot(d, d .* problem.output_cost) : dot(first(z)[5:6] - u_ref, first(z)[5:6] - u_ref)
        end
        @objective(model, Min, o)

        isnothing(start_values) || set_start_value.(all_variables(model), start_values)

        if !isnothing(xu_init)
            set_start_value.(first(z), xu_init)
        end

        optimize!(model)
        @show termination_status(model)
        first_z = NeuralVerification.value(first(z))
        obj_value = objective_value(model)
        all_var = NeuralVerification.value.(all_variables(model))
        last_z = NeuralVerification.value(last(z))

        if termination_status(model) == OPTIMAL
            new_input_center = NeuralVerification.value(first(z))
            sub_input = Hyperrectangle(new_input_center, [1e-8 for _ in 1:length(new_input_center)])
            new_model = Model(solver)
            set_silent(new_model)
            set_optimizer_attribute(model, "TimeLimit", 100000)
            set_optimizer_attribute(model, "Presolve", 0) 
            set_optimizer_attribute(model, "NonConvex", 2) 

            if IA_bounds
                new_model[:bounds] = bounds = NeuralVerification.get_bounds(problem.network, sub_input, false)
            else 
                L, U  = get_bounds_convdual(problem.network, sub_input)
                pushfirst!(L, low(sub_input))
                pushfirst!(U, high(sub_input))
                bounds = Vector{Hyperrectangle}(undef, size(L)[1])
                for (i, _) in enumerate(L)
                    bounds[i] = Hyperrectangle(low=L[i], high=U[i])
                end
                new_model[:bounds] = bounds
            end
            new_model[:before_act] = true
            _z = NeuralVerification.init_vars(new_model, problem.network, :z, with_input=true)
            NeuralVerification.add_set_constraint!(new_model, sub_input, first(_z))
            NeuralVerification.add_complementary_set_constraint!(new_model, problem.output, last(_z))
            if BPO_degree == 1
                NeuralVerification.encode_network!(new_model, problem.network, NeuralVerification.TriangularRelaxedLP())
            else
                NeuralVerification.encode_network!(new_model, problem.network, NeuralVerification.BernsteinPolynomial2LP())
            end
            if p == 1
                new_o = sum(NeuralVerification.symbolic_abs.(last(_z) - problem.output_ref) .* problem.output_cost)
            else
                d = (last(_z) - problem.output_ref)
                new_o = isnothing(u_ref) ? dot(d, d .* problem.output_cost) : dot(first(_z)[5:6] - u_ref, first(_z)[5:6] - u_ref)
            end
            @objective(new_model, Min, new_o)
            isnothing(start_values) || set_start_value.(all_variables(new_model), start_values)

            if !isnothing(xu_init)
                set_start_value.(first(_z), xu_init)
            end

            optimize!(new_model)

            if termination_status(new_model) == OPTIMAL
                push!(infeasible_input_vec, Hyperrectangle(new_input_center, [1e-11, 1e-11, 1e-11, 1e-11, rand()*1e-9, rand()*1e-9]))
                i += 1
                @show i
                continue
            end
            return NeuralVerification.TrackingResult(:holds, first_z, obj_value), all_var, last_z
        else
            @show termination_status(model)
            return NeuralVerification.TrackingResult(:violated), start_values, nothing
        end
    end
    println("TIMEOUT...")
    return NeuralVerification.TrackingResult(:violated), start_values, nothing
end


function solve_BPO(solver::NNDynTrack, problem::TrackingProblem, start_values=nothing; u_ref=nothing, xu_init=nothing, IA_bounds=false, BPO_degree=1, p=2)
    model = Model(solver)
    set_silent(model)
    set_optimizer_attribute(model, "CPX_PARAM_EPOPT", 1e-8)
    set_optimizer_attribute(model, "CPX_PARAM_EPAGAP", 1e-6)

    if IA_bounds
        model[:bounds] = bounds = NeuralVerification.get_bounds(problem.network, problem.input, false)
    else
        L, U  = get_bounds_convdual(problem.network, problem.input)
        pushfirst!(L, low(problem.input))
        pushfirst!(U, high(problem.input))
        bounds = Vector{Hyperrectangle}(undef, size(L)[1])
        for (i, _) in enumerate(L)
            bounds[i] = Hyperrectangle(low=L[i], high=U[i])
        end
        model[:bounds] = bounds
    end
    model[:before_act] = true

    z = NeuralVerification.init_vars(model, problem.network, :z, with_input=true)
    NeuralVerification.add_set_constraint!(model, problem.input, first(z))
    NeuralVerification.add_set_constraint!(model, problem.output, last(z))
    if BPO_degree == 1
        NeuralVerification.encode_network!(model, problem.network, NeuralVerification.TriangularRelaxedLP())
    else
        NeuralVerification.encode_network!(model, problem.network, NeuralVerification.BernsteinPolynomial2LP())
    end

    if p == 1
        o = sum(NeuralVerification.symbolic_abs.(last(z) - problem.output_ref) .* problem.output_cost)
    else
        d = (last(z) - problem.output_ref)
        o = isnothing(u_ref) ? dot(d, d .* problem.output_cost) : dot(first(z)[5:6] - u_ref, first(z)[5:6] - u_ref)
    end
    @objective(model, Min, o)

    isnothing(start_values) || set_start_value.(all_variables(model), start_values)
    if !isnothing(xu_init)
        set_start_value.(first(z), xu_init)
    end

    optimize!(model)
    if termination_status(model) == OPTIMAL
        return NeuralVerification.TrackingResult(:holds, NeuralVerification.value(first(z)), objective_value(model)), NeuralVerification.value.(all_variables(model)), NeuralVerification.value(last(z))
    else
        @show termination_status(model)
        return NeuralVerification.TrackingResult(:violated), start_values, nothing
    end
end

function solve_BPO(solver::NNDynTrackGurobi, problem::TrackingProblem, start_values=nothing; u_ref=nothing, xu_init=nothing, IA_bounds=false, BPO_degree=1, p=2)
    model = Model(solver)
    set_silent(model)
    set_optimizer_attribute(model, "TimeLimit", 100000)
    set_optimizer_attribute(model, "Presolve", 0)
    set_optimizer_attribute(model, "NonConvex", 2)

    if IA_bounds
        model[:bounds] = bounds = NeuralVerification.get_bounds(problem.network, problem.input, false)
    else
        L, U  = get_bounds_convdual(problem.network, problem.input)
        pushfirst!(L, low(problem.input))
        pushfirst!(U, high(problem.input))
        bounds = Vector{Hyperrectangle}(undef, size(L)[1])
        for (i, _) in enumerate(L)
            bounds[i] = Hyperrectangle(low=L[i], high=U[i])
        end
        model[:bounds] = bounds
    end
    model[:before_act] = true

    z = NeuralVerification.init_vars(model, problem.network, :z, with_input=true)
    NeuralVerification.add_set_constraint!(model, problem.input, first(z))
    NeuralVerification.add_set_constraint!(model, problem.output, last(z))
    if BPO_degree == 1
        NeuralVerification.encode_network!(model, problem.network, NeuralVerification.TriangularRelaxedLP())
    else
        NeuralVerification.encode_network!(model, problem.network, NeuralVerification.BernsteinPolynomial2LP())
    end

    if p == 1
        o = sum(NeuralVerification.symbolic_abs.(last(z) - problem.output_ref) .* problem.output_cost)
    else
        d = (last(z) - problem.output_ref)
        o = isnothing(u_ref) ? dot(d, d .* problem.output_cost) : dot(first(z)[5:6] - u_ref, first(z)[5:6] - u_ref)
    end
    @objective(model, Min, o)

    isnothing(start_values) || set_start_value.(all_variables(model), start_values)
    if !isnothing(xu_init)
        set_start_value.(first(z), xu_init)
    end

    optimize!(model)
    if termination_status(model) == OPTIMAL
        return NeuralVerification.TrackingResult(:holds, NeuralVerification.value(first(z)), objective_value(model)), NeuralVerification.value.(all_variables(model)), NeuralVerification.value(last(z))
    else
        @show termination_status(model)
        return NeuralVerification.TrackingResult(:violated), start_values, nothing
    end
end

function solve_BPO(solver::NNDynTrackNLopt, problem::TrackingProblem, start_values=nothing; u_ref=nothing, xu_init=nothing, IA_bounds=false, BPO_degree=1)
    model = Model(solver)
    set_silent(model)
    set_optimizer_attribute(model, "algorithm", :LD_SLSQP)

    if IA_bounds
        model[:bounds] = bounds = NeuralVerification.get_bounds(problem.network, problem.input, false)
    else
        L, U  = get_bounds_convdual(problem.network, problem.input)
        pushfirst!(L, low(problem.input))
        pushfirst!(U, high(problem.input))
        bounds = Vector{Hyperrectangle}(undef, size(L)[1])
        for (i, _) in enumerate(L)
            bounds[i] = Hyperrectangle(low=L[i], high=U[i])
        end
        model[:bounds] = bounds
    end
    model[:before_act] = true

    z = NeuralVerification.init_vars(model, problem.network, :z, with_input=true)
    NeuralVerification.add_set_constraint!(model, problem.input, first(z))
    NeuralVerification.add_set_constraint!(model, problem.output, last(z))
    if BPO_degree == 1
        NeuralVerification.encode_network!(model, problem.network, NeuralVerification.TriangularRelaxedLP())
    else
        NeuralVerification.encode_network!(model, problem.network, NeuralVerification.BernsteinPolynomial2LP())
    end

    d = (last(z) - problem.output_ref)
    o = isnothing(u_ref) ? dot(d, d .* problem.output_cost) : dot(first(z)[5:6] - u_ref, first(z)[5:6] - u_ref)
    @NLobjective(model, Min, o)

    isnothing(start_values) || set_start_value.(all_variables(model), start_values)
    if !isnothing(xu_init)
        set_start_value.(first(z), xu_init)
    end
    optimize!(model)

    if termination_status(model) == OPTIMAL || termination_status(model) == LOCALLY_SOLVED || termination_status(model) == ALMOST_LOCALLY_SOLVED
        return NeuralVerification.TrackingResult(:holds, NeuralVerification.value(first(z)), objective_value(model)), NeuralVerification.value.(all_variables(model)), NeuralVerification.value(last(z))
    else
        @show termination_status(model)
        return NeuralVerification.TrackingResult(:violated), start_values, nothing
    end
end

function solve_BPO(solver::NNDynTrackIpopt, problem::TrackingProblem, start_values=nothing; u_ref=nothing, xu_init=nothing, IA_bounds=false, BPO_degree=1, p=2)
    model = Model(solver)
    set_silent(model)
    set_optimizer_attribute(model, "max_cpu_time", 1000.0)
    set_optimizer_attribute(model, "print_level", 0)

    if IA_bounds
        model[:bounds] = bounds = NeuralVerification.get_bounds(problem.network, problem.input, false)
    else
        L, U  = get_bounds_convdual(problem.network, problem.input)
        pushfirst!(L, low(problem.input))
        pushfirst!(U, high(problem.input))
        bounds = Vector{Hyperrectangle}(undef, size(L)[1])
        for (i, _) in enumerate(L)
            bounds[i] = Hyperrectangle(low=L[i], high=U[i])
        end
        model[:bounds] = bounds
    end
    model[:before_act] = true

    z = NeuralVerification.init_vars(model, problem.network, :z, with_input=true)
    NeuralVerification.add_set_constraint!(model, problem.input, first(z))
    NeuralVerification.add_set_constraint!(model, problem.output, last(z))
    if BPO_degree == 1
        NeuralVerification.encode_network!(model, problem.network, NeuralVerification.TriangularRelaxedLP())
    else
        NeuralVerification.encode_network!(model, problem.network, NeuralVerification.BernsteinPolynomial2LP())
    end

    if p == 1
        o = sum(NeuralVerification.symbolic_abs.(last(z) - problem.output_ref) .* problem.output_cost)
    else
        d = (last(z) - problem.output_ref)
        o = isnothing(u_ref) ? dot(d, d .* problem.output_cost) : dot(first(z)[5:6] - u_ref, first(z)[5:6] - u_ref)
    end
    @objective(model, Min, o)

    isnothing(start_values) || set_start_value.(all_variables(model), start_values)
    if !isnothing(xu_init)
        set_start_value.(first(z), xu_init)
    end
    optimize!(model)

    if termination_status(model) == OPTIMAL || termination_status(model) == LOCALLY_SOLVED || termination_status(model) == ALMOST_LOCALLY_SOLVED
        return NeuralVerification.TrackingResult(:holds, NeuralVerification.value(first(z)), objective_value(model)), NeuralVerification.value.(all_variables(model)), NeuralVerification.value(last(z))
    else
        @show termination_status(model)
        return NeuralVerification.TrackingResult(:violated), start_values, nothing
    end
end

function solve_BPO(solver::NNDynTrackGLPK, problem::TrackingProblem, start_values=nothing; u_ref=nothing, xu_init=nothing, IA_bounds=false, BPO_degree=1, p=2)
    model = Model(solver)
    set_silent(model)
    set_optimizer_attribute(model, "tm_lim", 60 * 1_000)
    set_optimizer_attribute(model, "msg_lev", GLPK.GLP_MSG_OFF)

    if IA_bounds
        model[:bounds] = bounds = NeuralVerification.get_bounds(problem.network, problem.input, false)
    else
        L, U  = get_bounds_convdual(problem.network, problem.input)
        pushfirst!(L, low(problem.input))
        pushfirst!(U, high(problem.input))
        bounds = Vector{Hyperrectangle}(undef, size(L)[1])
        for (i, _) in enumerate(L)
            bounds[i] = Hyperrectangle(low=L[i], high=U[i])
        end
        model[:bounds] = bounds
    end
    model[:before_act] = true

    z = NeuralVerification.init_vars(model, problem.network, :z, with_input=true)
    NeuralVerification.add_set_constraint!(model, problem.input, first(z))
    NeuralVerification.add_set_constraint!(model, problem.output, last(z))
    if BPO_degree == 1
        NeuralVerification.encode_network!(model, problem.network, NeuralVerification.TriangularRelaxedLP())
    else
        NeuralVerification.encode_network!(model, problem.network, NeuralVerification.BernsteinPolynomial2LP())
    end

    if p == 1
        o = sum(NeuralVerification.symbolic_abs.(last(z) - problem.output_ref) .* problem.output_cost)
    else
        d = (last(z) - problem.output_ref)
        o = isnothing(u_ref) ? dot(d, d .* problem.output_cost) : dot(first(z)[5:6] - u_ref, first(z)[5:6] - u_ref)
    end
    @objective(model, Min, o)

    isnothing(start_values) || set_start_value.(all_variables(model), start_values)
    if !isnothing(xu_init)
        set_start_value.(first(z), xu_init)
    end

    optimize!(model)
    if termination_status(model) == OPTIMAL
        return NeuralVerification.TrackingResult(:holds, NeuralVerification.value(first(z)), objective_value(model)), NeuralVerification.value.(all_variables(model)), NeuralVerification.value(last(z))
    else
        @show termination_status(model)
        return NeuralVerification.TrackingResult(:violated), start_values, nothing
    end
end

function compute_hatzi(nnet, input)
    curr_value = input
    values = Vector{Vector}(undef, size(nnet.layers)[1])
    for (i, layer) in enumerate(nnet.layers)
        values[i] = NeuralVerification.affine_map(layer, curr_value)
        curr_value = layer.activation(NeuralVerification.affine_map(layer, curr_value))
    end
    return values
end

function solve_original_convdual(solver::NNDynTrack, problem::TrackingProblem, start_values=nothing; u_ref=nothing, xu_init=nothing)
    model = Model(solver)
    set_silent(model)
    set_optimizer_attribute(model, "CPX_PARAM_EPOPT", 1e-8)
    set_optimizer_attribute(model, "CPX_PARAM_EPAGAP", 1e-6)

    L, U  = get_bounds_convdual(problem.network, problem.input)
    pushfirst!(L, low(problem.input))
    pushfirst!(U, high(problem.input))
    bounds = Vector{Hyperrectangle}(undef, size(L)[1])
    for (i, _) in enumerate(L)
        bounds[i] = Hyperrectangle(low=L[i], high=U[i])
    end
    model[:bounds] = bounds

    model[:before_act] = true
    z = NeuralVerification.init_vars(model, problem.network, :z, with_input=true)
    δ = NeuralVerification.init_vars(model, problem.network, :δ, binary=true)

    NeuralVerification.add_set_constraint!(model, problem.input, first(z))
    NeuralVerification.add_set_constraint!(model, problem.output, last(z))
    NeuralVerification.encode_network!(model, problem.network, NeuralVerification.BoundedMixedIntegerLP())

    d = (last(z) - problem.output_ref)
    o = isnothing(u_ref) ? dot(d, d .* problem.output_cost) : dot(first(z)[5:6] - u_ref, first(z)[5:6] - u_ref)

    @objective(model, Min, o)
    isnothing(start_values) || set_start_value.(all_variables(model), start_values)
    if !isnothing(xu_init)
        set_start_value.(first(z), xu_init)
    end

    optimize!(model)
    if termination_status(model) == OPTIMAL
        return NeuralVerification.TrackingResult(:holds, NeuralVerification.value(first(z)), objective_value(model)), NeuralVerification.value.(all_variables(model))
    else
        return NeuralVerification.TrackingResult(:violated), start_values
    end
end

function solve_original(solver::NNDynTrackGurobi, problem::TrackingProblem, start_values=nothing; u_ref=nothing, xu_init=nothing, IA_bounds=true, p=2)
    model = Model(solver)
    set_silent(model)
    set_optimizer_attribute(model, "TimeLimit", 1000)
    set_optimizer_attribute(model, "Presolve", 0)
    set_optimizer_attribute(model, "NonConvex", 2)

    z = NeuralVerification.init_vars(model, problem.network, :z, with_input=true)
    δ = NeuralVerification.init_vars(model, problem.network, :δ, binary=true)

    if IA_bounds
        model[:bounds] = NeuralVerification.get_bounds(problem.network, problem.input, false)
    else
        L, U  = get_bounds_convdual(problem.network, problem.input)
        pushfirst!(L, low(problem.input))
        pushfirst!(U, high(problem.input))
        bounds = Vector{Hyperrectangle}(undef, size(L)[1])
        for (i, _) in enumerate(L)
            bounds[i] = Hyperrectangle(low=L[i], high=U[i])
        end
        model[:bounds] = bounds
    end
    model[:before_act] = true

    NeuralVerification.add_set_constraint!(model, problem.input, first(z))
    NeuralVerification.add_set_constraint!(model, problem.output, last(z))
    NeuralVerification.encode_network!(model, problem.network, NeuralVerification.BoundedMixedIntegerLP())

    if p == 1
        o = sum(NeuralVerification.symbolic_abs.(last(z) - problem.output_ref) .* problem.output_cost)
    else
        d = (last(z) - problem.output_ref)
        o = isnothing(u_ref) ? dot(d, d .* problem.output_cost) : dot(first(z)[5:6] - u_ref, first(z)[5:6] - u_ref)
    end
    @objective(model, Min, o)

    isnothing(start_values) || set_start_value.(all_variables(model), start_values)
    if !isnothing(xu_init)
        set_start_value.(first(z), xu_init)
    end

    optimize!(model)
    if termination_status(model) == OPTIMAL
        return NeuralVerification.TrackingResult(:holds, NeuralVerification.value(first(z)), objective_value(model)), NeuralVerification.value.(all_variables(model)), NeuralVerification.value(last(z))
    else
        return NeuralVerification.TrackingResult(:violated), start_values, nothing
    end
end

function solve_original(solver::NNDynTrackGLPK, problem::TrackingProblem, start_values=nothing; u_ref=nothing, xu_init=nothing, IA_bounds=true, p=2)
    model = Model(solver)
    set_silent(model)
    set_optimizer_attribute(model, "tm_lim", 60 * 1_000)
    set_optimizer_attribute(model, "msg_lev", GLPK.GLP_MSG_OFF)

    z = NeuralVerification.init_vars(model, problem.network, :z, with_input=true)
    δ = NeuralVerification.init_vars(model, problem.network, :δ, binary=true)

    if IA_bounds
        model[:bounds] = NeuralVerification.get_bounds(problem.network, problem.input, false)
    else
        L, U  = get_bounds_convdual(problem.network, problem.input)
        pushfirst!(L, low(problem.input))
        pushfirst!(U, high(problem.input))
        bounds = Vector{Hyperrectangle}(undef, size(L)[1])
        for (i, _) in enumerate(L)
            bounds[i] = Hyperrectangle(low=L[i], high=U[i])
        end
        model[:bounds] = bounds
    end
    model[:before_act] = true

    NeuralVerification.add_set_constraint!(model, problem.input, first(z))
    NeuralVerification.add_set_constraint!(model, problem.output, last(z))
    NeuralVerification.encode_network!(model, problem.network, NeuralVerification.BoundedMixedIntegerLP())

    if p == 1
        o = sum(NeuralVerification.symbolic_abs.(last(z) - problem.output_ref) .* problem.output_cost)
    else
        d = (last(z) - problem.output_ref)
        o = isnothing(u_ref) ? dot(d, d .* problem.output_cost) : dot(first(z)[5:6] - u_ref, first(z)[5:6] - u_ref)
    end
    @objective(model, Min, o)

    isnothing(start_values) || set_start_value.(all_variables(model), start_values)
    if !isnothing(xu_init)
        set_start_value.(first(z), xu_init)
    end

    optimize!(model)
    if termination_status(model) == OPTIMAL
        return NeuralVerification.TrackingResult(:holds, NeuralVerification.value(first(z)), objective_value(model)), NeuralVerification.value.(all_variables(model)), NeuralVerification.value(last(z))
    else
        return NeuralVerification.TrackingResult(:violated), start_values, nothing
    end
end

function solve_original(solver::NNDynTrack, problem::TrackingProblem, start_values=nothing; u_ref=nothing, xu_init=nothing, IA_bounds=true, p=2)
    model = Model(solver)
    set_silent(model)
    set_optimizer_attribute(model, "CPX_PARAM_EPOPT", 1e-8)
    set_optimizer_attribute(model, "CPX_PARAM_EPAGAP", 1e-6)

    z = NeuralVerification.init_vars(model, problem.network, :z, with_input=true)
    δ = NeuralVerification.init_vars(model, problem.network, :δ, binary=true)

    if IA_bounds
        model[:bounds] = NeuralVerification.get_bounds(problem.network, problem.input, false)
    else
        L, U  = get_bounds_convdual(problem.network, problem.input)
        pushfirst!(L, low(problem.input))
        pushfirst!(U, high(problem.input))
        bounds = Vector{Hyperrectangle}(undef, size(L)[1])
        for (i, _) in enumerate(L)
            bounds[i] = Hyperrectangle(low=L[i], high=U[i])
        end
        model[:bounds] = bounds
    end
    model[:before_act] = true

    NeuralVerification.add_set_constraint!(model, problem.input, first(z))
    NeuralVerification.add_set_constraint!(model, problem.output, last(z))
    NeuralVerification.encode_network!(model, problem.network, NeuralVerification.BoundedMixedIntegerLP())

    if p == 1
        o = sum(NeuralVerification.symbolic_abs.(last(z) - problem.output_ref) .* problem.output_cost)
    else
        d = (last(z) - problem.output_ref)
        o = isnothing(u_ref) ? dot(d, d .* problem.output_cost) : dot(first(z)[5:6] - u_ref, first(z)[5:6] - u_ref)
    end

    @objective(model, Min, o)
    isnothing(start_values) || set_start_value.(all_variables(model), start_values)
    if !isnothing(xu_init)
        set_start_value.(first(z), xu_init)
    end

    optimize!(model)
    if termination_status(model) == OPTIMAL
        return NeuralVerification.TrackingResult(:holds, NeuralVerification.value(first(z)), objective_value(model)), NeuralVerification.value.(all_variables(model)), NeuralVerification.value(last(z))
    else
        return NeuralVerification.TrackingResult(:violated), start_values, nothing
    end
end
