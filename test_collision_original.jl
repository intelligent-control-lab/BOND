using Plots
using LinearAlgebra
using Revise
using NeuralVerification
using NeuralVerification:Network, Layer, ReLU, Id, read_nnet, compute_output
using LazySets
using Random
using BlackBoxOptim
using ProgressBars
using Statistics
pyplot()
include("unicycle_env.jl")
include("controller.jl")
include("problem.jl")
include("safe_set.jl")


net_path = "unicycle-FC3-50-rk4-extra/epoch_1000.nnet"
ci_index = CollisionIndex(0.702511, 5.18048, 0.142777, 3.28991)


net = read_nnet(net_path);
obs_radius = 0.5
USE_IA_FLAG = false
# choose solver: NNDynTrackGurobi(),  NNDynTrack():CPLEX, NNDynTrackGLPK
BPO_SOLVER = NNDynTrack()
P_NORM=2
SIS = false
VISUALIZE = false 
STATS = true

function generate_moving_target(;fps=10, tf=2, v=nothing, v_lim=0.5)
    T = tf*fps
    v = isnothing(v) ? [rand(), rand()]*v_lim*2 .- v_lim : v
    p = [0, 1.5]
    return [Obstacle(p+v*(i/fps), v, obs_radius) for i in 0:T-1]
end

function get_Xref(x0, xg, T, dt)
    tf = T*dt
    dp = [xg[1]-x0[1], xg[2]-x0[2]]
    da = xg[4]-x0[4]
    a = atan(dp[2], dp[1])
    v = norm(dp)/tf
    v = max(min(v, 1),-1)
    vx = v * cos(a)
    vy = v * sin(a)
    Xref = [[x0[1]+vx*k*dt, x0[2]+vy*k*dt, v, a] for k = 1:T]
    Xref[end][3] = 0
    return Xref
end

function tracking(rp::RP, ctrl; fps=10, tf=2, obstacles=nothing, safety_index=nothing, verbose=false)
    T=Int(ceil(fps*tf))
    # @show T
    dt=1.0/fps
    x = rp.x0
    X = [copy(rp.x0) for k = 1:T]
    U = [zeros(2) for k = 1:T-1]
    safe_sets = []
    Xrefs = [copy(rp.x0) for k in 1:T]
    Xopt = [copy(rp.x0) for k in 1:T]
    tot_time = 0
    time_vec = []
    col_cnt = 0
    infeas=false
    for i in 1:T-1
        Xref = get_Xref(x, rp.xg, fps, dt)
        xref = Xref[1]
        Xrefs[i+1] = xref
        
        timed_result = @timed get_control(ctrl, xref, x, rp.net, rp.obj_cost, dt, obstacles=obstacles, safety_index=safety_index, IA_bounds=USE_IA_FLAG,p=P_NORM)
        timed_result.value == (nothing, nothing ,nothing) && return nothing, nothing, nothing,nothing, nothing, true, nothing, nothing
        u, safe_set, last_z = timed_result.value
        
        Xopt[i+1] = x + last_z * dt
        dot_x = compute_output(net, [x; u])
        if !(dot_x ∈ safe_set)
            println("ALERT! Pass verification but execuated not safe, maybe floating error exists or try better SIS, still return feasible")
        end
        push!(safe_sets, safe_set)
        tot_time += timed_result.time
        push!(time_vec, timed_result.time)
        if verbose
            @show x
            @show xref
            @show u
            p = phi(x, obstacle)
            @show p
        end
        x = forward(rp.net, x, u, dt)
        X[i+1] = x
        U[i] = u
        if norm(x[1:2] - rp.xg[1:2]) < 0.1 
            return X[1:i+1], U[1:i], safe_sets[1:i], Xrefs[1:i+1], Xopt[1:i+1], infeas, tot_time, time_vec
        end
    end
    return X, U, safe_sets, Xrefs, Xopt, infeas, tot_time, time_vec
end

function collision_samples()
    nx = 20
    ny = 20
    nv = 10
    nt = 10
    xs = range(0,stop=5,length=nx)
    ys = range(0,stop=5,length=ny)
    vs = range(-2,stop=2,length=nv)
    θs = range(-π,stop=π,length=nt)
    samples = [([x,y,v,θ],[Obstacle([2.5, 2.5],[0,0],obs_radius)]) for x in xs, y in ys, v in vs, θ in θs];
    return samples
end
col_samples = collision_samples();

function exists_valid_control_original(safety_index, ctrl::ShootingController, x, obs, net, dt; IA_bounds=nothing)
    safe_set = phi_safe_set(safety_index, x, obs, dt)
    phi_now = phi(safety_index, x, obs[1])
    for j in 1:ctrl.num_sample
        u_cand = rand(2) .* ctrl.u_lim * 2 - ctrl.u_lim
        x_cand = forward(net, x, u_cand, dt)
        phi_next = phi(safety_index, x_cand, obs[1])
        if phi_next < phi_next_con
            return true
        end
    end
    return false
end
function exists_valid_control(safety_index, ctrl::ShootingController, x, obs, net, dt; IA_bounds=nothing)
    safe_set = phi_safe_set(safety_index, x, obs, dt)
    for j in 1:ctrl.num_sample
        u_cand = rand(2) .* ctrl.u_lim * 2 - ctrl.u_lim
        dot_x_cand = compute_output(net, [x; u_cand])
        dot_x_cand ∈ safe_set && (return true)
    end
    return false
end
function exists_valid_control(safety_index, ctrl::NvController, x, obs, net, dt; IA_bounds=true)
    safe_set = phi_safe_set(safety_index, x, obs, dt)
    obj_cost = [0,0,0,0]
    
    input = Hyperrectangle(low=[x.-ctrl.ϵ; -ctrl.u_lim], high=[x.+ctrl.ϵ; ctrl.u_lim])
    dot_x_ref = zeros(size(x))
    
    start_values = nothing
    result = nothing

    output = Hyperrectangle(dot_x_ref, ctrl.err_bound/dt)
    output = intersection(output, safe_set)
    isempty(output) && return false
    problem = TrackingProblem(net, input, output, dot_x_ref, obj_cost* (dt^2))

    if IA_bounds
        result, start_values, last_z = solve_original_IA(ctrl.solver, problem, ctrl.start_values, u_ref=nothing, xu_init=nothing)
    else
        result, start_values = solve_original_convdual(ctrl.solver, problem, ctrl.start_values, u_ref=nothing, xu_init=nothing)
    end

    result.status == :violated && (return false)
    
    return true
end

function eval_collision_index(coes; whole_space=true)
    margin, gamma, phi_power, dot_phi_coe = coes
    index = CollisionIndex(margin, gamma, phi_power, dot_phi_coe)
    valid = 0
    net = read_nnet(net_path);
    dt = 0.1
    tot_cnt = 0
    col_samples_flat = reduce(vcat, col_samples)
    for i in ProgressBar(1:length(col_samples_flat))
        sample = col_samples_flat[i]
        x, obs = sample
        p = phi(index, x, obs[1])
        if !whole_space && (isnan(p) || abs(p) > 1e-1) # only evaluate on the boundary
            continue
        end
        if norm(x[1:2] - obs[1].center) ≤ obs[1].radius # overlaped with the obstacle
            continue
        end
        tot_cnt += 1
        ctrl = ShootingController(1000, inputs_bounds=[2, π])
        evc = exists_valid_control(index, ctrl, x, obs, net, dt; IA_bounds=USE_IA_FLAG)
        valid += evc
    end
    @show coes, tot_cnt, valid / tot_cnt
    return (tot_cnt - valid) / tot_cnt
end




function draw_heat_plot(coes)
    margin, gamma, phi_power, dot_phi_coe = coes
    index = CollisionIndex(margin, gamma, phi_power, dot_phi_coe)
    valid = 0
    net = read_nnet(net_path);
    dt = 0.1
    for sample in col_samples
        x, obs = sample
        if norm(x[1:2]) < 1e-8 # overlaped with the obstacle
            valid += 1
            continue
        end
        ctrl = ShootingController(1000, inputs_bounds=[2, π])
        valid += exists_valid_control(index, ctrl, x, obs, net, dt)
    end
    return Float64(length(col_samples)-valid)
end

function find_infeas_states(coes)
    margin, gamma, phi_power, dot_phi_coe = coes
    index = CollisionIndex(margin, gamma, phi_power, dot_phi_coe)
    valid = 0
    net = read_nnet(net_path);
    dt = 0.1
    infeas_states = Dict()
    infeas_map = zeros(size(col_samples)[1:2])
    for (idx, sample) in pairs(col_samples)
        x, obs = sample
        if norm(x[1:2]) < 1e-8 # overlaped with the obstacle
            valid += 1
            continue
        end
        ctrl = ShootingController(1000, inputs_bounds=[2, π])
        feas = exists_valid_control(index, ctrl, x, obs, net, dt)
        valid += feas
        feas && continue
        haskey(infeas_states, (idx[1], idx[2])) || (infeas_states[(idx[1], idx[2])] = [])
        push!(infeas_states[(idx[1], idx[2])], sample)
        infeas_map[idx[1], idx[2]] += 1
    end
    return Float64(length(col_samples)-valid), infeas_states, infeas_map
end


function collision_stat(num, ci; ctrl=nothing, verbose=false)
    all_pos = []
    all_vel = []
    all_angle = []
    all_error = []
    tot_time = 0
    Random.seed!(127)
    success = 0
    phi0_vio_cnt = 0
    infeas_cnt = 0
    j = 0
    n=1
    total_steps = 0
    while j < num
        j += 1
        @show j
        obstacles = [Obstacle([2.5,2.5], [0,0], obs_radius) for i in 1:n]
        x0 = [rand(),rand(),rand()*4-2,π/2+rand()*π/2-π/4]
        xg = [5,5,0,-π]
        p = phi(ci, x0, obstacles[1])
        if p > 0
            j -= 1
            continue
        end
        obj_cost = [1,1,1,0.1]
        rp = RP(net, obj_cost, x0, xg)
        
        ctrl = isnothing(ctrl) ? ShootingController(1000, inputs_bounds=[2, π]) : ctrl
        Xtrack, Utrack, safe_sets, Xrefs,Xopts, infeas, time, time_vec = tracking(rp, ctrl, fps=10, tf=20, obstacles=obstacles, safety_index=ci, verbose=false);
        isnothing(Xtrack) && (@assert infeas==true;j -= 1; infeas_cnt += infeas; continue) # @show i; 
        @assert infeas==false
        infeas_cnt += infeas
        vio=false
        for obs in obstacles
            # all the obses are the same... can be reduced to one for loop...
            for x in Xtrack
                if norm(x[1:2]-obs.center) < obs.radius
                    vio=true
                    break
                end 
            end
        end
        phi0_vio_cnt += vio
        vio==true && (j -= 1;continue)
        @assert vio==false
        tot_time += time
        success += 1-(vio|infeas)
        total_steps += length(Xtrack)
        tp = TP(net, obj_cost, 1, Xopts)
        error = costs(tp, Xtrack, norm=P_NORM, cost=obj_cost)
        all_error = [all_error; error]
        if verbose
            @show j, success, phi0_vio_cnt, infeas_cnt, tot_time / total_steps, mean(all_error)#, mean(all_angle)
        end
    end
    @assert success == num
    total_num = success + phi0_vio_cnt + infeas_cnt
    return success*1.0/total_num, phi0_vio_cnt*1.0/total_num, infeas_cnt*1.0/total_num, tot_time / total_steps, mean(all_error), std(all_error)#, mean(all_vel), std(all_vel), mean(all_angle), std(all_angle)
end



function myvisualize(X; Xref=nothing, Xmpcs=nothing, xlims = nothing, ylims = nothing, obstacles=nothing, targets=nothing, safe_sets=nothing, save_name=nothing, fps=10, save_frame=nothing, traj_label=nothing, time_vec=nothing)
    step = length(X)
    if !isnothing(Xref)
        xrefs = [Xref[i][1] for i in 1:length(Xref)]
        yrefs = [Xref[i][2] for i in 1:length(Xref)]
    end
    xs = [X[i][1] for i in 1:length(X)]
    ys = [X[i][2] for i in 1:length(X)]
    dt = 1 / fps
    xlims == nothing && (xlims = [min(xs), max(xs)])
    ylims == nothing && (ylims = [min(ys), max(ys)])
    for i = 1:step-1
        dpi = isnothing(save_name) & (isnothing(save_frame) || save_frame[1] != i) ? 100 : 300
        x = X[i]
        
        l = 0.2
        vx, vy, w1x, w1y, w2x, w2y = get_unicycle_endpoints(x, 0.2)
        p = plot(xtickfontsize=14,ytickfontsize=14,xguidefontsize=14,yguidefontsize=14,legendfontsize=14,xticks = -1:1:6,yticks = -1:1:6)
        plot!(p, w1x, w1y, linewidth=43, color=:black, label="")
        plot!(p, w2x, w2y, linewidth=43, color=:black, label="")
        plot!(p, vx, vy, linewidth=34, xlims = xlims, ylims = ylims, color=2, label="", aspect_ratio=:equal, dpi=dpi, legend=:bottomright)

        if !isnothing(obstacles)
            for obs in obstacles
                plot!(p, Ball2(Float64.(obs.center), obs.radius))
            end
        end
        if !isnothing(targets)
            plot!(p, Ball2(Float64.(targets[i].center), targets[i].radius))
        end
        if !isnothing(Xref)
            plot!(xrefs[1:i], yrefs[1:i], label="Reference Trajectory")
        end
        if !isnothing(Xmpcs)
            plot!([Xmpcs[i][j][1] for j in 1:length(Xmpcs[i])], [Xmpcs[i][j][2] for j in 1:length(Xmpcs[i])], label="Xmpc")
            scatter!([Xmpcs[i][1][1]], [Xmpcs[i][1][2]], label="current mpc")
        end
        plot!(xs[1:i], ys[1:i], label=isnothing(traj_label) ? "Executed Trajectory" : traj_label, color=3)
        if !isnothing(safe_sets)
            if isa(safe_sets[i], HalfSpace)
                plot!(HalfSpace(safe_sets[i].a[1:2], safe_sets[i].b * dt + safe_sets[i].a[1]*xs[i]+safe_sets[i].a[2]*ys[i]), label="Safety Constraint")
            else
                safe_set = reduce(intersection, [HalfSpace(con.a[1:2], con.b* dt + con.a[1]*xs[i]+con.a[2]*ys[i]) for con in safe_sets[i].constraints])
                plot!(safe_set, label="Safety Constraint")
            end
        end
        
        if !isnothing(save_frame) && save_frame[1] == i
            savefig(save_frame[2])
        end
        if isnothing(save_name)
            display(p)
            isnothing(time_vec) ? sleep(1/fps) : sleep(time_vec[i])
        end
    end
    isnothing(save_name) || return gif(anim, save_name, fps = fps)
end

if SIS
    Random.seed!(0)
    x0 = [0.1, 5, 2.0, 1.0]
    res = bboptimize(eval_collision_index, x0; SearchRange = [(1e-3, 1), (5, 10), (0.1, 5.0), (0.1, 5.0)], TraceMode=:verbose, MaxFuncEvals=100, TargetFitness=0.0, FitnessTolerance=1e-6);
end

if VISUALIZE
    n = 1
    obstacles = [Obstacle([2.5,2.5], [0,0], obs_radius) for i in 1:n]
    x0 = [0,0,0, π/2]
    xg = [5,5,0,-π]
    @show obstacles, x0, xg
    obj_cost = [1,1,1,0.1]

    rp = RP(net, obj_cost, x0, xg)
    shoot_ctrl = ShootingController(1000, inputs_bounds=[2, π])
    err_bound = [1, 1, 0.1, 0.1] * 10^9 
    nv_ctrl = NvController(err_bound,inputs_bounds=[2, π], warm_start=true, bin_precision=2,solver = BPO_SOLVER)

    
    @assert phi(ci_index, x0, obstacles[1]) < 0
    (net_path == "unicycle-FC4-100-rk4-extra/epoch_1000.nnet") || (net_path == "unicycle-FC3-200-rk4-extra/epoch_1000.nnet") ? tf=1 : tf=50
    Xtrack, Utrack, safe_sets, Xrefs, Xopts, infeas, time, time_vec = tracking(rp, nv_ctrl, fps=10, tf=tf, obstacles=obstacles, safety_index=ci_index, verbose=false);

    tp = TP(net, obj_cost, 1, Xopts)
    error = costs(tp, Xtrack, norm=P_NORM, cost=obj_cost)

    @show length(Xtrack), time/length(Xtrack), mean(error)
    myvisualize(Xtrack, obstacles=obstacles, xlims=[-1.0,6.0], ylims=[-1.0,6.0], fps=10,safe_sets=safe_sets)
    
end


if STATS
    err_bound = [1, 1, 0.1, 0.1] * 10^9 
    nv_ctrl = NvController(err_bound,inputs_bounds=[2, π], warm_start=true, bin_precision=2,solver = BPO_SOLVER)
    n=10
    success_rate, phi0_vio_rate, infeas_rate, mean_time, mean_error, std_error = collision_stat(n, ci_index, ctrl=nv_ctrl, verbose=true)
    @show success_rate, phi0_vio_rate, infeas_rate, mean_time, mean_error, std_error
end
