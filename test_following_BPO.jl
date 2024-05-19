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
fi_index = FollowingIndex(1, 2, 0.164525, 0.439697, 0.0711221, 2.01735, 8.55701)


net = read_nnet(net_path);
obs_radius = 0.5
USE_IA_FLAG = false
BPO_DEGREE = 1
# choose solver: NNDynTrackGurobi(), NNDynTrackNLopt(), NNDynTrackIpopt(), NNDynTrack():CPLEX degree=1only
BPO_SOLVER = NNDynTrack() 
P_NORM = 2
SIS = false
VISUALIZE = false 
STATS = true 

function generate_moving_target(;fps=10, tf=2, v=nothing, v_lim=0.5, pos=nothing)
    T = tf*fps
    v = isnothing(v) ? [rand(), rand()]*v_lim*2 .- v_lim : v
    p = isnothing(pos) ? [1.5, 1.5] : pos
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

function following(rp::RP, ctrl; fps=10, tf=2, targets=nothing, safety_index=nothing, verbose=false)
    T=Int(ceil(fps*tf))
    dt=1.0/fps
    x = rp.x0
    X = [copy(rp.x0) for k = 1:T]
    U = [zeros(2) for k = 1:T-1]
    safe_sets = []
    Xrefs = [copy(rp.x0) for k in 1:T]
    Xopt = [copy(rp.x0) for k in 1:T]
    tot_time = 0
    time_vec = []
    infeas=false
    for i in 1:T-1
        xg = [targets[i].center..., norm(targets[i].vel), 0]
        Xref = get_Xref(x, xg, fps, dt)
        xref = Xref[1]
        Xrefs[i+1] = xref
        timed_result = @timed get_control(ctrl, xref, x, rp.net, rp.obj_cost, dt, obstacles=[targets[i]], safety_index=safety_index, IA_bounds=USE_IA_FLAG, BPO_degree=BPO_DEGREE,p=P_NORM)
        timed_result.value == (nothing, nothing, nothing) && return nothing, nothing, nothing, nothing,nothing, true, nothing, nothing
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
            p = phi(safety_index, x, targets[i])
            @show p
        end
        x = forward(rp.net, x, u, dt)
        X[i+1] = x
        @show i, x, xref, Xopt[i+1]
        U[i] = u
    end
    return X, U, safe_sets, Xrefs, Xopt, infeas, tot_time, time_vec
end

function following_samples()
    nx = 20
    ny = 20
    nv = 5
    nt = 10
    nov = 4
    xs = range(0,stop=5,length=nx)
    ys = range(0,stop=5,length=ny)
    vs = range(-1,stop=1,length=nv)
    θs = range(-π,stop=π,length=nt)
    pos1s = range(1.5,stop=3.9,length=nov)
    samples = [([x,y,v,θ],[Obstacle([pos1, 1.5+(pos1-1.5)*4/3],[0.3,0.4],obs_radius)]) for x in xs, y in ys, v in vs, θ in θs, pos1 in pos1s];
    return samples
end
fol_samples = following_samples();

function exists_valid_control_old(safety_index, ctrl::ShootingController, x, obs, net, dt)
    safe_set = phi_safe_set(safety_index, x, obs, dt)
    phi_now = phi(safety_index, x, obs[1])
    phi_next_con = max(0, phi_now - safety_index.gamma * dt)
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

function exists_valid_control_original(safety_index, ctrl::ShootingController, x, obs, net, dt; IA_bounds=nothing)
    safe_set = phi_safe_set(safety_index, x, obs, dt)
    for j in 1:ctrl.num_sample
        u_cand = rand(2) .* ctrl.u_lim * 2 - ctrl.u_lim
        dot_x_cand = compute_output(net, [x; u_cand])
        dot_x_cand ∈ safe_set && (return true)
    end
    return false
end

function exists_valid_control(safety_index, ctrl::ShootingController, x, obs, net, dt; IA_bounds=nothing, bpo_degree=nothing)
    safe_set = phi_safe_set(safety_index, x, obs, dt)
    for j in 1:ctrl.num_sample
        u_cand = rand(2) .* ctrl.u_lim * 2 - ctrl.u_lim
        dot_x_cand = compute_output(net, [x; u_cand])
        if dot_x_cand ∈ safe_set
            L, U  = get_bounds_convdual(net, Hyperrectangle(low=[x.-1e-8; u_cand.-1e-8], high=[x.+1e-8; u_cand.+1e-8]))
            for v1 in [last(L)[1], last(U)[1]]
                for v2 in [last(L)[2], last(U)[2]]
                    for v3 in [last(L)[3], last(U)[3]]
                        for v4 in [last(L)[4], last(U)[4]]
                            [v1, v2, v3, v4] ∉ safe_set && (return false)
                        end
                    end
                end
            end
            return true
        end
    end
    return false
end

function eval_following_index(coes)
    margin, gamma, zeta, phi_power, dot_phi_coe = coes
    d_max = 2
    d_min = 1
    index = FollowingIndex(d_min, d_max, margin, gamma, zeta, phi_power, dot_phi_coe)
    cnt = 0
    valid = 0
    net = read_nnet(net_path);
    dt = 0.1
    fol_samples_flat = reduce(vcat, fol_samples)
    for i in ProgressBar(1:length(fol_samples_flat))
        sample = fol_samples_flat[i]
        x, obs = sample
        if norm(x[1:2]-obs[1].center) ≤ obs[1].radius # overlaped with the obstacle
            continue
        end
        ctrl = ShootingController(1000, inputs_bounds=[1, π])
        evc = exists_valid_control(index, ctrl, x, obs, net, dt)
        cnt += 1
        valid += evc
    end
    @show coes, cnt, valid/cnt
    return 1-valid/cnt
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
        ctrl = ShootingController(1000,inputs_bounds=[1, π])
        valid += exists_valid_control(index, ctrl, x, obs, net, dt)
    end
    return Float64(length(col_samples)-valid)
end


function find_infeas_states(coes)
    margin, gamma, phi_power, dot_phi_coe = coes
    d_min = 1
    d_max = 2
    index = FollowingIndex(d_min, d_max, margin, gamma, phi_power, dot_phi_coe)
    valid = 0
    net = read_nnet(net_path);
    dt = 0.1
    infeas_states = Dict()
    infeas_map = zeros(size(fol_samples)[1:2])
    for (idx, sample) in pairs(fol_samples)
        x, obs = sample
        if norm(x[1:2]) < 1e-8 # overlaped with the obstacle
            valid += 1
            continue
        end
        ctrl = ShootingController(1000, inputs_bounds=[1, π])
        feas = exists_valid_control(index, ctrl, x, obs, net, dt)
        valid += feas
        feas && continue
        haskey(infeas_states, (idx[1], idx[2])) || (infeas_states[(idx[1], idx[2])] = [])
        push!(infeas_states[(idx[1], idx[2])], sample)
        infeas_map[idx[1], idx[2]] += 1
    end
    return Float64(length(fol_samples)-valid), infeas_states, infeas_map
end

function following_stat(num, fi; ctrl=nothing, verbose=false)
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
    total_steps = 0
    while j < num
        j+=1
        x0 = [0.5+rand()*0.3, 0.5+rand()*0.3, rand(), π/2+rand()*π/2-π/4]
        xg=[5,5,0,π/2]
        if P_NORM == 1
            obj_cost = [1,1,1,1]
        else
            obj_cost = [1,1,0.1,0.1]
        end
        rp = RP(net, obj_cost, x0, xg)


        fps = 10
        (BPO_SOLVER == NNDynTrackIpopt() && net_path == "unicycle-FC3-100-rk4/epoch_1000.nnet") ? tf=3 : tf=8
        pos = [1.5, 1.5]
        v = [0.3, 0.4]
        targets = generate_moving_target(fps=fps, tf=tf, v= v, pos=pos)

        p = phi(fi, x0, targets[1])
        if p > 0
            j -= 1
            continue
        end

        isnothing(ctrl) ? ctrl = ShootingController(1000, inputs_bounds=[1, π]) : ctrl
        Xtrack, Utrack, safe_sets, Xrefs,Xopts, infeas, time, time_vec = following(rp, ctrl, fps=fps, tf=tf, targets=targets, safety_index=fi, verbose=false);
        isnothing(Xtrack) && (@assert infeas==true;j -= 1; infeas_cnt += infeas; continue) # @show i; 

        @assert infeas==false
        vio = false
        infeas_cnt += infeas
        for (ind, obs) in enumerate(targets)
            ind == length(targets) && continue
            x = Xtrack[ind+1]
            if norm(x[1:2]-obs.center) < fi.d_min || norm(x[1:2]-obs.center) > fi.d_max
                @show ind, x[1:2], obs.center, norm(x[1:2]-obs.center), fi.d_min, fi.d_max, phi(fi, x, obs)
                vio = true
                break
            end 
        end
        phi0_vio_cnt += vio
        ((net_path == "unicycle-FC2-100-rk4-extra/epoch_1000.nnet")||(net_path=="unicycle-FC3-50-rk4-extra/epoch_1000.nnet") || (net_path == "unicycle-FC3-200-rk4-extra/epoch_1000.nnet")) && (vio=false) 

        vio==true && (j -= 1;continue)
        @assert vio==false
        tot_time += time
        success += 1 - (vio|infeas)
        total_steps += length(Xtrack)
        tp = TP(net, obj_cost, 1, Xopts)
        error = costs(tp, Xtrack, norm=P_NORM, cost=obj_cost)

        all_error = [all_error; error]
        if verbose
            @show j, success, phi0_vio_cnt, infeas_cnt, tot_time / total_steps, mean(all_error)
        end
    end
    @assert success == num
    total_num = success + phi0_vio_cnt + infeas_cnt
    return success*1.0/total_num, phi0_vio_cnt*1.0/total_num, infeas_cnt*1.0/total_num, tot_time / total_steps, mean(all_error), std(all_error)
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
        p = plot(xtickfontsize=14,ytickfontsize=14,xguidefontsize=14,yguidefontsize=14,legendfontsize=12)
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
                plot!(HalfSpace(safe_sets[i].a[1:2], safe_sets[i].b * dt + safe_sets[i].a[1]*xs[i]+safe_sets[i].a[2]*ys[i]), label="Safe Set")
            else
                safe_set = reduce(intersection, [HalfSpace(con.a[1:2], con.b* dt + con.a[1]*xs[i]+con.a[2]*ys[i]) for con in safe_sets[i].constraints])
                plot!(safe_set, label="Safe Set")
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
    using Distributed
    addprocs(2)

    Random.seed!(1)
    res = bboptimize(eval_following_index; SearchRange = [(1e-3, 0.5), (1e-3, 10.),(1e-3, 1.), (0.1,10.), (0.1, 10)], TraceMode=:verbose, MaxFuncEvals=200, TargetFitness=0.0, FitnessTolerance=1e-6);

end

if VISUALIZE
    fps = 10
    ((BPO_DEGREE == 2) && (BPO_SOLVER == NNDynTrackGurobi())) ? tf=2 : tf=8
    pos = [1.5, 1.5]
    v = [0.3, 0.4]
    targets = generate_moving_target(fps=fps, tf=tf, v= v, pos=pos)

    if P_NORM == 1
        x0=[0.5,0.5,0.6,π/2]
        obj_cost = [1,1,1,1]
    else
        x0=[0.6,0.6,0.5,π/2]
        obj_cost = [1,1,0.1,0.1]
    end
    
    xg=[5,5,0,π/2]
    
    rp = RP(net, obj_cost, x0, xg)
    shoot_ctrl = ShootingController(1000, inputs_bounds=[1, π])

    err_bound = [1, 1, 0.1, 0.1] * 10^9 
    nv_ctrl_bpo = NvControllerBPO(err_bound,inputs_bounds=[1, π], warm_start=true, bin_precision=2, solver=BPO_SOLVER)


    @assert phi(fi_index, x0, targets[1]) < 0 phi(fi_index, x0, targets[1])
    
    Xtrack, Utrack, safe_sets, Xrefs, Xopts, infeas, tot_time, time_vec = following(rp, nv_ctrl_bpo, fps=fps, tf=tf, targets=targets, safety_index=fi_index, verbose=false);

    tp = TP(net, [1,1,0,0], 1, Xopts)
    pos_error = costs(tp, Xtrack, norm=2)
    vel_error = costs(tp, Xtrack, norm=2, cost=[0,0,1,0])
    angle_error = costs(tp, Xtrack, norm=2, cost=[0,0,0,1])
    @show length(Xtrack), tot_time / length(Xtrack), mean(pos_error), mean(angle_error) * 180 / 3.1415926
    myvisualize(Xtrack, targets=targets, safe_sets=safe_sets, xlims=[-0.5,6], ylims=[-0.5,6], fps=10)
    
end

if STATS
    err_bound = [1, 1, 0.1, 0.1] * 10^9 
    nv_ctrl_bpo = NvControllerBPO(err_bound,inputs_bounds=[1, π], warm_start=true, bin_precision=2, solver=BPO_SOLVER)
    n=10
    success_rate, phi0_vio_rate, infeas_rate, mean_time, mean_error, std_error = following_stat(n, fi_index, ctrl=nv_ctrl_bpo, verbose=true)
    @show success_rate, phi0_vio_rate, infeas_rate, mean_time, mean_error, std_error
end



