mutable struct CollisionIndex
    margin
    gamma
    zeta
    phi_power
    dot_phi_coe
end

mutable struct FollowingIndex
    d_min
    d_max
    margin
    gamma
    zeta
    phi_power
    dot_phi_coe
end


mutable struct Obstacle
    center
    vel
    radius
end


function CollisionIndex(margin, gamma, phi_power, dot_phi_coe)
    CollisionIndex(margin, gamma, 0, phi_power, dot_phi_coe)
end

function FollowingIndex(d_min,d_max,margin,gamma,phi_power,dot_phi_coe)
    FollowingIndex(d_min,d_max,margin,gamma,0,phi_power,dot_phi_coe)
end

function phi_safe_set(index, x, obstacles, dt)
    safe_set = nothing
    for obs in obstacles
        p = phi(index, x, obs)
        p_phi_p_x, p_phi_p_o = grad_phi(index, x, obs)
        dot_o = [obs.vel; [0.,0.]]
        d = p < index.gamma * dt ? -p/dt-index.zeta/dt : -index.gamma-index.zeta/dt
        safe_set = isnothing(safe_set) ? HalfSpace(p_phi_p_x, d) : intersection(HalfSpace(p_phi_p_x, d), safe_set)
    end
    return safe_set
end


function phi(index::CollisionIndex, x, obs)
    o = [obs.center; [0,0]]
    d = sqrt((x[1]-o[1])^2 + (x[2]-o[2])^2)
    dM = [x[1]-o[1], x[2]-o[2], x[3]*cos(x[4])-o[3], x[3]*sin(x[4])-o[4]]
    dp = dM[[1,2]]
    dv = dM[[3,4]]
    dot_d = dp'dv / d
    return (index.margin + obs.radius)^index.phi_power - d^index.phi_power - index.dot_phi_coe*dot_d
end

function grad_phi(index::CollisionIndex, x, obs)
    o = [obs.center; [0,0]]
    d = (x[1]-o[1])^2 + (x[2]-o[2])^2
    grad_d = [2*(x[1]-o[1]), 2*(x[2]-o[2]), 0, 0]

    d = sqrt((x[1]-o[1])^2 + (x[2]-o[2])^2)
    dM = [x[1]-o[1], x[2]-o[2], x[3]*cos(x[4])-o[3], x[3]*sin(x[4])-o[4]]
    dim = 2
    dp = dM[[1,dim]]
    dv = dM[[dim+1,dim*2]]
    dot_d = dp'dv / d
    p_dot_d_p_dp = dv / d - (dp'dv) * dp / (d^3)
    p_dot_d_p_dv = dp / d
    p_dp_p_M = hcat(I(dim), zeros(dim,dim))
    p_dv_p_M = hcat(zeros(dim,dim), I(dim))
    p_dot_d_p_M = p_dp_p_M'p_dot_d_p_dp + p_dv_p_M'p_dot_d_p_dv
    p_M_p_X = [
        1 0 0 0
        0 1 0 0
        0 0 cos(x[4]) -x[3]*sin(x[4])
        0 0 sin(x[4]) x[3]*cos(x[4])
    ]
    p_dot_d_p_X = p_M_p_X'p_dot_d_p_M
    grad_dot_d = vec(p_dot_d_p_X)
    return -index.phi_power*d^(index.phi_power-1)*grad_d - index.dot_phi_coe*grad_dot_d, zeros(4)
end

function phi(index::FollowingIndex, x, target)
    o = [target.center; target.vel]
    d = sqrt((x[1]-o[1])^2 + (x[2]-o[2])^2)
    dM = [x[1]-o[1], x[2]-o[2], x[3]*cos(x[4])-o[3], x[3]*sin(x[4])-o[4]]
    dim = 2
    dp = dM[[1,dim]]
    dv = dM[[dim+1,dim*2]]
    dot_d = dp'dv / d

    d_min = index.d_min + index.margin
    d_max = index.d_max - index.margin
    return -((d_max - d_min)/2)^index.phi_power + abs(d - (d_min + d_max)/2)^index.phi_power + index.dot_phi_coe * (2*d*dot_d - dot_d*(d_min + d_max)) #old
end

function grad_phi(index::FollowingIndex, x, target)
    o = [target.center; target.vel]
    d = (x[1]-o[1])^2 + (x[2]-o[2])^2
    grad_d = [2*(x[1]-o[1]), 2*(x[2]-o[2]), 0, 0]

    d = sqrt((x[1]-o[1])^2 + (x[2]-o[2])^2)
    dM = [x[1]-o[1], x[2]-o[2], x[3]*cos(x[4])-o[3], x[3]*sin(x[4])-o[4]]
    dim = 2
    dp = dM[[1,dim]]
    dv = dM[[dim+1,dim*2]]
    
    p_Mr_p_Xr = [
        1 0 0 0
        0 1 0 0
        0 0 cos(x[4]) -x[3]*sin(x[4])
        0 0 sin(x[4]) x[3]*cos(x[4])
    ]
    p_Mh_p_Xh = I(4)
    #dot_d is the component of velocity lies in the dp direction
    dot_d = dp'dv / d

    p_dot_d_p_dp = dv / d - (dp'dv) * dp / (d^3)
    p_dot_d_p_dv = dp / d

    p_dp_p_Mr = hcat(I(dim), zeros(dim,dim))
    p_dp_p_Mh = -p_dp_p_Mr

    p_dv_p_Mr = hcat(zeros(dim,dim), I(dim))
    p_dv_p_Mh = -p_dv_p_Mr

    p_dot_d_p_Mr = p_dp_p_Mr'p_dot_d_p_dp + p_dv_p_Mr'p_dot_d_p_dv
    p_dot_d_p_Mh = p_dp_p_Mh'p_dot_d_p_dp + p_dv_p_Mh'p_dot_d_p_dv

    p_dot_d_p_Xr = p_Mr_p_Xr'p_dot_d_p_Mr
    p_dot_d_p_Xh = p_Mh_p_Xh'p_dot_d_p_Mh

    d = (d == 0) ? 1e-3 : d
    dot_d = (dot_d == 0) ? 1e-3 : dot_d

    p_d_p_Mr = vcat(dp / d, zeros(dim))
    p_d_p_Mh = vcat(-dp / d, zeros(dim))

    p_d_p_Xr = p_Mr_p_Xr'p_d_p_Mr
    p_d_p_Xh = p_Mh_p_Xh'p_d_p_Mh
    
    
    d_min = index.d_min + index.margin
    d_max = index.d_max - index.margin

    pos = d - (d_min + d_max)/2
    p_phi_p_d = sign(pos)*index.phi_power*abs(pos)^(index.phi_power-1) + 2 * index.dot_phi_coe * dot_d #old
    p_phi_p_dot_d = index.dot_phi_coe * 2 * pos #old

    p_phi_p_x = p_d_p_Xr'p_phi_p_d + p_dot_d_p_Xr'p_phi_p_dot_d
    p_phi_p_o = p_d_p_Xh'p_phi_p_d + p_dot_d_p_Xh'p_phi_p_dot_d

    return p_phi_p_x', p_phi_p_o'
end
