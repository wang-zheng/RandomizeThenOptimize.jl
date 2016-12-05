module RandomizeThenOptimize
	using NLopt

    # ===== TYPE: Problem =====
    # contains everything we need to describe inference problem
    export Problem
    type Problem
        n::Integer #dimension of parameters
        m::Integer #dimension of data
        f::Function #forward model
        θ_pr::AbstractVector # prior mean
        y::AbstractVector # observation data
        L_pr::AbstractMatrix #L'*L = Covariance^(-1), for prior
        L_obs::AbstractMatrix #L'*L = Covariance^(-1), for observational noise
        opt::NLopt.Opt #NLopt optimization settings
    end

    function Problem(n::Integer,m::Integer)
        opt = Opt(:LD_SLSQP,n)
        xtol_rel!(opt,1e-8)
        ftol_rel!(opt,1e-8)
        Problem(n,m,x->x,zeros(n),zeros(m),eye(n),eye(m),opt)
    end

    import Base.print
    export print
    function print(p::Problem)
        print("Problem(",p.n,")")
    end

    # functions to setup problem
    export forward_model!, nlopt_opt!, pr_mean!, pr_covariance!, pr_σ!, pr_precision!, obs_data!, obs_covariance!, obs_σ!, obs_precision!
    function forward_model!(p::Problem,f::Function)
        p.f = f
    end

    function nlopt_opt!(p::Problem, opt::NLopt.Opt)
        p.opt = opt
    end

    function pr_mean!(p::Problem, x::AbstractVector)
        p.θ_pr = x
    end

    function pr_mean!(p::Problem, x::Real)
        pr_mean!(p,[x])
    end

    function pr_covariance!(p::Problem, C::AbstractMatrix)
        L_inv = chol(C)
        p.L_pr = inv(L_inv)
    end

    function pr_σ!(p::Problem, σ::AbstractVector)
        p.L_pr = Diagonal(1 ./ σ)
    end

    function pr_σ!(p::Problem, σ::Real)
        pr_σ!(p,[σ])
    end

    function pr_precision!(p::Problem, C_inv::AbstractMatrix)
        p.L_pr = chol(C_inv)'
    end

    function obs_data!(p::Problem, y::AbstractVector)
        p.y = y
    end

    function obs_data!(p::Problem, y::Real)
        obs_data!(p,[y])
    end

    function obs_covariance!(p::Problem, C::AbstractMatrix)
        L_inv = chol(C)
        p.L_obs = inv(L_inv)
    end

    function obs_σ!(p::Problem, σ::AbstractVector)
        p.L_obs = Diagonal(1 ./ σ)
    end

    function obs_σ!(p::Problem, σ::Real)
        obs_σ!(p,[σ])
    end

    function obs_precision!(p::Problem, C_inv::AbstractMatrix)
        p.L_obs = chol(C_inv)'
    end

    # ===== Useful Utilities =====
    # hcat for each element of the tuple
    function tuple_hcat(a::Tuple, b::Tuple)
        l = length(a)
        c = [ hcat(a[i],b[i]) for i = 1:l ]
        return (c...)
    end

    # functions used in rto_mcmc
    function residual!(x::AbstractVector, jac::AbstractMatrix, p::Problem)
        
        if length(jac) > 0
            jacf = view(jac,1:p.m,1:p.n)
            fx = p.f(x,jacf)
            jac[:] = [p.L_pr; p.L_obs*jacf]
        else 
            fx = p.f(x,zeros(Float64,0,0))
        end
        
        r = [p.L_pr*(x - p.θ_pr); p.L_obs*(fx - p.y)]
        
        return r
    end

    function Q_residual!(x::AbstractVector, jac::AbstractMatrix, p::Problem, Q::AbstractMatrix, ξ::AbstractVector, temp::AbstractMatrix)
        
        if length(jac)>0
            Qr = Q'*residual!(x,temp,p) - ξ;
            jac[:] = Q'*temp
        else
            Qr = Q'*residual!(x,zeros(0,0),p) - ξ;
        end
        
        return Qr
    end

    function neglogpost!(x::AbstractVector, grad::AbstractVector, n::Integer, m::Integer, resid::Function)
        if length(grad) > 0
            jac = zeros(Float64,m,n)
        else
            jac = zeros(Float64,0,0)
        end
        
        r::AbstractVector = resid(x,jac)
        
        if length(grad) > 0
            g::AbstractVector = jac'*r
            grad[:] = g
        end
        
        return dot(r,r)/2
    end

    # main functions for rto
    function find_map(p::Problem)
        opt = copy(p.opt)
        resid = (x,jac) -> residual!(x,jac,p)
        min_objective!(opt, (x,g) -> neglogpost!(x,g,p.n,p.n+p.m,resid) )
        print("Optimizing for MAP... ")
        (optf,θ_map,ret) = optimize!(opt, zeros(p.n))
        println(ret,'.')
        J_map = zeros(p.n+p.m,p.n)
        residual!(θ_map, J_map, p)
        (Q,R) = qr(J_map,thin=true)
        return (θ_map, Q)
    end

    function rto_sample(p::Problem, start::AbstractVector, Q::AbstractMatrix, ξ::AbstractVector)
        opt = Opt(:LD_SLSQP,p.n)
        xtol_rel!(opt,1e-8)
        ftol_rel!(opt,1e-8)
        temp = zeros(p.n+p.m,p.n)
        Q_resid = (x,jac) -> Q_residual!(x,jac,p,Q,ξ,temp)
        min_objective!(opt, (x,grad) -> neglogpost!(x,grad,p.n,p.n,Q_resid))
        (optf,θ,ret) = optimize!(opt, start)
        if optf > 1e-8
            warn("Large residual ",optf)
        end
        J = zeros(p.n+p.m,p.n)
        r = residual!(θ, J, p)
        log_w =  - dot(r,r)/2 - logdet(Q'*J) + dot(Q'*r,Q'*r)/2
        return (θ,log_w)
    end

    function metropolize!(chain,log_ws)
        (n,nsamps) = size(chain);
        us = rand(n,nsamps)
        for i = 2:nsamps
            if us[i] > exp(log_ws[i]-log_ws[i-1])
                # reject
                chain[:,i] = chain[:,i-1]
                log_ws[i] = log_ws[i-1]
            end
        end
    end

    export rto_mcmc
    function rto_mcmc(p::Problem, nsamps::Integer)
        (θ_map,Q) = find_map(p)
        ξs = randn(p.n,nsamps)
        print("Sampling... ")
        (chain,log_ws) = @parallel (tuple_hcat) for i=1:nsamps
            rto_sample(p,θ_map,Q,ξs[:,i])
        end
        println("done.")
        print("Metropolizing... ")
        metropolize!(chain,log_ws)
        println("done.")
        return chain'
    end
end