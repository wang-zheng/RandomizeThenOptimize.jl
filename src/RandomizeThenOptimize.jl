module RandomizeThenOptimize
	using NLopt

    # ===== TYPE: Problem =====
    # contains everything we need to describe inference problem
    export Problem, EmptyModel
    type Problem
        n::Integer # dimension of parameters
        m::Integer # dimension of data
        f::Function # forward model
        θ_pr::AbstractVector # prior mean
        y::AbstractVector # observation data
        L_pr::AbstractMatrix # L'*L = Covariance^(-1), for prior
        L_obs::AbstractMatrix # L'*L = Covariance^(-1), for observational noise
        opt::NLopt.Opt # NLopt optimization settings
        guess::AbstractVector # guess for parameter MAP
        verbose::Bool # whether to output progress
    end

    function Problem(n::Integer,m::Integer)
        opt = Opt(:LD_SLSQP,n)
        xtol_rel!(opt,1e-8)
        ftol_rel!(opt,1e-8)
        Problem(n,m,(x,jac)->EmptyModel(x,jac,n,m),zeros(Float64,n),zeros(Float64,m),eye(Float64,n),eye(Float64,m),opt,zeros(Float64,n),false)
    end

    function EmptyModel(x::AbstractVector,jac::AbstractMatrix,n::Integer,m::Integer)
        if length(jac) > 0
            jac[:] = zeros(Float64,m,n)
        end
        return zeros(Float64,m)
    end

    import Base.show
    export show
    function show(io::IO,p::Problem)
        print(io,"Problem(",p.n,",",p.m,")")
    end

    # ===== Functions to change Problem =====
    export forward_model!, nlopt_opt!, pr_mean!, pr_covariance!, pr_σ!, pr_precision!, obs_data!, obs_covariance!, obs_σ!, obs_precision!, verbose!, guess!
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

    function guess!(p::Problem, x::AbstractVector)
        p.guess = x
    end

    function verbose!(p::Problem, v::Bool)
        p.verbose = v
    end

    # ===== Utilities =====
    # hcat for each element of the tuple
    function tuple_hcat(a::Tuple, b::Tuple)
        l = length(a)
        c = [ hcat(a[i],b[i]) for i = 1:l ]
        return (c...)
    end

    # functions used in rto_mcmc
    function residual!(x::AbstractVector, jac::AbstractMatrix, p::Problem)
        
        if length(jac) > 0
            jacf = Array(Float64,p.m,p.n)
            #jacf = view(jac,1:p.m,1:p.n)
            fx = p.f(x,jacf)
            jac[:] = [p.L_pr; p.L_obs*jacf]
        else 
            fx = p.f(x,Array(Float64,0,0))
        end
        
        r = [p.L_pr*(x - p.θ_pr); p.L_obs*(fx - p.y)]
        
        return r
    end

    function Q_residual!(x::AbstractVector, jac::AbstractMatrix, p::Problem, Q::AbstractMatrix, ξ::AbstractVector, temp::AbstractMatrix)
        
        if length(jac)>0
            Qr = Q'*residual!(x,temp,p) - ξ;
            jac[:] = Q'*temp
        else
            Qr = Q'*residual!(x,Array(Float64,0,0),p) - ξ;
        end
        
        return Qr
    end

    function neglogpost!(x::AbstractVector, grad::AbstractVector, n::Integer, m::Integer, resid::Function)
        if length(grad) > 0
            jac = Array(Float64,m,n)
        else
            jac = Array(Float64,0,0)
        end
        
        r = resid(x,jac)
        
        if length(grad) > 0
            g = jac'*r
            grad[:] = g
        end
        
        return dot(r,r)/2
    end

    # used to interface with Mamba
    export logpostgrad
    function logpostgrad(x::AbstractVector, p::Problem)
        ngrad = zeros(x)
        resid = (x,jac) -> residual!(x,jac,p)
        nlogf = neglogpost!(x,ngrad,p.n,p.n+p.m,resid)
        return -nlogf, -ngrad
    end

    # ===== Main functions =====
    function find_map(p::Problem)
        opt = copy(p.opt)
        resid = (x,jac) -> residual!(x,jac,p)
        min_objective!(opt, (x,g) -> neglogpost!(x,g,p.n,p.n+p.m,resid) )
        if p.verbose 
            print("Optimizing for MAP... ")
        end
        (optf,θ_map,ret) = optimize!(opt, p.guess)
        if p.verbose
            println(ret,'.')
        end
        J_map = Array(Float64,p.n+p.m,p.n)
        residual!(θ_map, J_map, p)
        (Q,R) = qr(J_map,thin=true)
        return (θ_map, Q)
    end

    function rto_sample(p::Problem, start::AbstractVector, Q::AbstractMatrix, ξ::AbstractVector)
        opt = Opt(:LD_SLSQP,p.n)
        xtol_rel!(opt,1e-8)
        ftol_rel!(opt,1e-8)
        temp = Array(Float64,p.n+p.m,p.n)
        Q_resid = (x,jac) -> Q_residual!(x,jac,p,Q,ξ,temp)
        min_objective!(opt, (x,grad) -> neglogpost!(x,grad,p.n,p.n,Q_resid))
        (optf,θ,ret) = optimize!(opt, start)
        if optf > 1e-8
            warn("Large residual ",optf)
        end
        J = Array(Float64,p.n+p.m,p.n)
        r = residual!(θ, J, p)
        log_w =  - dot(r,r)/2 - logabsdet(Q'*J)[1] + dot(Q'*r,Q'*r)/2
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

    # to sample
    export rto_mcmc
    function rto_mcmc(p::Problem, nsamps::Integer)
        (θ_map,Q) = find_map(p)
        ξs = randn(p.n,nsamps)
        if p.verbose
            print("Sampling... ")
        end

        (chain,log_ws) = @parallel (tuple_hcat) for i=1:nsamps
            rto_sample(p,θ_map,Q,ξs[:,i])
        end

        #chain = Array(Float64,p.n,nsamps)
        #log_ws = Array(Float64,nsamps)

        #for i=1:nsamps
        #    (chaini, log_wsi) = rto_sample(p,θ_map,Q,ξs[:,i])
        #    chain[:,i] = chaini
        #    log_ws[i] = log_wsi
        #end
        
        if p.verbose
            println("done.")
            print("Metropolizing... ")
        end
        metropolize!(chain,log_ws)
        if p.verbose
            println("done.")
        end
        return chain'
    end
end
