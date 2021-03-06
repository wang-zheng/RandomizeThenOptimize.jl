# The `RandomizeThenOptimize` module for Julia

This module implements the Randomize-then-Optimize (RTO) algorithm for sampling Bayesian posterior distributions. 

Please see [this paper](http://epubs.siam.org/doi/abs/10.1137/140964023) for a detailed explanation of the algorithm.

## Installation

Within Julia, you can install the package by typing

```julia
Pkg.clone("https://github.com/wang-zheng/RandomizeThenOptimize.jl","RandomizeThenOptimize")
```

## Tutorial

The following example generates samples from the Bayesian posterior distribution given a user-specified forward model. The prior and observational noise are standard normal distributions (by default).

```julia
using RandomizeThenOptimize

n = 2 # size of parameters
m = 1 # size of data

function myf!(x::Vector, jac::Matrix)
    a = 3 
    b = 6 
    c = 20
    if length(jac) > 0
        jac[:] = c.*[cos(a*x[1]) * a/b; -1.0]' # Jacobian must be a 1 by 2 matrix
    end
    
    return c*(sin(a*x[1])/b - x[2])
end

prob = Problem(n, m)
forward_model!(prob, myf!)
chain = rto_mcmc(prob, 1000) # draw 1000 samples
```

After running the code, `chain` should be a 1000 by 2 matrix whose rows are raw samples from the posterior distribution.

### Parallel

To run the code in parallel, `RandomizeThenOptimize` must be imported by all processors and the forward model must be defined on all processors.

```julia
addprocs(4)
@everywhere using RandomizeThenOptimize

n = 2 # size of parameters
m = 1 # size of data

@everywhere function myf!(x::Vector, jac::Matrix)
    a = 3 
    b = 6 
    c = 20
    if length(jac) > 0
        jac[:] = c.*[cos(a*x[1]) * a/b; -1.0]' # Jacobian must be a 1 by 2 matrix
    end
    
    return c*(sin(a*x[1])/b - x[2])
end

prob = Problem(n, m)
forward_model!(prob, myf!)
chain = rto_mcmc(prob, 1000) # draw 1000 samples
```

## Reference

Coming soon!
