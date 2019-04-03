# FastAUC.jl

Computationally fast implementation of calculating area under ROC and PR curves

Requires: SortingAlgorithms

```
julia> typeof(y_pred)
Array{Float64,1}

julia> typeof(y_obs)
Array{Int32,1}

julia> size(y_obs)
(6199200,)

julia> @btime auroc =  auc_ROC(y_obs,y_pred);
  477.419 ms (27 allocations: 260.32 MiB)

julia> @btime aupr =  auc_PR(y_obs,y_pred);
  577.486 ms (33 allocations: 402.21 MiB)
  
```
