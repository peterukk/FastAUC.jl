# Function for fast computation of area under ROC curve
# Adopted from https://www.ibm.com/developerworks/community/blogs/jfp/entry/Fast_Computation_of_AUC_ROC_score?lang=en
function auc_ROC(y_true,y_pred)
    y_true = y_true[faster_sortperm_radix(y_pred)]
    nfalse = 0
    auc = 0
    n = length(y_true)
    for i in range(1,stop=n)
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    end
    auc = auc / (nfalse * (n - nfalse))
    return auc
end

# Function for fast computation of area under Precision-Recall curve
# adopted from https://tamasnagy.com/blog/precision-recall-julia/
function auc_PR(y_true,y_pred::Vector{F} ) where F<:AbstractFloat
    num_y_pred = length(y_pred) + 1
    ordering = reverse(faster_sortperm_radix(y_pred))
    y_true = y_true[ordering]

    num_pos = sum(y_true) #count(i->(i==0), y_true)
    num_neg = num_y_pred - num_pos - 1

    tn, fn, tp, fp = 0, 0, num_pos, num_neg

    p = Vector{F}(undef,num_y_pred)
    r = Vector{F}(undef,num_y_pred)

    p[num_y_pred] = tp/(tp+fp)
    r[num_y_pred] = tp/(tp+fn)
    auprc, prev_r = 0.0, r[num_y_pred]
    pmin, pmax = p[num_y_pred], p[num_y_pred]

    # traverse y_pred from lowest to highest
    for i in num_y_pred-1:-1:1
        dtn = y_true[i]==1 ? 0 : 1
        tn += dtn
        fn += 1-dtn
        tp = num_pos - fn
        fp = num_neg - tn
        p[i] = (tp+fp) == 0 ? 1-dtn : tp/(tp+fp)
        r[i] = tp/(tp+fn)

        # update max precision observed for current recall value
        if r[i] == prev_r
            pmax = p[i]
        else
            pmin = p[i] # min precision is always at recall switch
            auprc += (pmin + pmax)/2*(prev_r - r[i])
            prev_r = r[i]
            pmax = p[i]
        end
    end
    auprc
end


function faster_sortperm(v)
    ai = [Pair(i, a) for (i,a) in enumerate(v)]
    sort!(ai, by=x->x.second)
    [a.first for a in ai]
end

function faster_sortperm_radix(v)
    ai = [Pair(i, a) for (i,a) in enumerate(v)]
    sort!(ai, by=x->x.second, alg=RadixSort)
    [a.first for a in ai]
end