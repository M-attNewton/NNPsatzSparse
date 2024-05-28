# Functions for NNPsatzChor

function intBoundProp(u_min,u_max,dims,dim_hidden,weights,biases,AF)
    z_min = u_min
    z_max = u_max
    y_min = z_min
    y_max = z_max
    z_mintemp = z_min
    z_maxtemp = z_max
    #mu1 = zeros(size(dim_hidden,2)+1,1)
    #r1 = zeros(size(dim_hidden,2)+1,1)
    for k in 2:(size(dim_hidden,2)+1)
        mu1 = (z_maxtemp + z_mintemp)/2
        r1 = (z_maxtemp - z_mintemp)/2
        mu2 = weights[1:dims[k], 1:dims[k-1],k-1]*mu1 + biases[1:dims[k],k-1]
        r2 = abs.(weights[1:dims[k], 1:dims[k-1],k-1])*r1
        z_mintemp = mu2 - r2
        z_maxtemp = mu2 + r2
        y_min = vcat(y_min,mu2 - r2)
        y_max = vcat(y_max,mu2 + r2)
        # just do relu for now but add others later
        if AF == "relu"
            for j in 1:size(z_mintemp,1)
                if z_mintemp[j] <= 0
                    z_mintemp[j] = 0
                end
                if z_maxtemp[j] <= 0
                    z_maxtemp[j] = 0
                end
            end
        elseif AF == "sigmoid"
            for j in 1:size(z_mintemp,1)
                z_mintemp[j] = 1/(1+exp(-z_mintemp[j]))
                z_maxtemp[j] = 1/(1+exp(-z_maxtemp[j]))
            end
        elseif AF == "tanh"
            for j in 1:size(z_mintemp,1)
                z_mintemp[j] = tanh(z_mintemp[j])
                z_maxtemp[j] = tanh(z_maxtemp[j])
            end
        end
        z_min = vcat(z_min,z_mintemp)
        z_max = vcat(z_max,z_maxtemp)
    end
    k = size(dim_hidden,2) + 2
    mu1 = (z_maxtemp + z_mintemp)/2
    r1 = (z_maxtemp + z_mintemp)/2
    mu2 = weights[1:dims[k], 1:dims[k-1],k-1]*mu1 + biases[1:dims[k],k-1]
    r2 = abs.(weights[1:dims[k], 1:dims[k-1],k-1]*r1)
    out_min = mu2 - r2
    out_max = mu2 + r2
    X_min = z_min[dims[1]+1:end]
    X_max = z_max[dims[1]+1:end]
    Y_min = y_min[dims[1]+1:end]
    Y_max = y_max[dims[1]+1:end]
    return Y_min, Y_max, X_min, X_max, out_min, out_max
end

function solvePolytope(bound,dim_poly,C)
    tmp = C[1:2,:] \ bound[1:2]
    X = tmp[1]
    Y = tmp[2]
    for i in 2:(dim_poly-1)
        tmp = C[i:(i+1),:] \ bound[i:(i+1)]
        X = vcat(X,tmp[1])
        Y = vcat(Y,tmp[2])
    end
    tmp = ([C[1,:]'; C[end,:]']) \ ([bound[1]; bound[end]])
    #tmp = C[[1,dim_poly],:] \ B[[1 dim_poly]]
    X = vcat(X,tmp[1])
    Y = vcat(Y,tmp[2])
    return X, Y
end



# WORK THIS OUT LATER
function hiddenLayerCons(u_min,u_max,u,x,dims,weights,biases)

    #include("NNPsatzChorFuncs.jl")

    # Extract dimensions
    dim_in = dims[1]
    dim_hidden = dims[2:end-1]
    dim_out = dims[end]

    # Hidden layer constraints
    global ineq_cons = 1
    global eq_cons = 1

    # IBP - get preprocessing values
    Y_min,Y_max,X_min,X_max,out_min,out_max = intBoundProp(u_min,u_max,dims,dim_hidden,weights,biases)
    Ip = findall(Y_min -> Y_min >= 0.001,Y_min)
    In = findall(Y_max -> Y_max <= -0.001,Y_max)
    # don't actually need to compute Ipn I don't think
    Ip2 = zeros(size(Ip,1))
    for j in 1:size(Ip,1)
        Ip2[j] = round.(Int,Ip[j][1])
    end
    In2 = zeros(size(In,1))
    for j in 1:size(In,1)
        In2[j] = round.(Int,In[j][1])
    end

    for j in 1:size(dim_hidden,2)
        if j == 1
            local x_curr_layer = x[1:dim_hidden[j]]
            local v = weights[1:dims[j+1], 1:dims[j],j]*u + biases[1:dims[j+1],j]
        else
            local x_prev_layer = x[sum(dim_hidden[1:j-2]) + 1 : sum(dim_hidden[1:j-1])]
            local x_curr_layer = x[sum(dim_hidden[1:j-1]) + 1 : sum(dim_hidden[1:j])]
            local v = weights[1:dims[j+1], 1:dims[j],j]*x_prev_layer + biases[1:dims[j+1],j]
        end
        for k in 1:dim_hidden[j]
            local node_num = sum(dim_hidden[1:j-1]) + k
            if in(node_num).(Ip) == 1
                global ineq_cons = vcat(ineq_cons, x_curr_layer[k])
                global eq_cons = vcat(eq_cons, x_curr_layer[k] - v[k])
                global eq_cons = vcat(eq_cons, x_curr_layer[k]*(x_curr_layer[k] - v[k]))
            elseif in(node_num).(In) == 1
                global eq_cons = vcat(eq_cons, x_curr_layer[k])
                global ineq_cons = vcat(ineq_cons, x_curr_layer[k] - v[k])
                global eq_cons = vcat(eq_cons, x_curr_layer[k]*(x_curr_layer[k] - v[k]))
            else
                global ineq_cons = vcat(ineq_cons, x_curr_layer[k])
                global ineq_cons = vcat(ineq_cons, x_curr_layer[k] - v[k])
                global eq_cons = vcat(eq_cons, x_curr_layer[k]*(x_curr_layer[k] - v[k]))
            end
            global ineq_cons = vcat(ineq_cons, -(x_curr_layer[k] - X_min[sum(dim_hidden[1:j-1]) + k])*(x_curr_layer[k] - X_max[sum(dim_hidden[1:j-1]) + k]))
        end
    end
    ineq_cons = ineq_cons[2:end]
    eq_cons = eq_cons[2:end]

    return ineq_cons, eq_cons
end




#mu = zeros(size(dim_hidden,2)+1,1)
#r = zeros(size(dim_hidden,2)+1,1)
#for k = 2:(size(dim_hidden,2)+1)
#    mu[k-1] = (z_max[k-1] + z_min[k-1])/2
#    r[k-1] = (z_max[k-1] - z_min[k-1])/2
#    mu[k] = weights[1:dims[k], 1:dims[k-1],k-1]*mu[k-1] + biases[1:dims[k],k-1]
#    r[k] = abs(weights[1:dims[k], 1:dims[k-1],k-1]*r[k-1])
#    z_min[k] = mu[k] - r[k]
#    z_max[k] = mu[k] + r[k]
#    y_min[k] = z_min[k]
#    y_max[k] = z_max[k]
    # just do relu for now but add other later
#    z_min[k] = maximum(0,z_min[k])
#    z_max[k] = maximum(0,z_max[k])
#end
