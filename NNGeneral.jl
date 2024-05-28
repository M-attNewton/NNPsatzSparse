# General Neural Network Attempt 1
using TSSOS
using DynamicPolynomials
using Random
using MAT
using JuMP, Ipopt
using BenchmarkTools
#using SDPT3

include("NNPsatzChorFuncs.jl")

# Specify dimensions
#dim_in = 1
#dim_hidden = [2 2 2 2]
#dim_out = 1

# Collate dimensions
#dims = hcat(dim_in, dim_hidden)
#dims = hcat(dims, dim_out)

# Create NN params
#rng = MersenneTwister(1)
#weights = zeros(maximum(dims), maximum(dims), size(dim_hidden,2) + 1)
#biases = zeros(maximum(dims), size(dim_hidden,2) + 1)
#for i in 1:(size(dim_hidden,2) + 1)
#    weights[1:dims[i+1], 1:dims[i],i] = (1/(dim_in)^0.5)*randn(rng, dims[i+1],dims[i])
#    biases[1:dims[i+1],i] = (1/(dim_in)^0.5)*randn(rng, dims[i+1],1)
#end

# Read NN from file
net = matopen("matlabNNSavesTRUE\\net2x5x12x2relu.mat")
biases = read(net,"biases")
weights = read(net,"weights")
dims = read(net,"dims")
dims = round.(Int,dims)
#AF = read(net,"AF")

#weights = weights*0.75
#biases = biases*0.75

AF = "tanh"

# Extract dimensions
#dim_in = round.(Int,dims[1])
#dim_hidden = round.(Int,dims[2:end-1])
#dim_out = round.(Int,dims[end])
dim_in = dims[1]
dim_hidden = transpose(dims[2:end-1])
dim_out = dims[end]

# Input bounds
#u_min = 0.5*ones(dim_in,1)
#u_max = 1.5*ones(dim_in,1)

#u_min = 1*ones(dim_in,1)
#u_max = 2*ones(dim_in,1)

u_min = 5*ones(dim_in,1)
u_max = 15*ones(dim_in,1)

#u_min = -15*ones(dim_in,1)
#u_max = 35*ones(dim_in,1)

# Number of edges of polytope, only when dim_out = 2
dim_poly = 8

# Order of relaxation, can be 0,1,2,etc. or "min" that uses minimum for each clique
order = 2#"min"

# Create varaibles for optimisation
@polyvar x[1:sum(dim_hidden)]
@polyvar u[1:dim_in]
@polyvar d
vars = [x;u;d]

# Input constraints
con_in1 = u - u_min
con_in2 = u_max - u

# Hidden layer constraints
#ineq_cons,eq_cons = hiddenLayerCons(u_min,u_max,u,x,dims,weights,biases)

ineq_cons = 1
eq_cons = 1

# IBP - get preprocessing values
Y_min,Y_max,X_min,X_max,out_min,out_max = intBoundProp(u_min,u_max,dims,dim_hidden,weights,biases,AF)

if AF == "relu"
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
elseif AF == "sigmoid"

    # Mid point of sector
    x_m = 1

    # Right side upper line L_ub
    m1 = Model(Ipopt.Optimizer)
    @variable(m1,d1)
    @NLconstraint(m1, (1/(1+exp(-d1)))*(1 - 1/(1+exp(-d1))) == (1/(1+exp(-x_m)) - 1/(1+exp(-d1)))/(x_m - d1))
    @constraint(m1, d1 <= 0.9)
    optimize!(m1)
    d1 = value(d1)
    grad_L_ub = (1/(1+exp(-d1)))*(1 - 1/(1+exp(-d1)))
    c_L_ub = (1/(1+exp(-d1))) - grad_L_ub*d1

    # Left side upper line L_lb
    m2 = Model(Ipopt.Optimizer)
    @variable(m2,d2)
    @NLconstraint(m2, (1/(1+exp(-d2)))*(1 - 1/(1+exp(-d2))) == (1/(1+exp(x_m)) - 1/(1+exp(-d2)))/(-x_m - d2))
    @constraint(m2, d2 >= -0.9)
    optimize!(m2)
    d2 = value(d2)
    grad_L_lb = (1/(1+exp(-d2)))*(1 - 1/(1+exp(-d2)))
    c_L_lb = (1/(1+exp(-d2))) - grad_L_lb*d2

    for j in 1:size(dim_hidden,2)
       if j == 1
           local x_curr_layer = x[1:dim_hidden[j]]
           local v = weights[1:dims[j+1], 1:dims[j],j]*u + biases[1:dims[j+1],j]
           local X_min_curr_layer = X_min[1:dim_hidden[j]]
           local X_max_curr_layer = X_max[1:dim_hidden[j]]
           local Y_min_curr_layer = Y_min[1:dim_hidden[j]]
           local Y_max_curr_layer = Y_max[1:dim_hidden[j]]
       else
           local x_prev_layer = x[sum(dim_hidden[1:j-2]) + 1 : sum(dim_hidden[1:j-1])]
           local x_curr_layer = x[sum(dim_hidden[1:j-1]) + 1 : sum(dim_hidden[1:j])]
           local v = weights[1:dims[j+1], 1:dims[j],j]*x_prev_layer + biases[1:dims[j+1],j]
           local X_min_curr_layer = X_min[sum(dim_hidden[1:j-1]) + 1 : sum(dim_hidden[1:j])]
           local X_max_curr_layer = X_max[sum(dim_hidden[1:j-1]) + 1 : sum(dim_hidden[1:j])]
           local Y_min_curr_layer = Y_min[sum(dim_hidden[1:j-1]) + 1 : sum(dim_hidden[1:j])]
           local Y_max_curr_layer = Y_max[sum(dim_hidden[1:j-1]) + 1 : sum(dim_hidden[1:j])]
       end
       for k in 1:dim_hidden[j]
           # Two sector constraints
           if Y_max_curr_layer[k] > 0 && Y_min_curr_layer[k] < 0
            # Sector in right hand plane
            if Y_max_curr_layer[k] > x_m
                local grad1a = (X_max_curr_layer[k] - (1/(1+exp(-x_m))))/(Y_max_curr_layer[k] - x_m)
                local c1a = X_max_curr_layer[k] - grad1a*Y_max_curr_layer[k];
                # Check for overlapping sectors
                if X_min_curr_layer[k] >  Y_min_curr_layer[k]*grad1a + c1a
                    local grad1a = (X_min_curr_layer[k] - (1/(1+exp(-x_m))))/(Y_min_curr_layer[k] - x_m)
                    local c1a = X_min_curr_layer[k] - grad1a*Y_min_curr_layer[k]
                end
            else
                local grad1a = 0;
                #c1a = X_max_curr_layer(k);
                local c1a = (1/(1+exp(-x_m)))
            end
            global ineq_cons = vcat(ineq_cons, (x_curr_layer[k] - (grad1a*v[k] + c1a))*((grad_L_ub*v[k] + c_L_ub) - x_curr_layer[k]) )

            # Sector in left hand plane
            if Y_min_curr_layer[k] < -x_m
                local grad2a = (X_min_curr_layer[k] - (1/(1+exp(x_m))) )/(Y_min_curr_layer[k] - -x_m)
                local c2a = X_min_curr_layer[k] - grad2a*Y_min_curr_layer[k];
                # Check for overlapping sectors
                if X_max_curr_layer[k] <  Y_max_curr_layer[k]*grad2a + c2a
                    local grad2a = (X_max_curr_layer[k] - (1/(1+exp(x_m))) )/(Y_max_curr_layer[k] - -x_m)
                    local c2a = X_max_curr_layer[k] - grad2a*Y_max_curr_layer[k]
                end
            else
                local grad2a = 0;
                #c2a = X_min_curr_layer(k);
                local c2a = (1/(1+exp(x_m)))
            end
            global ineq_cons = vcat(ineq_cons, (x_curr_layer[k] - (grad2a*v[k] + c2a))*((grad_L_lb*v[k] + c_L_lb) - x_curr_layer[k]) )

        elseif Y_max_curr_layer[k] < 0 && Y_min_curr_layer[k] < 0
            local Ysec = (Y_max_curr_layer[k] + Y_min_curr_layer[k])/2
            local Xsec = (1/(1+exp(-Ysec)))

            local grad1a = (X_max_curr_layer[k] - Xsec)/(Y_max_curr_layer[k] - Ysec)
            local c1a = X_max_curr_layer[k] - grad1a*Y_max_curr_layer[k]

            local grad2a = (X_min_curr_layer[k] - Xsec)/(Y_min_curr_layer[k] - Ysec)
            local c2a = X_min_curr_layer[k] - grad2a*Y_min_curr_layer[k]

            global ineq_cons = vcat(ineq_cons, (x_curr_layer[k] - (grad2a*v[k] + c2a))*((grad1a*v[k] + c1a) - x_curr_layer[k]) )

        elseif Y_max_curr_layer[k] > 0 && Y_min_curr_layer[k] > 0
            local Ysec = (Y_max_curr_layer[k] + Y_min_curr_layer[k])/2
            local Xsec = (1/(1+exp(-Ysec)))

            local grad1a = (X_max_curr_layer[k] - Xsec)/(Y_max_curr_layer[k] - Ysec)
            local c1a = X_max_curr_layer[k] - grad1a*Y_max_curr_layer[k]

            local grad2a = (X_min_curr_layer[k] - Xsec)/(Y_min_curr_layer[k] - Ysec)
            local c2a = X_min_curr_layer[k] - grad2a*Y_min_curr_layer[k]

            global ineq_cons = vcat(ineq_cons, (x_curr_layer[k] - (grad1a*v[k] + c1a))*((grad2a*v[k] + c2a) - x_curr_layer[k]) )

        end
        global ineq_cons = vcat(ineq_cons, -(x_curr_layer[k] - X_min[sum(dim_hidden[1:j-1]) + k])*(x_curr_layer[k] - X_max[sum(dim_hidden[1:j-1]) + k]))
        end
    end
elseif AF == "tanh"

    # Mid point of sector
    #x_m = 1.1
    x_m = 1.1

    # Right side upper line L_ub
    m1 = Model(Ipopt.Optimizer)
    @variable(m1,d1)
    @NLconstraint(m1, 1 - (tanh(d1))^2 == (tanh(x_m) - tanh(d1))/(x_m - d1))
    @constraint(m1, d1 <= x_m - 0.1)
    optimize!(m1)
    d1 = value(d1)
    grad_L_ub = 1 - (tanh(d1))^2
    c_L_ub = tanh(d1) - grad_L_ub*d1

    # Left side upper line L_lb
    m2 = Model(Ipopt.Optimizer)
    @variable(m2,d2)
    @NLconstraint(m2, 1 - (tanh(d2))^2 == (tanh(-x_m) - tanh(d2))/(-x_m - d2))
    @constraint(m2, d2 >= -x_m + 0.1)
    optimize!(m2)
    d2 = value(d2)
    grad_L_lb = 1 - (tanh(d2))^2
    c_L_lb = tanh(d2) - grad_L_lb*d2

    for j in 1:size(dim_hidden,2)
       if j == 1
           local x_curr_layer = x[1:dim_hidden[j]]
           local v = weights[1:dims[j+1], 1:dims[j],j]*u + biases[1:dims[j+1],j]
           local X_min_curr_layer = X_min[1:dim_hidden[j]]
           local X_max_curr_layer = X_max[1:dim_hidden[j]]
           local Y_min_curr_layer = Y_min[1:dim_hidden[j]]
           local Y_max_curr_layer = Y_max[1:dim_hidden[j]]
       else
           local x_prev_layer = x[sum(dim_hidden[1:j-2]) + 1 : sum(dim_hidden[1:j-1])]
           local x_curr_layer = x[sum(dim_hidden[1:j-1]) + 1 : sum(dim_hidden[1:j])]
           local v = weights[1:dims[j+1], 1:dims[j],j]*x_prev_layer + biases[1:dims[j+1],j]
           local X_min_curr_layer = X_min[sum(dim_hidden[1:j-1]) + 1 : sum(dim_hidden[1:j])]
           local X_max_curr_layer = X_max[sum(dim_hidden[1:j-1]) + 1 : sum(dim_hidden[1:j])]
           local Y_min_curr_layer = Y_min[sum(dim_hidden[1:j-1]) + 1 : sum(dim_hidden[1:j])]
           local Y_max_curr_layer = Y_max[sum(dim_hidden[1:j-1]) + 1 : sum(dim_hidden[1:j])]
       end
       for k in 1:dim_hidden[j]
           # Two sector constraints
           if Y_max_curr_layer[k] > 0 && Y_min_curr_layer[k] < 0
            # Sector in right hand plane
            if Y_max_curr_layer[k] > x_m
                local grad1a = (X_max_curr_layer[k] - tanh(x_m))/(Y_max_curr_layer[k] - x_m)
                local c1a = X_max_curr_layer[k] - grad1a*Y_max_curr_layer[k];
                # Check for overlapping sectors
                if X_min_curr_layer[k] >  Y_min_curr_layer[k]*grad1a + c1a
                    local grad1a = (X_min_curr_layer[k] - tanh(x_m))/(Y_min_curr_layer[k] - x_m)
                    local c1a = X_min_curr_layer[k] - grad1a*Y_min_curr_layer[k]
                end
            else
                local grad1a = 0;
                #c1a = X_max_curr_layer(k);
                local c1a = tanh(x_m)
            end
            global ineq_cons = vcat(ineq_cons, (x_curr_layer[k] - (grad1a*v[k] + c1a))*((grad_L_ub*v[k] + c_L_ub) - x_curr_layer[k]) )

            # Sector in left hand plane
            if Y_min_curr_layer[k] < -x_m
                local grad2a = (X_min_curr_layer[k] - tanh(-x_m) )/(Y_min_curr_layer[k] - -x_m)
                local c2a = X_min_curr_layer[k] - grad2a*Y_min_curr_layer[k];
                # Check for overlapping sectors
                if X_max_curr_layer[k] <  Y_max_curr_layer[k]*grad2a + c2a
                    local grad2a = (X_max_curr_layer[k] -  tanh(-x_m) )/(Y_max_curr_layer[k] - -x_m)
                    local c2a = X_max_curr_layer[k] - grad2a*Y_max_curr_layer[k]
                end
            else
                local grad2a = 0;
                #c2a = X_min_curr_layer(k);
                local c2a =  tanh(-x_m)
            end
            global ineq_cons = vcat(ineq_cons, (x_curr_layer[k] - (grad2a*v[k] + c2a))*((grad_L_lb*v[k] + c_L_lb) - x_curr_layer[k]) )

        elseif Y_max_curr_layer[k] < 0 && Y_min_curr_layer[k] < 0
            local Ysec = (Y_max_curr_layer[k] + Y_min_curr_layer[k])/2
            local Xsec = tanh(Ysec)

            local grad1a = (X_max_curr_layer[k] - Xsec)/(Y_max_curr_layer[k] - Ysec)
            local c1a = X_max_curr_layer[k] - grad1a*Y_max_curr_layer[k]

            local grad2a = (X_min_curr_layer[k] - Xsec)/(Y_min_curr_layer[k] - Ysec)
            local c2a = X_min_curr_layer[k] - grad2a*Y_min_curr_layer[k]

            global ineq_cons = vcat(ineq_cons, (x_curr_layer[k] - (grad2a*v[k] + c2a))*((grad1a*v[k] + c1a) - x_curr_layer[k]) )

        elseif Y_max_curr_layer[k] > 0 && Y_min_curr_layer[k] > 0
            local Ysec = (Y_max_curr_layer[k] + Y_min_curr_layer[k])/2
            local Xsec = tanh(Ysec)

            local grad1a = (X_max_curr_layer[k] - Xsec)/(Y_max_curr_layer[k] - Ysec)
            local c1a = X_max_curr_layer[k] - grad1a*Y_max_curr_layer[k]

            local grad2a = (X_min_curr_layer[k] - Xsec)/(Y_min_curr_layer[k] - Ysec)
            local c2a = X_min_curr_layer[k] - grad2a*Y_min_curr_layer[k]

            global ineq_cons = vcat(ineq_cons, (x_curr_layer[k] - (grad1a*v[k] + c1a))*((grad2a*v[k] + c2a) - x_curr_layer[k]) )

        end
        global ineq_cons = vcat(ineq_cons, -(x_curr_layer[k] - X_min[sum(dim_hidden[1:j-1]) + k])*(x_curr_layer[k] - X_max[sum(dim_hidden[1:j-1]) + k]))
        end
    end
end

ineq_cons = ineq_cons[2:end]
if size(eq_cons,1) >= 2
eq_cons = eq_cons[2:end]
end
# Output constraints
v_out = weights[1:dims[end],1:dims[end-1],end]*x[end - dim_hidden[end] + 1 : end] + biases[1:dims[end],end]

# Apply Psatz
if dim_out == 1
    global c = [-1 1]
    global opt = zeros(2,1)
    for i in 1:2
        local f = c[i]*d
        local g0 = c[i]*(d - v_out)
        local pop = vcat(f, g0)
        local pop = vcat(pop,con_in1)
        local pop = vcat(pop,con_in2)
        local pop = vcat(pop, ineq_cons)
        local pop = vcat(pop, eq_cons)
        if size(eq_cons,1) >= 2
            global opt[i],sol,data = cs_tssos_first(pop, vars, order, numeq=size(eq_cons,1), TS="block", solution=true, solver="Mosek")
            #global opt[i],sol,data = cs_tssos_first(pop, vars, order, numeq=size(eq_cons,1), TS="MD", solution=true, solver="Mosek")
            #global opt[i],sol,data = cs_tssos_first(pop, vars, order, numeq=size(eq_cons,1), TS="block", solution=true, solver="COSMO")
        else
            global opt[i],sol,data = cs_tssos_first(pop, vars, order, numeq=0, TS="block", solution=true)
        end
        global opt[1] = -opt[1]
    end
elseif dim_out == 2
    global opt = zeros(dim_poly,1)
    global C = zeros(dim_poly,2)
    for i in 1:dim_poly
        local theta = (i-1)/dim_poly*2*pi
        global C[i,:] = [cos(theta) sin(theta)]
        local f = -d
        #local g0 = C[i,:]*v_out - d
        local g0 = [cos(theta) sin(theta)]*v_out - d
        local pop = vcat(f, g0)
        local pop = vcat(pop,con_in1)
        local pop = vcat(pop,con_in2)
        #local pop = vcat(pop,con_in1.*con_in2)
        local pop = vcat(pop, ineq_cons)
        local pop = vcat(pop, eq_cons)
        if size(eq_cons,1) >= 2
            global opt[i],sol,data = cs_tssos_first(pop, vars, order, numeq=size(eq_cons,1), TS="block", solution=true)
        else
            global opt[i],sol,data = cs_tssos_first(pop, vars, order, numeq=0, TS="block", solution=true)
            #@btime cs_tssos_first(pop, vars, 2, numeq=0, TS="block", solution=true)
        end
        #opt[i],sol,data = cs_tssos_first(pop, vars, 2, numeq=size(eq_cons,1), TS="block", solution=true)
    end
    global opt = -opt
    X_SOS,Y_SOS = solvePolytope(opt,dim_poly,C)
end


# Lower bound
#f = d - v_out
#f = d
#g0 = d - v_out

# Solve CSTSSOS
#pop = vcat(f, g0)
#pop = vcat(pop,con_in1)
#pop = vcat(pop,con_in2)
#pop = vcat(pop, ineq_cons)
#pop = vcat(pop, eq_cons)
#LB_opt,sol,data = cs_tssos_first(pop, vars, 2, numeq=size(eq_cons,1), TS="block", solution=true)

# Upper bound
#f = -d
#g0 = v_out - d

#pop = vcat(f, g0)
#pop = vcat(pop,con_in1)
#pop = vcat(pop,con_in2)
#pop = vcat(pop, ineq_cons)
#pop = vcat(pop, eq_cons)
#UB_opt,sol,data = cs_tssos_first(pop, vars, 2, numeq=size(eq_cons,1), TS="block", solution=true)

#UB_opt = -UB_opt

#print(LB_opt)
#print(UB_opt)

#if 1 == 2 && 2 == 2
#    test1 = 2
#end
