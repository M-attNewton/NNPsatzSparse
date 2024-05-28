# Compares the solve time in Julia by parsing the matlab data from the SOSTOOLS construction
using JuMP
import LinearAlgebra
#using SDPT3
using SeDuMi
#import SCS
using MAT
using MATLAB

# Read NN from file
test = matopen("matlabNNPDSaves\\test.mat")
A = read(test,"A")
b = read(test,"b")
c = read(test,"c")
K = read(test,"K")
#K = round.(Int,dims)
pars = read(test,"pars")

model  = Model(SeDuMi.Optimizer)
@variable(model, X[i=1:size(A,1)])
@objective(model, Min, sum(transpose(c)*X))
@constraint(model, transpose(A)*X .== vec(b)) #LinearAlgebra.diagm(X)
optimize!(model)
objective_value(model)

#model  = Model(SeDuMi.Optimizer)
#@variable(model, X[i=1:size(A,2), j=1:size(b,1)], PSD)
#@objective(model, Min, LinearAlgebra.tr(transpose(c)*X))
#@constraint(model, A*X = b)
#@constraint(model, LinearAlgebra.tr(A*X) = b)
#optimize!(model)
#objective_value(model)
