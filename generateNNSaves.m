%% Generate NN parameters
% rng('default');
% rng(1,'twister');

% Input dimension. Value can be 1, 2 or 3
dim_in = 1; 

% Ouput dimension. Calue can be 1 or 2
dim_out = 1;

for a = 101:200 %[1,10,20,50,100,200,500,1000,2000,5000] %[1,2,5,7,10,15,20,25,30]
a

rng('default');
rng(1,'twister');

% Hidden layer dimensions
dim_hidden = [8*ones(1,a)];

% Activation function type. Can be relu, sigmoid or tanh
AF = 'relu'; 

% Create NN parameters
dims = [dim_in, dim_hidden, dim_out];
net = NNsetup(dims,AF);
mat2JuliaNet(net) % save the net to be used in other software
end