% Create and solve a test VI

clear all
clc
close all
addpath(genpath(pwd))

diary

seed = 1;
rng(seed); 

n_x = 1;
N = 2;

N_tests = 20;
Q = zeros(1,2,2);
Q(:,:,1) = [0, 1];
Q(:,:,2) = [-1, 0];
rep = @(y) repmat(reshape(y, [n_x * N,1] ), 1,1,N); % takes a 3D vector and stacks it in a 2D vector, then repeats this vector N times along the 3rd axis
VI.F = @(x) pagemtimes(Q,rep(x));
VI.J = @(x) zeros(1,1,2); % dummy
VI.A_sh = ones(1,n_x,2);
VI.b_sh = zeros(1,1,2);
VI.A_loc = zeros(2*n_x,n_x,2); % box
VI.b_loc = ones(2*n_x,1,2);

VI.A_loc(:,:,1) = [eye(n_x); -eye(n_x)];
VI.A_loc(:,:,2) = [eye(n_x); -eye(n_x)];
VI.b_loc(:,:,1) = [0.2; .4];
VI.b_loc(:,:,2) = [3; .2];


VI.n_x = n_x;
VI.N = N;

for test = 1:N_tests
    disp( "TEST: " + num2str(test) )
    x_0 = rand(1,1,2); 
    n_sh_constraints = size(VI.A_sh,1);
    d_0 = zeros(n_sh_constraints, 1);
    [x, d] = solveVICentrFB(VI, 10^3, 10^(-4), 0.1, 0.1, x_0, d_0);
    test_passed(test) = (x(:,:,1) + x(:,:,2) < 10^(-4) && x(:,:,1)>=0 && x(:,:,2)<=0 );
    disp("Test " + num2str(test) + " complete!")
%    workers_complete = workers_complete + 1;
%    disp("Workers complete: " + num2str(workers_complete) + " of " + num2str(N_tests))
end

if all(test_passed)
    disp("Test passed")
else
    disp("Test NOT passed")
end


% END script
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


