clear all
clc
close all
addpath(genpath(pwd))

n_x = 4;
n_u = 2;
N = 3;
T = 5;
T_sim = 20;

sparse_density=0.6;
posdef=0.1;
game = generateRandomGame(n_x, n_u, N, sparse_density, posdef);
[P, K, isInfHorStable] = solveInfHorCL(game, 1000, 10^(-6));

game.min_x = -1 * ones(n_x,1);
game.max_x = 1 * ones(n_x,1);
game.min_u = -1 * ones(n_u,1);
game.max_u = 1 * ones(n_u,1);

%setBoxStateConstraints(game, min_x, max_x);
%setBoxInputConstraints(game, min_u, max_u);

game.P = P;

for t=1:T_sim
    u = solveFinHorCL(game, T);
    x(:,t+1) = game.A * x(:,t) + sum(pagemtimes(B, u), 1);
end