% This script defines an OL-NE whose finite-horizon problem is a 
% non-monotone VI for an horizon T>4


clear all
clc
close all
addpath(genpath('../../')) % add all folders in the root folder
rmpath(genpath('../')) % remove folders of the other examples (function names are conflicting)
addpath(genpath(pwd)) % re-add the subfolders of this example

seed = 1;
rng(seed); 

n_x = 2; 
n_u = 1;
N = 2;
T = 5;

game = defineGame();

[game.C_x, game.d_x, game.C_u_loc, game.d_u_loc, game.C_x_mix, ...
    game.C_u_mix, game.d_mix] = defineConstraints(N, n_x, n_u);
[game.C_u_sh, game.d_u_sh] = defineDummySharedInputConstraints(n_u, N);

x_0=[1; 0];

%% Test monotonicity of OL-NE
[game.P_ol, game.K_cl, isInfHorStable_cl] = solveInfHorOL(game, 10000, 10^(-6));
VIgen = computeVIGenerator(game,T);
% F(x_0) = M*u + g 
[~, ~, ~, ~, ~, ~, ~, M, g] = VIgen(x_0);

if min(eig(M + M')) <= -eps
    warning("The OL-NE VI is not monotone: the minimum eigenvalue is %.2d for the mapping matrix\n", min(eig(M + M')))
else
    fprintf("The OL-NE VI is monotone with min eigenvalue %.2d for the mapping matrix\n", min(eig(M + M')))
end

disp( "Job complete" )
