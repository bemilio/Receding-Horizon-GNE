clear all
clc
close all
addpath(genpath('../')) % add all folders in the root folder
addpath(genpath(pwd)) % re-add the subfolders of this example

seed = 1;
rng(seed); 

n_x = 2; 
n_u = 1;
N = 2;

game = defineBasicGame();

[game.P_cl, game.K_cl, isInfHorStable_cl] = solveInfHorCL(game, 10000, 10^(-6));
[game.P_ol, game.K_ol, isInfHorStable_ol] = solveInfHorOL(game, 1000, 10^(-6));

disp( "Job complete" )



A_cl = game.A + sum(pagemtimes(game.B, game.K_cl), 3);
x1 = -2:.1:2;
x2 = -2:.1:2;
[X1, X2] = meshgrid(x1, x2);
X_dot = zeros(size(X1));
Y_dot = zeros(size(X2));
for i = 1:numel(X1)
    x = [X1(i); X2(i)];
    x_dot = A_cl * x - x;
    X_dot(i) = x_dot(1);
    Y_dot(i) = x_dot(2);
end

% quiver(X1, X2, X_dot, Y_dot);

