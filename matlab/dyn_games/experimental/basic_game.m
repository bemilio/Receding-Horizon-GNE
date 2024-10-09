clear all
clc
close all
addpath(genpath('../../')) % add all folders in the root folder
rmpath(genpath('../../examples')) % remove folders of the other examples (function names are conflicting)
addpath(genpath(pwd)) % re-add the subfolders of this example

game = defineLQR();

% [game.P_sdp, game.K_sdp, isInfHorStable_sdp] = solveInfHor_conic_opt(game, 10000, 10^(-6));


game = defineBasicGame();

[P_CL, K_ric, ~] = solveInfHorCL_cont_time(game, 10000, 10^(-6)); 
[P_OL, K_ric_OL, ~] = solveInfHorOL_cont_time(game, 10000, 10^(-6)); 

%[~, K_sdp, ~] = solveInfHor_conic_opt(game, 10000, 10^(-6));
[~, K_sdp, ~] = solveInfHor_conic_opt_2nd_ver(game, 10000, 10^(-6), K_ric);


