%% test active-set solver
clear all
clc 
global border_left border_right;
border_left = -5;
border_right = 5;
global half_l half_w;
half_w = 0.75;
half_l = 1.5;

global T;
dt = 0.1;
T = 30;
N = 2;
n = 8;
all_m = 4;
m{1} = 2;
m{2} = 2;


params.use_ws = false;
params.use_initialization = false;
params.linear = true;
params.open_loop = false;
params.debug_plot = false;
params.contact = false;

Ai = [1, 0, dt, 0;
     0  1, 0,  dt;
     0, 0,  1,  0;
     0, 0,  0,  1];

Bi = [0 0; 0 0; dt 0; 0 dt];

A = blkdiag(Ai,Ai);
B = blkdiag(Bi,Bi);
c = zeros(n,1); 
FF = [c A B];

Qa = zeros(1+n+all_m,1+n+all_m);
Qa(1+n+1:end,1+n+1:end) = diag([1,1,1,1]);
% PA is polite
Qb = zeros(1+n+all_m,1+n+all_m);
Qb(1+n+3:1+n+4,1+n+3:1+n+4) = eye(2);
% PB is selfish but has to avoid

Ga = zeros(0,1+n);
% Gb = [-4 0 1 0 0 0 -1 0 0];
Gb = [-4 0 1 0 0 0 -1 0 0];

for t = 1:T
    Q{t,1} = Qa;
    Q{t,2} = Qb;
    G{t,1} = zeros(0,1+n+2);
    G{t,2} = [Gb, 0, 0];
    H{t,1} = zeros(0,1+n+2);
    H{t,2} = zeros(0,1+n+2);
    F{t} = FF;
    working_set{t,1} = zeros(size(Ga,1),1);
    working_set{t,2} = zeros(size(Gb,1),1);
end
Q{T+1,1} = zeros(1+n);
Q{T+1,2} = zeros(1+n);
G{T+1,1} = Ga;
G{T+1,2} = Gb;
H{T+1,1} = [2.5 1 0 0 0 0 0 0 0];
%H{T+1,1} = zeros(0,n+1);
H{T+1,2} = [2.5 0 0 0 0 1 0 0 0];
%H{T+1,2} = zeros(0,n+1);
working_set{T+1,1} = zeros(size(Ga,1),1);
working_set{T+1,2} = zeros(size(Gb,1),1);

x0 = [2.5; 0; 0; 4; -2.5; -5; 0; 5]; 


[x, u] = active_set_lq_game_solver(F,H,G,Q,N,T,m,x0,working_set, false, params);

%%
plot_solution(x);
%%
function plot_solution(x)
    global T;
    close all;
    figure; hold on;
    draw_lanes();
    axis([-20,20,-20,20]);
    axis('equal');
    for t = 1:T+1
        car_a = draw_car(x{t}(1:2),'b');
        car_b = draw_car(x{t}(5:6),'r');
        axis([-20,20,x{t}(2)-20,x{t}(2)+20]);
        axis('equal');
        pause(0.1);
        delete(car_a);
        delete(car_b);
    end
end

function pt = draw_car(center,color)
    global half_l half_w;
    pt = patch('XData',[center(1)-half_w,center(1)-half_w,center(1)+half_w,center(1)+half_w,center(1)-half_w],...
               'YData',[center(2)-half_l,center(2)+half_l,center(2)+half_l,center(2)-half_l,center(2)-half_l],...
               'FaceAlpha',1,'FaceColor', color);
end

function draw_lanes()
    global border_left border_right;
    plot([border_left, border_left],[-200,200],'-k');
    plot([border_left/2, border_left/2],[-200,200],'--k');
    plot([0, 0],[-200,200],'-k');
    plot([border_right/2, border_right/2],[-200,200],'--k');
    plot([border_right, border_right],[-200,200],'-k');
end

