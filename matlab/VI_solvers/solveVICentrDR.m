function [x, r, solved] = solveVICentrDR(VI, ...
                              n_iter, ...
                              eps_err, ...
                              stepsize, ...
                              H, ...
                              x)
if ~exist ('n_iter', 'var')
    n_iter = 10^(6);
end

if ~exist ('eps_err', 'var')
    eps_err = 10^(-5);
end

if ~exist ('stepsize', 'var')
    stepsize = .5;
end

if ~exist ('x', 'var')
    x = zeros(VI.n_x, 1, VI.N);
end

if ~exist('H', 'var')
    H = eye(VI.N * VI.n_x);
end

% initialize container for residual
r = zeros(n_iter,1);

% Collect constraints in a single matrix and vector
n_loc_constr = size(VI.A_loc, 1);
A_loc_all = zeros(VI.N * n_loc_constr, VI.N * VI.n_x);
b_loc_all = zeros(VI.N * n_loc_constr, 1);
for i=1:VI.N
    A_loc_all((i-1) * n_loc_constr + 1 : i * n_loc_constr, (i-1) * VI.n_x + 1 : i * VI.n_x ) = VI.A_loc(:,:,i);
    b_loc_all((i-1) * n_loc_constr + 1: i * n_loc_constr ) = VI.b_loc(:,:,i);
end
n_sh_constr = size(VI.A_sh, 1);
A_sh_all = zeros(n_sh_constr, VI.N * VI.n_x);
b_sh_all = zeros(n_sh_constr, 1);
for i=1:VI.N
    A_sh_all(:, (i-1) * VI.n_x + 1 : i * VI.n_x ) = VI.A_sh(:,:,i);
    b_sh_all = b_sh_all + VI.b_sh(:,:,i);
end
A_all = [A_loc_all; A_sh_all];
b_all = [b_loc_all; b_sh_all];

% Define symmetric and non-symmetric parts of the VI
Q = VI.Q;
M_1 = (Q - Q')/2;
M_2 = (Q + Q')/2;

% Define relevant inverse matrix for the D-R algorihm
G_inv = eye(size(Q,1))/(H + M_1);
solved = false;
for k =1:n_iter
    % Compute residual
    r(k) = compute_residual(x,VI.F, A_all, b_all, eps_err);

    % Perform Douglas-Rachford step
    x = run_DR_once(x, M_1, M_2, G_inv, VI.q, H, A_all, b_all, stepsize, eps_err);
    
    % Display residual
    if mod(k,300)==0
        disp("Residual: " + num2str(r(k)));
    end
    if r(k) < eps_err
        solved = true;
        break
    end
end

end

function [x_new] = run_DR_once(x, M_1, M_2, G_inv, q,  H, A_all, b_all, stepsz, eps_err)
    
    N = size(x, 3);
    n_x = size(x,1);

    % Define quadratic program for first step of the algorithm
    proj = osqp;
    proj.setup(H + M_2, q + (M_1-H) * x(:), A_all, -Inf*ones(size(A_all,1),1), b_all, 'verbose', false, 'eps_abs', eps_err/1000,'eps_rel', eps_err/1000 )

    % options = optimoptions('quadprog','Display','off', 'Algorithm', 'active-set');
    % proj_2 = @(y) quadprog(H + M_2, q + (M_1-H) * y, ...
    %     A_all, b_all, [], [], [],[], y, options); 

    % Actual algorithm
    y_new = proj.solve.x;
    x_new = G_inv * (H * (2*stepsz*y_new + (1-2*stepsz) * x(:)) + M_1 * x(:));  

    % Reshape such that the 3rd dimension of the array indexes the agents
    x_new= reshape(x_new,[n_x, 1, N]);

end

function r = compute_residual(x, F, A_all, b_all, eps_err)
    n_x = size(x,1);
    N = size(x,3);
    
    % Forward step
    y = x-F(x);

    % Define projection operator to the constraint set
    proj = osqp;
    proj.setup(eye(N*n_x), -y(:), A_all, -Inf*ones(size(A_all,1),1), b_all, 'verbose', false, 'eps_abs', eps_err/1000,'eps_rel', eps_err/1000 )

    % options = optimoptions('quadprog','Display','off', 'Algorithm', 'active-set');
    % proj = @(y) reshape( ...
    % quadprog(eye(N*n_x), -y(:), A_all, b_all, [], [], [],[], y(:), options), [n_x,1, N]); 

    x_transf = reshape(proj.solve.x, [n_x,1, N]);
    r = norm(reshape(x - x_transf, [],1));
end