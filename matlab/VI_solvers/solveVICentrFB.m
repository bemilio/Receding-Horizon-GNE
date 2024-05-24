function [x, d, r, solved] = solveVICentrFB(VI, ...
                              n_iter, ...
                              eps_err, ...
                              p_step, ...
                              d_step, ...
                              x, ...
                              d)
if ~exist ('n_iter', 'var')
    n_iter = 10^(5);
end

if ~exist ('eps_err', 'var')
    eps_err = 10^(-5);
end

if ~exist ('p_step', 'var')
    p_step = 0.001;
end

if ~exist ('d_step', 'var')
    d_step = 0.001;
end

if ~exist ('x', 'var')
    x = zeros(VI.n_x, 1, VI.N);
end

if ~exist ('d', 'var')
    d = zeros(size(VI.A_sh, 1), 1);
end

n_loc_constr = size(VI.A_loc, 1);
A_loc_all = zeros(VI.N * n_loc_constr, VI.N * VI.n_x);
b_loc_all = zeros(VI.N * n_loc_constr, 1);
for i=1:VI.N
    A_loc_all((i-1) * n_loc_constr + 1 : i * n_loc_constr, (i-1) * VI.n_x + 1 : i * VI.n_x ) = VI.A_loc(:,:,i);
    b_loc_all((i-1) * n_loc_constr + 1: i * n_loc_constr ) = VI.b_loc(:,:,i);
end
options = optimoptions('quadprog','Display','off', 'Algorithm', 'active-set');
projection = @(y) reshape( ...
    quadprog(eye(VI.N * VI.n_x), reshape(-y, [VI.N*VI.n_x,1]), A_loc_all, b_loc_all, [], [], [],[], y, options), ...
    [VI.n_x,1, VI.N]); 

solved = false;
for k =1:n_iter
    [x, d] = run_FB_once(x, d, VI.F, VI.A_sh, VI.b_sh, projection, p_step, d_step);
    if mod(k,20)==0
        r = compute_residual(x,d, VI.F, VI.A_sh, VI.b_sh, projection);
        if mod(k,300)==0
            disp("Residual: " + num2str(r));
        end
        if r < eps_err
            solved = true;
            break
        end
    end
end

end

function [x_new,d_new] = run_FB_once(x,d, F, A_sh, b_sh, B, p_step, d_step)
    x_new = B(x - p_step*(F(x) + pagemtimes(T3D(A_sh), d)));
    d_half = 2 * sum(pagemtimes(A_sh, x_new) - pagemtimes(A_sh, x) - b_sh, 3);
    d_new = max(d + d_step * d_half, 0);    
end

function r = compute_residual(x,d, F, A_sh, b_sh, B)
    x_transf = B(x-F(x) - pagemtimes(T3D(A_sh), d) );
    d_transf = max(d + sum(pagemtimes(A_sh, x) - b_sh, 3), 0);
    x_res = norm(reshape(x - x_transf, [],1));
    d_res = norm(reshape(d-d_transf, [],1));
    r = sqrt( x_res^2 + d_res^2 );
end