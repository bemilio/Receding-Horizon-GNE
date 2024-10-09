function [P,K, isStable, A_cl] = solveInfHorCL_cont_time(game, n_iter, eps_err)
P = zeros(game.n_x, game.n_x, game.N);
K = zeros(game.n_u, game.n_x, game.N);
A = game.A;
B = game.B;
Q = game.Q;
R = game.R;
n_x = game.n_x;
n_u = game.n_u;
N = game.N;
B_all = [];
Q_all = zeros(n_x, n_x);
for i=1:N
    B_all = [B_all, B(:,:,i)];
    Q_all = Q_all + Q(:,:,i);
end
if ~is_stabilizable(A, B_all)
    warning("[solveInfHorCL]: stabilizability assump. not satisfied")
end
if ~is_detectable(A, Q_all)
    warning("[solveInfHorCL]: detectability assump. not satisfied")
end

% Initialize to a stabilizing controller
R_all = eye(N*n_u);
P_all = icare(A, B_all, Q_all, R_all);
K_all = -inv(R_all + B_all'*P_all*B_all) * B_all' * P_all * A;
for i =1:N
    indexes = (i-1)*n_u+1: i*n_u;
    K(:,:,i) = K_all(indexes,:);
end

% (adapted from sassano 22): 
% solve P[i] = Q[i] + K[i]'R[i]K[i] + A_cl' P[i] A_cl 
% b iterating the Riccati equation
% A_cl' P[i] A_cl + Q[i] + K[i]'R[i]K[i] = P[i];
% K[i] = - R[i]B[i]'P[i]A_cl 
for k=1:n_iter
    for i=1:N
        A_cl_not_i = A + sum(pagemtimes(B, K), 3) - B(:,:, i)*K(:,:,i);
        P(:,:,i) = icare(A_cl_not_i, B(:,:,i), Q(:,:,i), R(:,:,i));
        K(:,:,i) = - R(:,:,i) \ B(:,:,i)' * P(:,:,i);
    end
    if mod(k,20)==0
        % Test solution by verifying distance from optimal response
        err = 0;
        for i=1:N
            A_cl_not_i = A + sum(pagemtimes(B, K), 3) - B(:,:, i)*K(:,:,i);
            err = max(err, norm(P(:,:,i) - ...
                icare(A_cl_not_i, B(:,:,i), Q(:,:,i), R(:,:,i)) ));
        end
        if  err < eps_err
            break
        end
    end
end
if  err > eps_err
    warning("[solveInfHorCL] Could not find infinite horizon CL-NE")
end
for i=1:N
    if min(eig(P(:,:,i))) < -eps_err
        warning("The closed-loop P is non-positive definite")
    end
end
A_cl = A + sum(pagemtimes(B, K), 3);
if max(real(eig(A + sum(pagemtimes(B, K), 3)))) > 0.0001
    warning("The infinite horizon CL-GNE has an unstable dynamics")
    isStable = false;
else
    isStable=true;
end

end