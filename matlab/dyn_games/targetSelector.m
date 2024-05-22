function [x, u] = targetSelector(A,B,R, Q, C_eq_x, C_eq_u, d_eq_u, C_x, d_x, C_u_loc, d_u_loc)
% Solve min_u u'Ru  s.t.  Ax + \sum B_iu_i = x
n_x = size(A,1);
n_u = size(B,2);
N = size(B,3);
B_all = [];
R_all = [];
C_eq_u_all = [];
C_ineq_u_all = [];
d_ineq_u_all = [];
for i=1:N
    B_all = [B_all, B(:,:,i)];
    R_all = blkdiag(R_all, R(:,:,i));
    C_eq_u_all = [C_eq_u_all, C_eq_u(:,:,i)];
    C_ineq_u_all = blkdiag(C_ineq_u_all, C_u_loc(:,:,i));
    d_ineq_u_all = [d_ineq_u_all; d_u_loc(:,:,i)];
end
Q_R = blkdiag(zeros(n_x),R_all); % What is a good choice for the weight on the x?
C_eq_all = [[A-eye(n_x), B_all];
            C_eq_x, C_eq_u_all];
d_eq_all = [zeros(n_x,1); d_eq_u];

C_ineq_all = blkdiag(C_x, C_ineq_u_all);
d_ineq_all = [d_x; d_ineq_u_all];

options = optimoptions('quadprog', 'Display', 'none');
x_u_all = quadprog(Q_R, zeros(n_x + N*n_u,1), C_ineq_all, d_ineq_all, C_eq_all, d_eq_all, [], [], [], options);
x = x_u_all(1:n_x);
u = zeros(n_u,1,N);
for i =1:N
    indexes = n_x + (i-1)*n_u+1: n_x + i*n_u;
    u(:,:,i) = x_u_all(indexes);
end

end

