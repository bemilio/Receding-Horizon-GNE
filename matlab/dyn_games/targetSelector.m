function [u] = targetSelector(A,B,R,x)
% Solve min_u u'Ru  s.t.  Ax + \sum B_iu_i = x
n_x = size(A,1);
n_u = size(B,2);
N = size(B,3);
beq = x-A*x;
B_all = [];
R_all = [];
for i=1:N
    B_all = [B_all, B(:,:,i)];
    R_all = blkdiag(R_all, R(:,:,i));
end
options = optimoptions('quadprog', 'Display', 'none');
u_all = quadprog(R_all, zeros(N*n_u,1), [],[], B_all, beq, [], [], [], options);
u = zeros(n_u,1,N);
for i =1:N
    indexes = (i-1)*n_u+1: i*n_u;
    u(:,:,i) = u_all(indexes);
end

end

