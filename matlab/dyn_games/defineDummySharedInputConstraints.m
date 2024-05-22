function [C_u_sh,d_u_sh] = defineDummySharedInputConstraints(n_u,N)
C_u_sh = zeros(1, n_u, N);
d_u_sh = ones(1,1);
end