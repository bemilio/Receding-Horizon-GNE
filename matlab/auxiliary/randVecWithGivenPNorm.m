function x=randVecWithGivenPNorm(P,r)
% Generate random vector such that the P-weighted norm is r
n_x = size(P,1);
x = rand(n_x,1);
x = x*r/sqrt(x'*P*x);
end