function [game] = defineLQR()

n_x = 2;
n_u=1;
N=1;
game.n_x = n_x;
game.n_u = n_u;
game.N = N;

game.A = triu(ones(n_x,n_x), 1);

game.B = zeros(n_x, n_u, 1);
game.B(end,:,1) = 1;

game.Q = zeros(n_x, n_x, N);
game.Q(:,:,1) = eye(n_x);

game.R = zeros(n_u, n_u, N);
game.R(:,:,1) = .1*eye(n_u);

end

