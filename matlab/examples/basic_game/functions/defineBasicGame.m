function [game] = defineBasicGame()
n_x = 2;
n_u=1;
N=2;
game.n_x = n_x;
game.n_u = n_u;
game.N = N;

game.A = [.5, 1; 0, .5];

game.B = zeros(2, 1, 2);
game.B(:,:,1) = [0;1];
game.B(:,:,2) = [0;1];

game.Q = zeros(n_x, n_x, N);
game.Q(:,:,1) = diag([1, 0]);
game.Q(:,:,2) = diag([0, 1]);

game.R = zeros(n_u, n_u, N);
game.R(:,:,1) = .1*eye(n_u);
game.R(:,:,2) = .1*eye(n_u);

end

