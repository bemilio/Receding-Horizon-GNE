function game = genIntegratorsGame(n_x, n_u, N, contraction_rate)
game.A = contraction_rate * eye(n_x);
game.B = zeros(n_x, n_u, N);
for i=1:N
    game.B(:,:,i) = eye(n_x, n_u);
end
game.Q = zeros(n_x, n_x, N);
game.R = zeros(n_u, n_u, N);
for i=1:N
    game.Q(:,:,i) = eye(n_x);
    game.R(:,:,i) = eye(n_u);
end
game.n_x = n_x;
game.n_u = n_u;
game.N = N;
end
