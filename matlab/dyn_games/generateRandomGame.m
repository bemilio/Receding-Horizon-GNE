function game = generateRandomGame(n_x, n_u, N, sparse_density, pos_def)
% generateRandomGame  Generates random LQ dynamic game.
%   C = generateRandomGame(n_x, n_u, N, sparse_density, pos_def)
%   generates a random game with n_x states, n_u inputs per agent and N
%   agents. The resulting matrices are sparse with density sparse_density.
%   The Q and R matrices are positive definite with minimum eigenvalues >=
%   pos_def.
isControllable = false;
while isControllable==false
    game.A = full(sprand(n_x,n_x,sparse_density)); % turn to full because page multiplication does not support spars
    game.B = zeros(n_x, n_u, N);
    game.Q = zeros(n_x, n_x, N);
    game.R = zeros(n_u, n_u, N);
    isControllable=true;
    for i=1:N
        game.B(:,:,i) = full(sprand(n_x,n_u, sparse_density));
        game.Q(:,:,i) = full(posdefrand(n_x, pos_def, sparse_density));
        game.R(:,:,i) = full(posdefrand(n_u, pos_def, sparse_density));
        Co = ctrb(game.A,game.B(:,:,i));
        if length(game.A)-rank(Co)~= 0
            isControllable=false;
        end
    end
end
game.n_x = n_x;
game.n_u = n_u;
game.N = N;
end

function M = posdefrand(n, pos_def,sparse_density)
M = sprand(n, n, sparse_density);
M = M + M';
min_eig = min(eig(M));
M = M - min(min_eig, 0) * eye(n) + pos_def * eye(n);
end