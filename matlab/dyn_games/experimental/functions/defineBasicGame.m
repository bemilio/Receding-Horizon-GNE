function [game] = defineBasicGame()
n_x = 2;
n_u=1;
N=2;
game.n_x = n_x;
game.n_u = n_u;
game.N = N;

game.Q = zeros(n_x, n_x, N);
game.B = zeros(n_x, n_u, N);
game.R = zeros(n_u, n_u, N);

if n_x ==1
    game.A = 1;
    game.B(:,:,1) = 1;
    game.B(:,:,2) = 1;
    game.Q(:,:,1) = 1;
    game.Q(:,:,2) = .5;
    game.R(:,:,1) = .1*eye(n_u);
    game.R(:,:,2) = .1*eye(n_u);
end


if n_x ==2
    game.A = [-.1, 1; 0, -.1];
    game.B(:,:,1) = [0;1];
    game.B(:,:,2) = [0;1];
    game.Q(:,:,1) = .1*diag([1, 0]);
    game.Q(:,:,2) = .1*diag([0, 1]);

    game.R(:,:,1) = .1*eye(n_u);
    game.R(:,:,2) = .1*eye(n_u);
end

% Decoupled systems
% if n_x ==2
%     game.A = [-.5, 0; 0, -.5];
%     game.B(:,:,1) = [1;0];
%     game.B(:,:,2) = [0;1];
%     game.Q(:,:,1) = .1*diag([1, 0]);
%     game.Q(:,:,2) = .1*diag([0, 1]);
% 
%     game.R(:,:,1) = .1*eye(n_u);
%     game.R(:,:,2) = .1*eye(n_u);
% end


end

