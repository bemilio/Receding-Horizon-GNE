function [X_f] = computeTerminalSetOL(game)

% Define closed-loop system
A_ol = game.A + sum(pagemtimes(game.B, game.K_ol), 3);

% Compute control Lyap. function for lifted system
P = dlyap(A_ol, eye(game.n_x)); 

% Define state constraints-admissible set
poly = Polyhedron('A', game.C_x, 'b', game.d_x);

% Define input-admissible set for the controller game.K_ol
for i=1:game.N
    % Set admissible for i-th local input constraints
    U_i = Polyhedron(game.C_u_loc(:,:,i) * game.K_ol(:,:,i), game.d_u_loc(:,:,i));
    % Intersect with current set estimation
    if ~U_i.isEmptySet() && ~poly.isEmptySet()
        poly = poly & U_i;
    end
end
% Set admissible for shared input constraints
U_sh = Polyhedron( sum(pagemtimes(game.C_u_sh, game.K_ol), 3), game.d_u_sh );
% Intersect with current set estimation
poly = poly & U_sh;

r = inscribeEllipseInPoly(P, poly.A, poly.b);
X_f = EllipsoidSet(P, r);

end



