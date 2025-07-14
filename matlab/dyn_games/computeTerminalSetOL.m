function [X_f] = computeTerminalSetOL(game)

% Find invariant set for the lifted system by solving for the optimal
% controller of the lifted system and finding a lyap. function for the
% controlled system. This second step is necessary as the P matrix of the
% lifted system might not be pos. def.
for i=1:game.N
    A_ol = game.A + sum(pagemtimes(game.B, game.K_ol), 3);
    sum_BK_not_i = sum(pagemtimes(game.B, game.K_ol), 3) - game.B(:,:,i)*game.K_ol(:,:,i);
    A_lifted = [game.A, sum_BK_not_i;
                zeros(game.n_x), A_ol];
    B_lifted = [game.B(:,:,i); zeros(game.n_x, game.n_u)];
    Q_lifted = [game.Q(:,:,i),      zeros(game.n_x);
                zeros(game.n_x),    zeros(game.n_x)];
    [~, K_i_lifted] = idare(A_lifted, B_lifted, Q_lifted, game.R(:,:,i));
    K_i_lifted = -K_i_lifted;
    P_lifted = dlyap(A_lifted + B_lifted*K_i_lifted , eye(2*game.n_x)); 
    poly = Polyhedron('A', blkdiag(game.C_x,game.C_x), 'b', [game.d_x; game.d_x]);
    K_i_lqr = K_i_lifted(:,1:game.n_u);
    K_i_tilde = K_i_lifted(:,game.n_u+1:end);
    U_i = Polyhedron(game.C_u_loc(:,:,i) * [K_i_lqr, K_i_tilde], game.d_u_loc(:,:,i));
    if ~U_i.isEmptySet() && ~poly.isEmptySet()
        poly = poly & U_i;
    end

    r = inscribeEllipseInPoly(P_lifted, poly.A, poly.b);
    X_f{i} = EllipsoidSet(P_lifted, r);

end


end



