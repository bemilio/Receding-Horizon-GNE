function [X_f] = computeTerminalSetCL(game)

% Find invariant set to closed loop
A_cl = game.A + sum(pagemtimes(game.B, game.K_cl), 3);
model  = LTISystem('A', A_cl);
model.x.min = -ones(game.n_x, 1) * 10^3; % large bounded set
model.x.max = ones(game.n_x, 1) * 10^3;
X_const_set = Polyhedron('A', game.C_x, 'b', game.d_x);  
model.x.with('setConstraint');
model.x.setConstraint = X_const_set;
poly = model.invariantSet();

% Intersect with set where input is feasible
for i=1:game.N
    U_i = Polyhedron(game.C_u_loc(:,:,i) *  game.K_cl(:,:,i), game.d_u_loc(:,:,i));
    if ~U_i.isEmptySet() && ~poly.isEmptySet()
        poly = poly & U_i;
    end
end

P = dlyap(A_cl, eye(game.n_x));

r = inscribeEllipseInPoly(P, poly.A, poly.b);
X_f = EllipsoidSet(P, r);

end



