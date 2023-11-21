function [X_f] = computeTerminalSetCL(game, K)

% Find invariant set to closed loop
A_cl = game.A + sum(pagemtimes(game.B, K), 3);
model  = LTISystem('A', A_cl);
model.x.min = game.min_x;
model.x.max = game.max_x;
X_f = model.invariantSet();

% Intersect with set where input is feasible
for i=1:game.N
    U_i = Polyhedron([K(:,:,i); -K(:,:,i)], [game.max_u; -game.min_u]);
    if ~U_i.isEmptySet() && ~X_f.isEmptySet()
        X_f = X_f & U_i;
    end
end

end