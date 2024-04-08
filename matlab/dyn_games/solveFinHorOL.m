function [y, dual] = solveVI(VI, prim_stepsz, dual_stepsz, N_iter, y_0, dual_0)
    
    projection = @(x) quadprog(eye(n_x), x, VI.A_loc, VI.b_loc);
    y = y_0;
    dual = dual_0;
    for k=1:N_iter
        y = projection(y - prim_stepsz*(VI.F(y) + pagemtimes(T3D(A_sh), dual)));
        d = 2 * sum(pagemtimes(A_sh, y) - b_sh, 3);
        dual = max(dual + dual_stepsz * d, 0);
    end
end