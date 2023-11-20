function u = solveFinHorCL(game, T)
%SOLVEFINHORCL

regions_list=num2cell(ones(game.N,1)');

% initialize each agent to their MPC
for i=1:game.N
    sys_0 = LTISystem('A', game.A, 'B', game.B(:,:,i));
    sys_0.x.min = game.min_x;
    sys_0.x.max = game.max_x;
    sys_0.u.min = game.min_u;
    sys_0.u.max = game.max_u;
    sys_0.x.penalty = QuadFunction(game.Q(:,:,i));
    sys_0.u.penalty = QuadFunction(game.R(:,:,i));
    mpc = MPCController(sys_0, T);
    expmpc = mpc.toExplicit();
    regions_list{i} = 1:expmpc.nr;
end


% Iterate explicit MPC keeping the remaining agents feedback fixe

for i=1:game.N
    % Compute closed loop dynamics (PWA) as seen by agent i
    all_regions_pwa = combinations(regions_list{1:i-1}, regions_list{1+1:end});
    n_regions = size(all_regions_pwa, 1);
    sys = cell(n_regions,1);
    region_ineq_A = [];
    region_ineq_b = [];
    for i_region = 1:size(regions_pwa,1)
        A_cl_i = game.A;
        b_cl_i = zeros(game.n_x, 1);
        
        for j=[1:i-1, i+1:game.N]
            region_j = all_regions_pwa(i_region, j);
            % Fetch controller for ajent j at the current region
            Fj = expmpc(j).feedback.Set(region_j).getFunction('primal').F;
            Fj = Fj(1:n_u, :);
            A_cl_i = A_cl_i + B(:,:,j) * Fj;
            gj = expmpc(j).feedback.Set(region_j).getFunction('primal').g;
            gj = gj(1:n_u,:);
            b_cl_i = b_cl_i + gj;
            region_ineq_A_j = expmpc(j).feedback.Set(region_j).A;
            region_ineq_b_j = expmpc(j).feedback.Set(region_j).b;
            region_ineq_A = [region_ineq_A; region_ineq_A_j];
            region_ineq_b = [region_ineq_A; region_ineq_b_j];
        end
        R = Polyhedron(region_ineq_A, region_ineq_b);
        R = R.minHRep();
        sys{i_region} = LTISystem('A', A_cl_i, 'B', B(:,:,i), 'f', b_cl_i);
        sys{i_region}.setDomain('x', R)
    end
    


end

PWASystem 

end