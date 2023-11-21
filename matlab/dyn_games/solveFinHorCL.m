function expmpc = solveFinHorCL(game, T)
%SOLVEFINHORCL

regions_list=num2cell(ones(game.N,1)');

N_test_points = 100;
test_points = (rand(game.n_x,N_test_points) .* (game.max_x - game.min_x)) + game.min_x; % used to test convergence

n_iter = 100;
eps_convergence = 10^(-4);

expmpc = cell(game.N, 1);

% initialize each agent to their MPC
for i=1:game.N
    sys_0 = LTISystem('A', game.A, 'B', game.B(:,:,i));
    sys_0.x.min = game.min_x;
    sys_0.x.max = game.max_x;
    sys_0.u.min = game.min_u;
    sys_0.u.max = game.max_u;
    sys_0.x.penalty = QuadFunction(game.Q(:,:,i));
    sys_0.u.penalty = QuadFunction(game.R(:,:,i));
    sys_0.x.with('terminalPenalty');
    sys_0.x.terminalPenalty =  QuadFunction(game.P(:,:,i));
    mpc = MPCController(sys_0, T);
    expmpc{i} = mpc.toExplicit();
    regions_list{i} = 1:expmpc{i}.nr;
end

last_ctrl_at_test = computeCtrlAtTestPoints(test_points, expmpc);

sys_0.LQRSet
% Iterate explicit MPC keeping the remaining agents feedback fixed
for k=1:n_iter
    for i=1:game.N
        % Compute closed loop dynamics (PWA) as seen by agent i
        all_regions_pwa = table2array(combinations(regions_list{1:i-1}, regions_list{1+1:end}));
        all_regions_pwa = [all_regions_pwa(:,1:i-1), ...
                           zeros(size(all_regions_pwa, 1), 1), ... % put a zero column associated to agent i (just to make indexing easier later on)
                           all_regions_pwa(:, i:end) ];
        n_regions = size(all_regions_pwa, 1);
        sys = [];
        region_ineq_A = [];
        region_ineq_b = [];
        for i_region = 1:size(all_regions_pwa,1)
            A_cl_i = game.A;
            b_cl_i = zeros(game.n_x, 1);
            R = Polyhedron.fullSpace(game.n_x);
            for j=[1:i-1, i+1:game.N]
                region_j = all_regions_pwa(i_region, j);
                R = R & expmpc{j}.feedback.Set(region_j); % intersection
                if ~R.isEmptySet
                    % Fetch controller for ajent j at the current region
                    Fj = expmpc{j}.feedback.Set(region_j).getFunction('primal').F;
                    Fj = Fj(1:game.n_u, :);
                    A_cl_i = A_cl_i + game.B(:,:,j) * Fj;
                    % Affine part of j-th control action
                    gj = expmpc{j}.feedback.Set(region_j).getFunction('primal').g;
                    gj = gj(1:game.n_u,:);
                    b_cl_i = b_cl_i + game.B(:,:,j) * gj;
                end
            end
            if ~R.isEmptySet
                R = R.minHRep();
                sys = [sys, LTISystem('A', A_cl_i, 'B', game.B(:,:,i), 'f', b_cl_i)];
                sys(end).setDomain('x', R);
            end
        end
        pwasys = PWASystem(sys); % WARNING: DO NOT FEED EMPTY SETS OR THERE WILL BE PROBLEMS!
        pwasys.x.min = game.min_x;
        pwasys.x.max = game.max_x;
        pwasys.u.min = game.min_u;
        pwasys.u.max = game.max_u;
        pwasys.x.penalty = QuadFunction(game.Q(:,:,i));
        pwasys.u.penalty = QuadFunction(game.R(:,:,i));
        pwasys.x.with('terminalPenalty');
        pwasys.x.terminalPenalty = QuadFunction(game.P(:,:,i));
        mpc = MPCController(pwasys, T);
        expmpc{i} = mpc.toExplicit();
        regions_list{i} = 1:expmpc{i}.nr;  
        poly = [];
        for idx_union=1:length(expmpc{i}.feedback)
            for idx_poly=1:length(expmpc{i}.feedback(idx_union).Set)
                poly = [poly, expmpc{i}.feedback(idx_union).Set(idx_poly)];
            end
        end
        expmpc{i}.feedback = PolyUnion(poly);
    end 
    ctrl_at_test = computeCtrlAtTestPoints(test_points, expmpc);
    err = max(max(max(last_ctrl_at_test - ctrl_at_test)));
    disp("Iteration: " + num2str(k) + "; Error = " + num2str(err))
    if err < eps_convergence
        break
    end
    last_ctrl_at_test = ctrl_at_test;
end

end


function ctrl_at_test = computeCtrlAtTestPoints(test_points, expmpc)
    N = length(expmpc);
    n_u = expmpc{1}.nu;
    n_tests = size(test_points,2);
    ctrl_at_test = zeros(n_u, N, n_tests); 
    for i_test=1:size(test_points,2)
        for i=1:N
            ctrl_at_test(:, i, i_test) = expmpc{i}.evaluate(test_points(:,i_test));
        end
    end
end
