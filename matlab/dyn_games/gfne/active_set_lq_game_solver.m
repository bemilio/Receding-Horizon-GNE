function [x,u,lam,mu,gam,psi,working_set,solve_iters,K] = active_set_lq_game_solver(F,H,G,Q,N,T,m,x0,working_set,unlimited,params)
    open_loop = params.open_loop;
    contact = params.contact;
    solve_iters = 0;
    H_active = H;
    for t = T+1:-1:1
        for i = 1:N
            for j = 1:size(working_set{t,i},1)
                if working_set{t,i}(j) > 0
                    H_active{t,i} = [H_active{t,i}; G{t,i}(j,:)];
                    working_set{t,i}(j) = size(H_active{t,i},1);
                end
            end
        end
    end
    if open_loop
        [K,k] = solve_ec_lq_game_r(F,H_active,Q,N,m,T);
%         for t = 1:T
%             for i = 1:N
%                 K{t,i} = zeros(size(K{t,i}));
%             end
%         end
        [x,u,lam,mu,psi] = solve_ec_lq_game_d(F,H_active,Q,N,T,K,x0);
    else
        disp('Active indices');
        for t= 1:T+1
            for i = 1:N
                for j = 1:size(working_set{t,i})
                    if working_set{t,i}(j) > 0
                        active = [t,i,j]
                    end
                end
            end
        end    
        [x,u,lam,mu,psi] = solve_ec_lq_game_super_fast(F,H_active,Q,N,T,x0,contact);
    end
    solve_iters = solve_iters+1;
    prevprev_working_set = working_set;
    prev_working_set = working_set;
    while true % until feasible initial point found. 
        feasible = true;
        for t = T+1:-1:1
            for i = 1:N
                if t < T+1
                    vec = [1;x{t};u{t,i}];
                else
                    vec = [1;x{t}];
                end
                for j = 1:size(G{t,i},1)
                    if G{t,i}(j,:)*vec < -1e-8 && feasible
                        feasible = false;
                        H_active{t,i} = [H_active{t,i}; G{t,i}(j,:)];  
                        working_set{t,i}(j) = size(H_active{t,i},1);
                    end
                end
            end
        end
        if feasible
            break;
        else
            if open_loop
                [K,k] = solve_ec_lq_game_r(F,H_active,Q,N,m,T);
            
%                 for t = 1:T
%                     for i = 1:N
%                         K{t,i} = zeros(size(K{t,i}));
%                     end
%                 end
                [x,u,lam,mu,psi] = solve_ec_lq_game_d(F,H_active,Q,N,T,K,x0);
            else
                disp('Active indices');
                for t= 1:T+1
                    for i = 1:N
                        for j = 1:size(working_set{t,i})
                            if working_set{t,i}(j) > 0
                                active = [t,i,j]
                            end
                        end
                    end
                end    
                [x,u,lam,mu,psi] = solve_ec_lq_game_super_fast(F,H_active,Q,N,T,x0,contact);
            end
            solve_iters = solve_iters+1;
            working_time_indices = [];
            for t = 1:T+1
                for i = 1:N
                    for j = 1:size(working_set{t,i},1)
                        if working_set{t,i}(j) > 0
                            working_time_indices = [working_time_indices, t];
                        end
                    end
                end
            end
%             working_time_indices
        end
    end
    iters = 0;
    disp('Feasible found');
    while true
        if open_loop
            [K,k] = solve_ec_lq_game_r(F,H_active,Q,N,m,T);
        
%             for t = 1:T
%                 for i = 1:N
%                     K{t,i} = zeros(size(K{t,i}));
%                 end
%             end
            [xn,un,lam,mu,psi] = solve_ec_lq_game_d(F,H_active,Q,N,T,K,x0);
        else
            disp('Active indices');
            for t= 1:T+1
                for i = 1:N
                    for j = 1:size(working_set{t,i})
                        if working_set{t,i}(j) > 0
                            active = [t,i,j]
                        end
                    end
                end
            end    
            [xn,un,lam,mu,psi] = solve_ec_lq_game_super_fast(F,H_active,Q,N,T,x0,contact);
        end
        solve_iters = solve_iters+1;
        step = 0;
        for t = 1:T+1
            if t <= T
                for i = 1:N
                    dU{t,i} = un{t,i}-u{t,i};
                end
            end
            
            dX{t} = xn{t}-x{t};
            step = step + norm(dX{t});
        end
        
        if (step < 1e-6)
            min_mult = inf;
            min_mult_ind = 0;
            num_neg_mults = 0;
            for t = 1:T+1
                for i = 1:N
                    for j = 1:size(G{t,i},1)
                        if working_set{t,i}(j) > 0 
                            mult_ind = working_set{t,i}(j);
                            mult_val = mu{t,i}(mult_ind);
                            if mult_val < -1e-6
                                num_neg_mults = num_neg_mults+1;
                                if mult_val < min_mult
                                    min_mult = mult_val;
                                    min_mult_ind = [t,i,j];
                                end
                            end
                        end
                    end
                end
            end
            if min_mult < -1e-6
                % Need to drop a constraint
                t = min_mult_ind(1);
                i = min_mult_ind(2);
                j = min_mult_ind(3);
                working_set{t,i}(j) = 0;
                H_active{t,i} = H{t,i};
                mu_new = mu{t,i}(1:size(H{t,i},1));
%                 mu{t,i} = [mu{t,i}(1:size(H{t,i},1)+j-1); mu{t,i}(size(H{t,i},1)+j+1:end)];
                ind = size(H_active{t,i},1)+1;
                for k = 1:size(G{t,i},1)
                    if working_set{t,i}(k) > 0
                        H_active{t,i} = [H_active{t,i};G{t,i}(k,:)];
                        mu_new = [mu_new; mu{t,i}(ind)];
                        working_set{t,i}(k) = ind;
                        ind = ind+1;
                    end
                end
                mu{t,i} = mu_new;
            else
                % Solution found!
                break;
            end
        else
            % Need to move, but watch out for constraints
            alpha = 1;
            for t = 1:T+1
                for i = 1:N
                    if t <= T
                        dvec = [dX{t};dU{t,i}];
                        vec = [x{t}; u{t,i}];
                    else
                        dvec = dX{t};
                        vec = x{t};
                    end
                    for j = 1:size(G{t,i},1)
                        if working_set{t,i}(j) == 0
                            if G{t,i}(j,2:end)*dvec < -1e-6
                                needed_alpha = -(G{t,i}(j,1)+G{t,i}(j,2:end)*vec)/(G{t,i}(j,2:end)*dvec);
                                needed_alpha = max(0,needed_alpha);
                                if needed_alpha < alpha
                                    min_alpha_ind = [t,i,j];
                                    alpha = needed_alpha;
                                end
                            end
                        end
                    end
                end
            end
            if alpha < 1
                t = min_alpha_ind(1);
                i = min_alpha_ind(2);
                j = min_alpha_ind(3);
                H_active{t,i} = [H_active{t,i}; G{t,i}(j,:)];
                working_set{t,i}(j) = size(H_active{t,i},1);
            else
                min_mult = inf;
                min_mult_ind = 0;
                num_neg_mults = 0;
                for t = 1:T+1
                    for i = 1:N
                        for j = 1:size(G{t,i},1)
                            if working_set{t,i}(j) > 0 
                                mult_ind = working_set{t,i}(j);
                                mult_val = mu{t,i}(mult_ind);
                                if mult_val < -1e-6
                                    num_neg_mults = num_neg_mults + 1;
                                    if mult_val < min_mult
                                        min_mult = mult_val;
                                        min_mult_ind = [t,i,j];
                                    end
                                end
                            end
                        end
                    end
                end
                if min_mult < -1e-6
                    % Need to drop a constraint
                    t = min_mult_ind(1);
                    i = min_mult_ind(2);
                    j = min_mult_ind(3);
                    working_set{t,i}(j) = 0;
                    H_active{t,i} = H{t,i};
                    mu_new = mu{t,i}(1:size(H{t,i},1));
                    ind = size(H{t,i},1)+1;
                    for k = 1:size(G{t,i},1)
                        if working_set{t,i}(k) > 0
                            H_active{t,i} = [H_active{t,i};G{t,i}(k,:)];
                            mu_new = [mu_new; mu{t,i}(ind)];
                            working_set{t,i}(k) = ind;
                            ind = ind+1;
                        end
                    end
                    mu{t,i} = mu_new;
                else
                    % Solution found!
                    break;
                end
            end
            for t = 1:T+1
                if t <= T
                    for i = 1:N
                        u{t,i} = u{t,i}+alpha*dU{t,i};
                    end
                end
                x{t} = x{t} + alpha*dX{t};
            end
        end  
        working_time_indices = [];
        for t = 1:T+1
            for i = 1:N
                for j = 1:size(working_set{t,i},1)
                    if working_set{t,i}(j) > 0
                        working_time_indices = [working_time_indices, t];
                    end
                end
            end
        end
%         working_time_indices
        iters = iters+1;
        if ~unlimited && iters > 100
            disp('Breaking early!');
            break;
        end
        if iters > 2 
            cycle = true;
            for t = 1:T+1
                for i = 1:N
                    if any(xor(working_set{t,i}>0,prevprev_working_set{t,i}>0))
                        cycle = false;
                    end
                end
            end
            if cycle
                disp('Cycle detected!');
                break;
            end
        end
        prevprev_working_set = prev_working_set;
        prev_working_set = working_set;
    end
    
    for t = 1:T+1
        for i = 1:N
            num_ineqs = 0;
            gam{t,i} = zeros(size(working_set{t,i}));
            for j = 1:size(working_set{t,i},1)
                if working_set{t,i}(j) > 0
                    if working_set{t,i}(j) <= size(mu{t,i},1)
                        num_ineqs = num_ineqs+1;
                        gam{t,i}(j) = mu{t,i}(working_set{t,i}(j));
                    end
                else
                    gam{t,i}(j) = 0;
                end
            end
            mu{t,i} = mu{t,i}(1:end-num_ineqs);
        end
    end
            
          
end