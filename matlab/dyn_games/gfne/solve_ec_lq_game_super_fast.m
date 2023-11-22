%% Test EC LQ
% This version computes the explicit riccati backup, without considering
% the lagrange multipliers. First substitutes in dynamics, then each player
% minimizes norm of their own constraints (may be suboptimal with respect 
% to their objective if there are multiple ways of doing this).
function [dX,dU,dL,dM,dP,Kk] = solve_ec_lq_game_super_fast(F,... % time-indexed dict of dynamics
                                                  H,... % time-indexed dict of player-indexed constraints
                                                  Q,... % time-indexed dict of player-indexed costs
                                                  N,... % number of players
                                                  T,...
                                                  x0,...
                                                  contact)     
        
                                              
                                              
    
%     K = cell(T,N);
%     k = cell(T,N);
%     L = cell(T,N);
%     l = cell(T,N);
    
    Kk = cell(T,1);
    Ll = cell(T+1,1);

    n = size(F{1},1);
    all_m = 0;
    for i = 1:N
        m{i} = size(H{1,i},2) - 1 - n;
        start_m{i} =  all_m;
        all_m = all_m + m{i};
    end
   
    for i = 1:N
        G{i} = H{T+1,i}; % constraint-to-go 
        P{i} = Q{T+1,i}; % cost-to-go
    end
    
    over_constrained = false;
	for t = T:-1:1
        all_feas{t} = true;
        for i = 1:N
            B = F{t}(:,1+n+start_m{i}+1:1+n+start_m{i}+m{i});
            Hu = H{t,i}(:,n+2:end);
            HHu = [G{i}(:,2:end)*B; Hu];
            rk = rank(HHu);
            all_feas{t} = all_feas{t} && (rk == size(HHu,1));
        end

        if all_feas{t}
            if over_constrained
                Ft = F{t};
                dyn = [1 zeros(1,n+all_m);
                       Ft];
                for i = 1:N
                    Ht{i} = H{t,i};
                    Qt{i} = Q{t,i};
                end

                [G,P,KKt] = solve_over_constrained_step(Ft,Ht,G,Qt,P,N,m);
                Kk{t} = KKt;
                

                S = 1 + first_over_constrained - t;
                for s = 1:S
                    FF{s} = F{t+s-1};
                    for i = 1:N
                        HH{s,i} = H{t+s-1,i};
                        QQ{s,i} = Q{t+s-1,i};
                        KK{s,i} = Kk{t+s-1}(start_m{i}+1:start_m{i}+m{i},2:end);
                    end
                end
                for i = 1:N
                    HH{S+1,i} = Goc{i};
                    QQ{S+1,i} = Poc{i};
                end
                [~,~,LAM,MM,~] = solve_ec_lq_game_dpol(FF,HH,QQ,N,S,KK);

                for i = 1:N
                   Ll{t+S,i} = MM{S+1,i};
                   LAMLAM{t+S-1,i} = LAM{S,i};
                end
                Lid{t+S} = t;

                for s = S:-1:1
                    for i = 1:N
                        Ll{t+s-1,i} = MM{s,i};
                        if s>1
                            LAMLAM{t+s-2,i} = LAM{s-1,i};
                        end
                    end
                    Lid{t+s-1} = t;
                end
                for i = 1:N
                    G{i} = zeros(0,n+1);
                end
                
                % need to go and get multipliers
            else
            
         
                Ft = F{t};
                dyn = [1 zeros(1,n+all_m);
                       Ft];
                for i = 1:N
                    Ht{i} = H{t,i};
                    Qt{i} = Q{t,i};
                end
                
                [G,P,KKt,LLt,Lampolt] = solve_under_constrained_step(Ft,Ht,G,Qt,P,N,m,all_m);
                Kk{t} = KKt;
                for i = 1:N
                    Ll{t,i} = LLt{i}(1:size(Ht{i},1),:);
                    if numel(G{i}) > 0 || t==T
                        Lid{t+1} = t;
                        Ll{t+1,i} = LLt{i}(1+size(Ht{i},1):end,:);
                    end
                    LAMLAM{t,i} = Lampolt{i};
                end
                Lid{t} = t;
        
                
            end
            over_constrained = false;
        else
            if ~over_constrained
                first_over_constrained = t;
                Goc = G;
                Poc = P;
            end
            over_constrained = true;
            Ft = F{t};
            dyn = [1 zeros(1,n+all_m);
                   Ft];
            for i = 1:N
                Ht{i} = H{t,i};
                Qt{i} = Q{t,i};
            end
            
            [G,P,KKt] = solve_over_constrained_step(Ft,Ht,G,Qt,P,N,m);
            Kk{t} = KKt;
            
        end
    end
    
    dX{1} = x0;
    for t = 1:T
        u = Kk{t}*[1;dX{t}];
        for i = 1:N
            dU{t,i} = u(start_m{i}+1:start_m{i}+m{i});
            dM{t,i} = Ll{t,i}*[1;dX{Lid{t}}];
            dL{t,i} = LAMLAM{t,i}*[1;dX{Lid{t}}];
        end
        dX{t+1} = F{t}*[1;dX{t};u];
    end
    for i = 1:N
        dM{T+1,i} = Ll{T+1,i}*[1;dX{Lid{T+1}}];
    end
    
    dP = 0;
     
end