function [G,P,K,L,Lampol] = solve_under_constrained_step(Ft,...
                                                  Ht,... 
                                                  G,...
                                                  Qt,... 
                                                  P,...
                                                  N,... 
                                                  m,all_m) 
                                              
            n = size(Ft,1);                                                   
            dyn = [1 zeros(1,n+all_m);
                   Ft];

            all_lg = 0;
            all_lh = 0;
            for i = 1:N
                lg{i} = size(G{i},1);
                lh{i} = size(Ht{i},1);
                all_lg = all_lg + lg{i};
                all_lh = all_lh + lh{i};
            end
            
            M = zeros(all_m + all_lh + (N+1)*n + all_lg);
            Mx = zeros(size(M,1),n);
            Mc = zeros(size(M,1),1);

            ind = 0;
            ind2 = 0;
            ind3 = 0;
            for i = 1:N
                M(ind+1:ind+m{i},1:all_m) = Qt{i}(ind+n+2:ind+n+m{i}+1,2+n:end);
                M(ind+1:ind+m{i},all_m+ind2+1:all_m+ind2+lh{i}) = Ht{i}(:,2+n:end)';
                M(ind+1:ind+m{i},all_m+all_lh+1+(i-1)*n:all_m+all_lh+n*i) = Ft(:,2+n+ind3:1+n+ind3+m{i})';
                Mx(ind+1:ind+m{i},:) = Qt{i}(ind+n+2:ind+n+m{i}+1,2:1+n);
                Mc(ind+1:ind+m{i}) = Qt{i}(ind+n+2:ind+n+m{i}+1,1);
                ind = ind+m{i};
                ind2 = ind2+lh{i};
                ind3 = ind3+m{i};
            end
            ind2 = 0;
            for i = 1:N
                M(ind+1:ind+lh{i},ind2+1:ind2+m{i}) = Ht{i}(:,2+n:end);
                Mx(ind+1:ind+lh{i},:) = Ht{i}(:,2:1+n);
                Mc(ind+1:ind+lh{i}) = Ht{i}(:,1);
                ind = ind+lh{i};
                ind2 = ind2+m{i};
            end
            M(ind+1:ind+n,1:all_m) = Ft(:,2+n:end);
            Mx(ind+1:ind+n,:) = Ft(:,2:1+n);
            Mc(ind+1:ind+n) = Ft(:,1);
            M(ind+1:ind+n,all_m+all_lh+n*N+1:all_m+all_lh+n*N+n) = -eye(n);
            ind = ind+n;
            ind2 = all_m+all_lh+n*N;
            ind3 = 0;
            for i = 1:N
                M(ind+1:ind+n,ind2+1:ind2+n) = P{i}(2:n+1,2:n+1);
                M(ind+1:ind+n,ind2-(N+1-i)*n+1:ind2-(N-i)*n) = -eye(n);
                M(ind+1:ind+n,ind2+n+ind3+1:ind2+n+ind3+lg{i}) = G{i}(:,2:end)';
                Mc(ind+1:ind+n) = P{i}(2:n+1,1);
                ind = ind+n;
                ind3 = ind3+lg{i};
            end

            for i = 1:N
                M(ind+1:ind+lg{i},ind2+1:ind2+n) = G{i}(:,2:end);
                Mc(ind+1:ind+lg{i}) = G{i}(:,1);
                ind = ind+lg{i};
            end
            
%             t = independent_systems(tt);
            M = sparse(M);
            Mrhs = sparse([Mc Mx]);
            policies = -M\Mrhs;
            K = policies(1:all_m,:);
            ind1 = 0;
            ind2 = 0;
            for i = 1:N
                L{i} = [policies(all_m+ind1+1:all_m+ind1+lh{i},:); policies(all_m+all_lh+(N+1)*n+ind2+1:all_m+all_lh+(N+1)*n+ind2+lg{i},:)];
                ind1 = ind1+lh{i};
                ind2 = ind2+lg{i};
                Lampol{i} = policies(all_m+all_lh+1+(i-1)*n:all_m+all_lh+n*i,:);
            end
%             L = [policies(all_m+1:all_m+all_lh,:); policies(all_m+all_lh+(N+1)*n+1:end,:)];
            pol = [eye(n+1); K];
            
            for i = 1:N
                P{i} =  pol'*(Qt{i} + dyn'*P{i}*dyn)*pol;
                G{i} = zeros(0,n+1);
            end
end