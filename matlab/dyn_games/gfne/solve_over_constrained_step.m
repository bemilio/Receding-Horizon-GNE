function [G,P,upol] = solve_over_constrained_step(Ft,...
                                                  Ht,... 
                                                  G,...
                                                  Qt,... 
                                                  P,...
                                                  N,... 
                                                  m)   
      
    n = size(Ft,1);                                         
    all_m = 0;
    for i = 1:N
        m_start{i} = all_m;
        all_m = all_m + m{i};
    end

    dyn = [1 zeros(1,n+all_m);
           Ft];
    Mall = [];
    tot = 0;
    for i = 1:N
        Qt{i} = Qt{i} + dyn'*P{i}*dyn;
        nh = size(Ht{i},1);
        Ht{i} = [Ht{i}(:,1:1+n) zeros(nh, m_start{i}) Ht{i}(:,2+n:end) zeros(nh,all_m-m_start{i}-m{i})];
        Ht{i} = [Ht{i}; G{i}*dyn];
        Hut{i} = Ht{i}(:,1+n+m_start{i}+1:1+n+m_start{i}+m{i});
        [U,S,V] = svd(Hut{i});
        rk{i} = rank(Hut{i},1e-6);
        z{i} = m{i}-rk{i};
        n_start{i} = tot;
        tot = tot+z{i};
        Ur = U(:,1:rk{i});
        Un = U(:,rk{i}+1:end);
        Vr{i} = V(:,1:rk{i});
        Vn{i} = V(:,rk{i}+1:end);
        Sr = S(1:rk{i},1:rk{i});
        Mt{i} = Ur'*Ht{i};
        Mall = [Mall; Mt{i}];
        Rt{i} = Un'*Ht{i};
    end

    Vra = blkdiag(Vr{:});
    Vna = blkdiag(Vn{:});
    Va = [Vra Vna];

    mod = [eye(1+n) zeros(1+n,all_m);
           zeros(all_m, 1+n) Va];
    % [1;x;u] = mod*[1;x;z1;z2]
    Mall = Mall*mod;

    z1pol = -Mall(:,1+n+1:1+n+size(Vra,2))\[Mall(:,1:1+n) Mall(:,2+n+size(Vra,2):end)];
    % z1 = z1pol * [1;x;z2]

    z2mod = mod*[eye(1+n) zeros(1+n, size(z1pol,2)-1-n);
                 z1pol;
                 zeros(size(z1pol,2)-1-n,1+n), eye(size(z1pol,2)-1-n)];
    % [1;x;u] = z2mod*[1;x;z2]

    Nall = [];
    for i = 1:N
        Qtz{i} = z2mod'*Qt{i}*z2mod;
        Nall = [Nall; Qtz{i}(1+n+n_start{i}+1:1+n+n_start{i}+z{i},:)];
    end
    z2pol = -Nall(:,1+n+1:end)\Nall(:,1:1+n);
    % z2 = z2pol * [1;x]

    upol = Va*[z1pol;
               zeros(size(z2pol,1),1+n) eye(size(z2pol,1))] * [eye(n+1);z2pol];

    % u = upol*[1;x]
    upd = [eye(1+n);upol];
    for i = 1:N
        P{i} = upd'*Qt{i}*upd;
        G{i} = Rt{i}*upd;
    end
end