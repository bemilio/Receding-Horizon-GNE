function [is_obs] = is_detectable(A,C)

Ob = obsv(A,C);

is_obs = (length(A)==rank(Ob));

%% Hautus lemma for stabilizability, see Rawlings, Maine, Diehl book Lemma 1.12
eigv_A = eig(A);
n_x = size(A,1);
is_stbl = true;
for i = 1:length(eigv_A)
    lambda = eigv_A(i);
    if abs(lambda)>=1
        is_stbl = is_stbl && (rank([lambda*eye(n_x)-A; C])==n_x);
    end
end


end

