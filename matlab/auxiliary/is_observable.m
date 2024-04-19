function [is_obs] = is_observable(A,C)

Ob = obsv(A,C);

is_obs = (length(A)==rank(Ob));

end

