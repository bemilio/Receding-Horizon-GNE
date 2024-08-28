function [is_ctrb] = is_controllable(A,B)

Co = ctrb(A,B);

is_ctrb = (length(A)==rank(Co));

end

