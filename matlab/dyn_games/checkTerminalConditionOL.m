function [flag] = checkTerminalConditionOL(x, X_f)

flag = true;
for i=1:length(X_f)
    x_lifted = [x;x];
    flag = flag & X_f{i}.contains(x_lifted);
end

end

