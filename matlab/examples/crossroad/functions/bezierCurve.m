
% Bézier Curve Function
function curve = bezierCurve(P, t)
    B = [(1 - t).^3; 3 * (1 - t).^2 .* t; 3 * (1 - t) .* t.^2; t.^3]; % (100x4)
    curve = B' * P; % Multiply with control points (ensures correct dimensions)
end