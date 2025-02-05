function [r] = inscribeEllipseInPoly(P,A,b)
% Given a polygon A x <= b and an ellipse x'Px <= r with r unknown
% Find r small such that the ellipse is insyde the poly

% radius of smallest circle inscribed in the poly
r_circle = inf;
for i=1:size(A,1)
    r_circle = min(r_circle, b(i)/norm(A(i,:)));
end

% find radius of ellipse inscribed in circle

r = r_circle *  min(eig(P));

end

