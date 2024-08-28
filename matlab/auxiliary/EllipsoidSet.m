classdef EllipsoidSet
    %ELLIPSOIDSET x'Px <= d
    
    properties
        P, d, n_x, P_half
    end
    
    methods
        function obj = EllipsoidSet(P, d)
            %ELLIPSOIDSET Construct an instance of this class
            if norm(P - P') > eps
                warning("[EllipsoidSet] Non-symmetric matrix detected");
            end
            if min(eig(P+P') < eps)
                error("[EllipsoidSet] Matrix must be pos. definite");
            end
            obj.P = P;
            obj.P_half = sqrtm(P);
            obj.d = d;
            obj.n_x = size(P,1);
        end
        
        function y = project(self, x)
            if size(x,1) ~= self.n_x
                error("[EllipsoidSet:project] Size mismatch")
            end
            % |y-x| <= z, z is an aux. var that I put in element 1
            A_epigr = [zeros(self.n_x,1), eye(self.n_x)];
            d_epigr = [1; zeros(self.n_x,1)];
            constr(1) = secondordercone(A_epigr, x, d_epigr, 0);
            % x'Px <= d
            A_ellips = [zeros(self.n_x,1), self.P_half];
            d_ellips = zeros(self.n_x+1,1);
            constr(2) = secondordercone(A_ellips, zeros(self.n_x,1), d_ellips, -self.d );

            cost = [1; zeros(self.n_x,1)];
            y_with_aux = coneprog(cost, constr);
            y = y_with_aux(2:end,1);
        end

        function flag = contains(self, x)
            flag =  (x'*self.P * x <= self.d);
        end
    end
end

