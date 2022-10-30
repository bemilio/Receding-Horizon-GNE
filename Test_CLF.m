N = 10;

A = [1, 1; 0,1];
B = [0;1];

n_states = size(A,1);
n_inputs = size(B,2);
x = zeros([N, n_states]);
u = zeros([N, n_inputs]);
F_x = zeros([N, n_states]);
F_u = zeros([N, n_inputs]);

t_simulations = 20;
T = 10;

gamma_shared_cost = 10;

% Update dynamics
for i=1:N
    sigma_x = sum(x,1);
    F_x(i, :) = x(i,:) + gamma_shared_cost * ( sigma_x.^2 + 2 * x(i,:) .* sigma_x );
    F_u(i, :) = u(i,:) + gamma_shared_cost * ( sigma_u.^2 + 2 * u(i,:) .* sigma_u );
    
    x(i,:)= A*x(i,:) + B* u;
end
