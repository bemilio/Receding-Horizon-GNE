function [p] = defineLoadFlexGameParameters(N, load, collective_load_ref)

%% Define Parameters 

p.k_g = 1 * ones(N,1); %weight of generator
p.k_b = .1 * ones(N,1); % weight of battery charge deviation from ref
p.k_d = .5 * ones(N,1); % weight of load flexibility usage 

p.r_g = .1 * ones(N,1); %weight of generator input
p.r_b = .1 * ones(N,1); % weight of battery charging rate
p.r_d = .1 * ones(N,1); % weight of load flexibility usage rate

p.q_des = .7 * ones(N,1); % desired battery charge state

p.v_to_P = ones(N,1); % from gen. velocity to power

p.a_g = 0.8 * ones(N,1); % Rate of deceleration of generator spin
p.a_b = 0.99 * ones(N,1); % Rate of battery leakage
p.a_d = .9 * ones(N,1); % Forgetfulness rate of load delay

p.b_b = 0.7 * ones(N,1); % power - to - battery charging rate (efficiency)

p.max_b_charge = 15 * ones(N,1); % kWh - max battery charge
p.max_b_rate = 0.7 * p.max_b_charge;
p.max_g_rate = 0.7 * ones(N,1);

p.min_load_delay = -1 * ones(N,1); %kWh
p.max_load_delay = 1 * ones(N,1); %kWh

p.load = load;
p.collective_load_ref = collective_load_ref;


end

