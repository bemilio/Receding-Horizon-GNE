function [p] = defineLoadFlexGameParameters(N, load, collective_load_ref)

%% Define Parameters 

p.k_g = .1 * ones(N,1); %weight of generator
p.k_b = .1 * ones(N,1); % weight of battery charge deviation from ref
p.k_d = .5 * ones(N,1); % weight of load flexibility usage 

p.r_g = .1 * ones(N,1); %weight of generator input
p.r_b = .1 * ones(N,1); % weight of battery charging rate
p.r_d = .3 * ones(N,1); % weight of load flexibility usage rate


p.v_to_P = 10 * ones(N,1); % from speed ratio to power (when v=1 generate 10 kWh)

p.a_g = .8 * ones(N,1); % Rate of deceleration of generator spin due to friction
p.a_b = .95 * ones(N,1); % Rate of battery leakage
p.a_d = .9 * ones(N,1); % Forgetfulness rate of load delay

p.b_b = .7 * ones(N,1); % power - to - battery charging rate (efficiency)

p.max_b_charge = 15 * ones(N,1); % kWh
p.max_b_rate = .4 * p.max_b_charge;
p.min_b_rate = -p.max_b_rate;
p.max_g_rate = .3 * ones(N,1);
p.min_g_rate = -p.max_g_rate;

p.q_des = .7 * p.max_b_charge; % desired battery charge state

p.min_cum_load_delay = -2 * ones(N,1); %kWh
p.max_cum_load_delay = 2 * ones(N,1);

p.min_load_delay = p.min_cum_load_delay/10;
p.max_load_delay = p.max_cum_load_delay/10;

p.load = load;
p.collective_load_ref = collective_load_ref;

p.max_cum_load = 1.5 * p.collective_load_ref;
p.min_cum_load = 0;

end

