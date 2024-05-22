

n_x_per_agent = 3;
for t = 1:T_sim
    indexes_velocity_gen = 1:n_x_per_agent:n_x;
    cum_gen_power = param.v_to_P'*x_ol(indexes_velocity_gen, t);
    cum_battery_power = sum(u_ol(2,:,:,t));
    cum_load = sum(load) + sum(u_ol(3,:,:,t));
    cum_consumed_power_at_grid(t) = cum_load + cum_battery_power - cum_gen_power;
end

% Plot the sequence
figure; % Create a new figure window
plot(cum_consumed_power_at_grid, '-o');
xlabel('t');
ylabel('Power main grid');
grid on;