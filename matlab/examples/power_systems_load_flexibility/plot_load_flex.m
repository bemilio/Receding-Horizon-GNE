close all

n_x_per_agent = 3;
for t = 1:T_sim
    indexes_velocity_gen = 1:n_x_per_agent:n_x;
    cum_gen_power = param.v_to_P'*x_ol(indexes_velocity_gen, t);
    cum_battery_power = sum(u_ol(2,:,:,t));
    cum_load = sum(grid_load_series(t,:)) + sum(u_ol(3,:,:,t));
    cum_consumed_power_at_grid_ol(t) = cum_load + cum_battery_power - cum_gen_power;
end
if run_cl
    for t = 1:T_sim
        indexes_velocity_gen = 1:n_x_per_agent:n_x;
        cum_gen_power = param.v_to_P'*x_cl(indexes_velocity_gen, t);
        cum_battery_power = sum(u_cl(2,:,:,t));
        cum_load = sum(load) + sum(u_cl(3,:,:,t));
        cum_consumed_power_at_grid_cl(t) = cum_load + cum_battery_power - cum_gen_power;
    end
end

% Plot the sequence
figure; % Create a new figure window
plot(cum_consumed_power_at_grid_ol, '-o', 'DisplayName', "OL-NE");
hold on
plot(sum(grid_load_series'), '-o', 'DisplayName', "Nom. cum. load");

if run_cl
    plot(cum_consumed_power_at_grid_cl, '-o');
end 
xlabel('t');
ylabel('Power main grid');
grid on;
legend
hold off

figure
plot(squeeze(x_ol(1,1,:))'*100, '-o', 'DisplayName', "Gen. speed");
hold on
plot(squeeze(x_ol(2,1,:))'*100/param.max_b_charge(1), '-o', 'DisplayName', "Batt. charge");
plot(squeeze(x_ol(3,1,:))'*100/param.max_load_delay(1), '-o', 'DisplayName', "Cumul. load deferral");

xlabel('t');
ylabel('(%)');
grid on;
legend

