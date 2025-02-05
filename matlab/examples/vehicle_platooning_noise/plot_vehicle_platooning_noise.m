close all

% Assuming u_ol is of size (n_u, 1, N, T_sim, N_tests)
[n_u, ~, N, T_sim, N_tests] = size(u_ol);

n_x_per_agent = 2;
p_ol = zeros(T_sim, N);
v_ol = zeros(T_sim, N);
p_cl = zeros(T_sim, N);
v_cl = zeros(T_sim, N);
%% Conversion from error variables to actual speed/position
for t = 1:T_sim
    [p_ol(t,:), v_ol(t,:)] = convertStateToPosVel(x_ol(:,1,t),...
                param.v_des_1, param.d_des, param.headway_time);
end
if run_cl
    for t = 1:T_sim
        [p_cl(t,:), v_cl(t,:)] =...
            convertStateToPosVel(x_ol(:,1,t),...
                param.v_des_1, param.d_des, param.headway_time);
    end
end


%% Boxplot of input deviation from nominal when noise is introduced to term. cost
figure;

% Get the number of variance values tested

% Extract baseline values (test 1)
baseline = u_ol(:,:,:,:,1); % Size: (n_u, 1, N, T_sim)

% Compute the norm of each T_sim-length vector in (n_u, N)
baseline_reshaped = reshape(baseline, n_u * N, T_sim); % Size: (n_u*N, T_sim)
baseline_norms = vecnorm(baseline_reshaped, 2, 2); % Size: (n_u*N, 1)

% Initialize matrix to store normalized sums
norm_ratios = zeros(N_tests-1, 1);

% Loop over tests (excluding test 1)
for test_idx = 2:N_tests
    % Compute the difference from baseline
    diff = u_ol(:,:,:,:,test_idx) - baseline;

    % Reshape to (n_u*N, T_sim)
    diff_reshaped = reshape(diff, n_u * N, T_sim);

    % Compute vector 2-norm for each row (each sequence of length T_sim)
    diff_norms = vecnorm(diff_reshaped, 2, 2); % Size: (n_u*N, 1)

    % Normalize by baseline norm (element-wise division)
    normalized_diff = diff_norms ./ baseline_norms; % Size: (n_u*N, 1)

    % Sum over all n_u*N elements to get a single scalar
    norm_ratios(test_idx-1) = sum(normalized_diff);
end

% Group results by parameter
num_tested_variance = length(variance_to_test);
grouped_data = cell(num_tested_variance, 1);
for i = 1:num_tested_variance
    grouped_data{i} = norm_ratios(test_variance(2:end) == variance_to_test(i));
end

% Convert to box plot format
grouped_data = cell2mat(grouped_data'); %transpose?

% Create box plot
figure;
boxplot(grouped_data, variance_to_test, 'Labels', string(variance_to_test));
xlabel('Parameter Value');
ylabel('Normalized Sum of Norm Differences');
title('Boxplot of Normalized Sum of Differences by Parameter');
grid on;

print('box_plot_noise_P.png', '-dpng', '-r600');  % Save with 600 dpi resolution


