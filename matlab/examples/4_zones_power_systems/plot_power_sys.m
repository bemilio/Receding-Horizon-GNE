% Create figure
close all
figure;

time = 1:T_sim+1;

% First subplot
delta_omega1 = zeros(length(time),1);
delta_omega1(:) = x_cl(1,1,:);
subplot(2,2,1);
plot(time, delta_omega1);
xlabel('Time (s)');
ylabel('\Delta \omega_1');
title('\Delta \omega_1 vs. Time');

% Second subplot
subplot(2,2,2);
delta_omega2 = zeros(length(time),1);
delta_omega2(:) = x_cl(4,1,:);
plot(time, delta_omega2);
xlabel('Time (s)');
ylabel('\Delta P_{ref}_1');
title('\Delta P_{ref}_1 vs. Time');

% Third subplot
% subplot(2,2,3);
% 
% plot(time, delta_ptie12);
% xlabel('Time (s)');
% ylabel('\Delta P_{tie}^{12}');
% title('\Delta P_{tie}^{12} vs. Time');
% 
% % Fourth subplot
% subplot(2,2,4);
% plot(time, delta_pref2);
% xlabel('Time (s)');
% ylabel('\Delta P_{ref}_2');
% title('\Delta P_{ref}_2 vs. Time');