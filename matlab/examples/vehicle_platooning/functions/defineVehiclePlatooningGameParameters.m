function [p] = defineVehiclePlatooningGameParameters()

%% Define Parameters 
% for details see IFAC conference Shi/Lazar 2017

p.T_sampl = .01;
p.headway_time = 1; % factors in safety distance dependancy on speed, seconds

p.v_des_1 = 30; %desired speed of leading vehicle, m/s
p.d_des = 100; % Desired distance, not dependent on speed, m

end

