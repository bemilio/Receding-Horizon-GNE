load 15_minute_data_new_york.mat
close all
T = minutedatanewyork;
T.local_15min = string(T.local_15min);
T.local_15min = datetime(T.local_15min, 'InputFormat', 'yyyy-MM-dd HH:mm:ssXXX', 'TimeZone', 'local');

% Get unique user IDs
userIDs = unique(T.dataid);

% Initialize a cell array to store the results
numUsers = 25;
userDataCell = {};


% Loop through each user ID and process the data
for i = 1:numUsers
    % Extract the data for the current user
    userData = T(T.dataid==userIDs(i), :);
     % Check if all data is positive
    % Sort the user data by timestamp
    userData = sortrows(userData, 'local_15min');
    
    % Store the relevant numerical data in a field named after the user ID
    userDataCell{i} = -1*userData.grid;
    userTimestampCell{i} = userData.local_15min;
end

minLength = min(cellfun(@length, userDataCell));
% Cut the data for each user to match the minimum length
for i = 1:numUsers
    userDataCell{i} = userDataCell{i}(1:minLength);
    userTimestampCell{i} = userTimestampCell{i}(1:minLength);
end


% Combine the data from all users
combinedData = [];

for i = 1:numUsers
    combinedData = [combinedData; userDataCell{i}'];
end

sum_values = sum(combinedData);

% Filter data for the first two days
startDateTime = min(userTimestampCell{1});
endDateTime = startDateTime + days(2);
indexesFirstTwoDays = find(userTimestampCell{1} >= startDateTime & userTimestampCell{1} <= endDateTime);
grid_load_data = table('Size', [length(indexesFirstTwoDays), 2], 'VariableTypes', {'datetime', 'double'}, 'VariableNames', {'Timestamp', 'Agg_Power'});
grid_load_data.Timestamp = userTimestampCell{1}(indexesFirstTwoDays);
grid_load_data.Agg_Power = sum_values(indexesFirstTwoDays)';
for i = 1:numUsers
    variableName = ['Load_', num2str(i)];  % Generate variable name
    grid_load_data.(variableName) = combinedData(i,indexesFirstTwoDays)';          % Add empty column with the generated variable name
end

% Plot the data
figure;
plot(grid_load_data.Timestamp, grid_load_data.Agg_Power, '-o');
xlabel('Timestamp');
ylabel('Aggregate Power');
grid on


save("new_york_parsed_housholds.mat", "grid_load_data")