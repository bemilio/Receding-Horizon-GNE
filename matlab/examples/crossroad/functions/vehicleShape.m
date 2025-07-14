function vehicleShape = vehicleShape(angle,vehicleLength, vehicleWidth)
    % Rotated vehicle shape
    R = [cos(angle), -sin(angle); sin(angle), cos(angle)];
    vehicleShape = R * [ ...
        -vehicleLength/2, vehicleLength/2, vehicleLength/2, -vehicleLength/2;
        -vehicleWidth/2, -vehicleWidth/2, vehicleWidth/2, vehicleWidth/2];

end

