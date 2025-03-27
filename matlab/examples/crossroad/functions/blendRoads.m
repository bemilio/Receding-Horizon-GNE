function [roadWN, roadNE, roadES, roadSW] ...
            = blendRoads(roadSN, roadWE, blendpt1, blendpt2)
idx_blend1 = ceil(blendpt1 * length(roadWE));
idx_blend2 = ceil(blendpt2 * length(roadWE));

% Compute midpoint on WE
WE_pt_1 = roadWE(idx_blend1,:);
WE_pt_2 = roadWE(idx_blend2,:);
% Compute midpoint on SN
SN_pt_1 = roadSN(idx_blend1,:);
SN_pt_2 = roadSN(idx_blend2,:);

dx = diff(roadWE(:,1)); dy = diff(roadWE(:,2));
parallel_roadWE = [dx, dy];
parallel_roadWE = parallel_roadWE./vecnorm(parallel_roadWE,2,2); 

dx = diff(roadSN(:,1)); dy = diff(roadSN(:,2));
parallel_roadSN = [dx, dy];
parallel_roadSN = parallel_roadSN./vecnorm(parallel_roadSN,2,2); 

alpha =5; % Scaling factor for smooth transition
beta =5;  % Adjust if needed

%% Compute control points

% WN
C1_WN = WE_pt_1;
C2_WN = WE_pt_1 + alpha * parallel_roadWE(idx_blend1,:); % First control point for transition
C3_WN = SN_pt_2 - beta * parallel_roadSN(idx_blend2,:);  % Second control point for transition
C4_WN = SN_pt_2;

% NE
C1_NE = SN_pt_2;
C2_NE = SN_pt_2 - alpha * parallel_roadSN(idx_blend2,:); % First control point for transition
C3_NE = WE_pt_2 - beta * parallel_roadWE(idx_blend2,:);  % Second control point for transition
C4_NE = WE_pt_2;

% ES
C1_ES = WE_pt_2;
C2_ES = WE_pt_2 - alpha * parallel_roadWE(idx_blend2,:); % First control point for transition
C3_ES = SN_pt_1 + beta * parallel_roadSN(idx_blend1,:);  % Second control point for transition
C4_ES = SN_pt_1;

% SW
C1_SW = SN_pt_1;
C2_SW = SN_pt_1 + alpha * parallel_roadSN(idx_blend1,:); % First control point for transition
C3_SW = WE_pt_1 + beta * parallel_roadWE(idx_blend1,:);  % Second control point for transition
C4_SW = WE_pt_1;

%% Define roads
t = linspace(0, 1, 1+idx_blend2-idx_blend1); % Parameter for curve
roadWN = [roadWE(1:idx_blend1-1,:);  
          bezierCurve([C1_WN; C2_WN; C3_WN; C4_WN], t);
          roadSN(idx_blend2+1:end,:)];

roadNE = [roadSN(end:-1:idx_blend2+1,:);  
          bezierCurve([C1_NE; C2_NE; C3_NE; C4_NE], t);
          roadWE(idx_blend2+1:end,:)];

roadES = [roadWE(end:-1:idx_blend2+1,:);  
          bezierCurve([C1_ES; C2_ES; C3_ES; C4_ES], t);
          roadSN(idx_blend1-1:-1:1,:)];

%FIX
roadSW = [roadSN(1:idx_blend1-1,:);  
          bezierCurve([C1_SW; C2_SW; C3_SW; C4_SW], t);
          roadWE(idx_blend1-1:-1:1,:)];


end

