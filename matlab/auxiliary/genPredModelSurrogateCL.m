function predmod=genPredModelSurrogateCL(A, B, K, T_hor)
%Prediction matrices generation
%This function computes the prediction matrices to be used in the
% surrogate CL-NE problem. Each agent assumes that the system 
% evolves at time t=1 as
% x_i[1] = A x_0 + B_i u_i[0] + sum_j~i B_j u_j[0]
% (Each agent has a different prediction of the state evolution x_i.)
% For each subsequent timestep, the prediction evolves as
% x_i[t+1] = Ax_[t] + sum_j~i B_jK_j x[t] + B_iu_i[t]

nx = size(A,1);
nu = size(B,2);
N = size(B,3);
A_cl_not_i = zeros(nx, nx, N);
for i=1:N
    A_cl_not_i(:,:,i) = A + sum(pagemtimes(B, K), 3) - B(:,:,i) * K(:,:,i);
end
%Prediction matrix from initial state
% T_i = col(A_cl_not_i^(tau-1) * A), with tau =1,...,T_hor 
T=cell(N,1);
for i=1:N
    T{i}=zeros(nx*(T_hor),nx);
    A_cl_not_i_power = eye(nx);
    for k=1:T_hor
        indexes = (k-1)*nx+1:k*nx;
        T{i}(indexes,:)=A_cl_not_i_power * A;
        A_cl_not_i_power = A_cl_not_i_power * A_cl_not_i(:,:,i);
    end
end

%Prediction matrix from input of all agents
S=cell(N,1);
for idx_agent=1:N
    S{idx_agent} = zeros(nx*T_hor,nu*T_hor, N);
    A_cl_not_i_diag_stack = [];
    for t=1:T_hor
        % COmpute here, use later
        A_cl_not_i_diag_stack = blkdiag(A_cl_not_i_diag_stack, A_cl_not_i(:,:,idx_agent));
    end
    % Dependence of prediction of agent idx_agent on the input of agent
    % idx_agent itself
    % [ B_i                  0                    0 0 ... 0 ;
    %   A_cl_not_i*B_i       B_i                  0 0 ... 0 ;
    %   ...
    %   A_cl_not_i^(T-1)*B_i A_cl_not_i^(T-2)*B_i ...     B_i]
    for k=1:T_hor
        indexes_x = (k-1)*nx+1:k*nx;
        for tau=1:k
            indexes_u = (tau-1)*nu+1:tau*nu;
            S{idx_agent}(indexes_x,indexes_u, idx_agent)=A_cl_not_i(:,:,idx_agent)^(k-i)*B(:,:,idx_agent);
        end
    end
    % Dependence of prediction of agent idx_agent on the input of remaining
    % agents. Notice this has a different struucture from the previous one!
    % [ B_j                  0     0 ;
    %   A_cl_not_i*B_j       0     0 ;
    %   ...
    %   A_cl_not_i^(T-1)*B_j 0 ... 0]
    % The zeros are because agent i's prediction only depends from the first input of
    % ajent j. The remaiing timesteps are predicted by the dynamics in
    % closed loop with K_cl.
    for j=1:N
        if j~=idx_agent            
            A_cl_not_i_power = eye(nx);
            for k=1:T_hor
                indexes = (k-1)*nx+1:k*nx;
                S{idx_agent}(indexes,1:nu,j)=A_cl_not_i_power * B(:,:,j);
                A_cl_not_i_power = A_cl_not_i_power * A_cl_not_i(:,:,idx_agent);
            end
        end
    end   
end
predmod.S = S;
predmod.T = T;

end
