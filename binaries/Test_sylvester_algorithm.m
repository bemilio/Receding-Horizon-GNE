clc 
clear all

n_x=5;
n_u = 3;
N = 5;

for i =1:1000
    is_controllable = false;
    while ~is_controllable
        A = rand(N, n_x, n_x);
        B = rand(N, n_x, n_u);
        Ctrb = ctrb(A,B);
        if rank(Ctrb) == n_x
            is_controllable = true;
        end
    end
    Q = rand(N, n_x, n_u);
    R = rand(N, n_x, n_u);
    for i = 1:N
        Q(i,:,:) = Q(i,:,:) + Q(i,:,:)';
        Q(i,:,:) = Q(i,:,:) - eye(n_x) * min(eig(Q(i,:,:))) + eye(n_x) * 0.1;
        R(i,:,:) = R(i,:,:) + R(i,:,:)';
        R(i,:,:) = R(i,:,:) - eye(n_u) * min(eig(R(i,:,:))) + eye(n_u) * 0.1;
    end
    is_initialization_stable = false;
    while ~is_initialization_stable
        K = rand(N, n_u, n_x);
        if max(abs(eig(A + sum(B * K))))
    end

    % Solve OL-NE
    for k=1:1000
        A_cl = A + sum(B)
        for i=1:N
            P[i] = 
        end
    end

    
    K = -K;
    I_x = eye(n_x);
    A_cl_T = ( A' * (I_x - B * inv(R + B' * P * B) * B' * P )' );
    
    S = B * inv(R) * B';
    
    A_test = A' * (I_x - P * inv(I_x + S * P) * S);
    
    A_test_2 = A' - A_cl_T * P * S;
    
    if norm(A_cl_T - A_test) < 10^(-5) && norm(A_cl_T - A_test_2) < 10^(-5)
        disp("ok")
    else
        disp("not ok")
    end
end