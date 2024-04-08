function predmod=genPredModel(A, B, T_hor)

%Prediction matrices generation
%This function computes the prediction matrices to be used in the
%optimization problem
nx = size(A,1);
nu = size(B,2);
%Prediction matrix from initial state
T=zeros(nx*(T_hor),nx);
for k=1:T_hor
    T((k-1)*nx+1:k*nx,:)=A^k;
end

%Prediction matrix from input
N = size(B,3);
S=zeros(nx*(T_hor),nu*(T_hor), N);
for idx_agent=1:N
    for k=1:T_hor
        for i=1:k
            S((k-1)*nx+1:k*nx,(i-1)*nu+1:i*nu, idx_agent)=A^(k-i)*B(:,:,idx_agent);
        end
    end
end
predmod.S = S;
predmod.T = T;

end
