function [phi_recover, se_recover,Wn,SE] = qmle(y,Wy,X,para,weight_total)

% This is the function for moment-based estimation.
%----------------------------
%% Input
% y: dependent variable
% Wy: the composite of weight matrices and the dependent variable
% X: the exogenous variable
% para: initial estimation
% weight_total: weight matrices
%% Ouput
% phi_recover: recovered estimated parameters
% se_recover: recovered parameters' se
% Wn: aggregated weight matrix (for impact measure)
% SE: covariance matrix (for impact measure)
%----------------------------

%% Preliminary setting
weight_mat_num = length(weight_total);
[n,p] = size(X);
param_num = p + weight_mat_num;

%----------------------------
%% Preliminary setting
options = optimoptions('fminunc','Algorithm','quasi-newton','Display','notify');
lambda_hat = fminunc(@CMLE,para,options);

    % Concentrated objective function.
    function fval = CMLE(para)
    Sy = y;
    Wn = sparse(n, n);
    for i = 1:weight_mat_num
        Sy = Sy - para(i)*Wy(:,i);
        Wn = Wn + para(i)*weight_total{i};
    end
    beta = (X'*X)\X'*Sy;
    fval = n/2*log(mean((Sy-X*beta).^2))- log(abs(det((speye(n) - Wn))));
    end

%----------------------------
%% recover parameter 
Sy = y;
for i = 1:weight_mat_num
    Sy = Sy - lambda_hat(i)*Wy(:,i);
end
beta = (X'*X)\X'*Sy;
gamma_hat = lambda_hat/sum(lambda_hat);
phi_hat = [sum(lambda_hat);gamma_hat;beta]; 

%----------------------------
%% Compute standard error.
Sy = y;
Wn = sparse(n, n);
for i = 1:weight_mat_num
    Sy = Sy - lambda_hat(i)*Wy(:,i);
    Wn = Wn + lambda_hat(i)*weight_total{i};
end
s = speye(n) - Wn;
Xb = X*beta;
v_hat = Sy - Xb;
[sige,mu3,mu4]=deal(mean(v_hat.^2),mean(v_hat.^3),mean(v_hat.^4));
dev = mu4 - 3*sige^2;

Hessian = zeros(param_num + 1);
Omega = zeros(param_num + 1);
for i = 1:weight_mat_num 
    G_i = weight_total{i}/s;
    for j = i:weight_mat_num
        G_j = weight_total{j}/s;
        % elements of lambda_hat for Hessian
        Hessian(j,i) = (G_i*Xb)'*(G_j*Xb) + sige*tr_AB((G_i + G_i'),G_j);
        % elements of lambda_hat for Omega
        if j == i
            Omega(j,i) = dev*diag(G_i)'*diag(G_j) + 2*mu3*(G_i*Xb)'*diag(G_i);
        else
            Omega(j,i) = dev*diag(G_i)'*diag(G_j) + mu3*((G_i*Xb)'*diag(G_j) + (G_j*Xb)'*diag(G_i));
        end
    end
    
    % elemtns of beta and sige for Hessian
    Hessian((weight_mat_num + 1):(end - 1), i) = X'*(G_i*Xb);
    Hessian(end, i) = trace(G_i);
    % elemtns of beta and sige for Omega
    Omega((weight_mat_num + 1):(end - 1), i) = mu3*X'*diag(G_i);
    Omega(end, i) = 1/2*(dev*trace(G_i) + mu3*sum(G_i*Xb));
end
Hessian((weight_mat_num + 1):(end - 1),(weight_mat_num + 1):(end - 1)) = X'*X;
Hessian(end, end) = n/(2*sige);
temp_Hessian = Hessian - diag(diag(Hessian));
temp_Hessian = temp_Hessian';
Hessian = Hessian + temp_Hessian;
Hessian = 1/(sige)*Hessian;

Omega(end,(weight_mat_num + 1):(end - 1)) = mu3/(2*sige)*zeros(1,n)*X;
Omega(end, end) = n*dev/(4*sige^2);
temp_Omega = Omega - diag(diag(Omega));
temp_Omega = temp_Omega';
Omega = Omega + temp_Omega;
Omega = 1/(sige^2)*Omega;
SE = Hessian\(Hessian + Omega)/Hessian;
SE(end,:) = [];
SE(:,end) = [];

%----------------------------
%% recover parameter 
temp_g = 1/phi_hat(1)*eye(weight_mat_num);
for i = 1:weight_mat_num
    temp_g(i,:) = temp_g(i,:) - lambda_hat(i)/(phi_hat(1)^2);
end
temp_g = [ones(1,weight_mat_num);temp_g];
temp_g = blkdiag(temp_g, eye(p));
se_recover = sqrt(abs(diag(temp_g*SE*temp_g')));
phi_recover = phi_hat;

end

