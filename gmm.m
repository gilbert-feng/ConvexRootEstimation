function [phi_recover, se_recover,Wn,SE] = gmm(y,Wy,X,para,weight_total)

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
options = optimoptions("fsolve","Algorithm","trust-region","Display","off");
delta = fsolve(@GMM,para,options);

    % Objective function.
    function f = GMM(para)
    Sy = y;
    Wn = sparse(n, n);
    for i = 1:weight_mat_num
        Sy = Sy - para(i)*Wy(:,i);
        Wn = Wn + para(i)*weight_total{i};
    end
    S = speye(n) - Wn;
    V = Sy - X*para((weight_mat_num + 1):end);
    f = zeros(weight_mat_num + p,1);
    for i = 1:weight_mat_num
        f(i) = V'*weight_total{i}*y - V'*diag(diag(weight_total{i}/S))*V;
    end
    f((weight_mat_num+1):end) = X'*V;
    end

%----------------------------
%% recover parameters
lambda_hat = delta(1:weight_mat_num);
gamma_hat = lambda_hat/sum(lambda_hat);
beta = delta((weight_mat_num + 1):end);
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
Sigma_hat = diag(v_hat.^2);


Gamma = zeros(param_num);
Omega = zeros(param_num);
for i = 1:weight_mat_num 
    w_i_s = weight_total{i}/s;
    w_i_sXb = w_i_s*X*beta;
    Tg_i = w_i_s - diag(diag(w_i_s));
    for j = i:weight_mat_num
        w_j_s = weight_total{j}/s;
        w_j_sXb = w_j_s*X*beta;
        Tg_j = w_j_s - diag(diag(w_j_s));
        % elements of lambda_hat for Gamma
        Gamma(j,i) = tr_AB(w_i_s'*(Tg_j+Tg_j'),Sigma_hat)+w_i_sXb'*w_j_sXb;
        % elements of lambda_hat for Omega
        Omega(j,i) = tr_AB(Tg_j*Sigma_hat,(Tg_i+Tg_i')*Sigma_hat)+w_j_sXb'*Sigma_hat*w_i_sXb;
    end
    
    % elements of beta for Gamma
    Gamma((weight_mat_num + 1):end, i) = X'*w_i_sXb;
    % elements of beta for Omega
    Omega((weight_mat_num + 1):end, i) = X'*Sigma_hat*w_i_sXb;
end


Omega((weight_mat_num + 1):end,(weight_mat_num + 1):end) = X'*Sigma_hat*X;
Gamma((weight_mat_num + 1):end,(weight_mat_num + 1):end) = X'*X;
temp_Gamma = Gamma - diag(diag(Gamma));
temp_Gamma = temp_Gamma';
Gamma = Gamma + temp_Gamma;
temp_Omega = Omega - diag(diag(Omega));
temp_Omega = temp_Omega';
Omega = Omega + temp_Omega;
SE = (Gamma\Omega)/(Gamma');

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
