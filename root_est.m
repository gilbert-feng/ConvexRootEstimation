function [phi_recover,se_recover,Wn,SE] = root_est(y, X, homo, order,initial_est, weight_total)

% This is the function for root estimation.
%----------------------------
%% Input
% y: dependent variable
% X: exogenous variable
% homo: disturbance if == 1, homo; O/W, hete
% order: order of inverse approximation (>=0, 0: no approximation)
% beta_ind: target beta index vector for impact measure (start from 1)
% initial_est: the given initial estimator (0: estimate with IV method)
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
XTX = X'*X;

%----------------------------
%% Initial consistency estimator (2SLSE)
if initial_est == 0
    phi_Best_t = est_initial(n,p,X,y,weight_mat_num,weight_total);
else 
    phi_Best_t = initial_est;
end

%% Compute some vectors and matrices needed for the root estimator.
Wn = sparse(n, n);
for i = 1:weight_mat_num
    Wn = Wn + phi_Best_t(i)*weight_total{i};
end
s = speye(n) - Wn;
if order == 0
    inv_s = s\speye(n);
else
    inv_s = power_sum(Wn, order);
end
% inv_s = pseudoinverse(s);
% inv_s = inv_s*speye(n);
% inv_s = inv_s.*(abs(inv_s)>1e-6); % approximate
Xb = X * phi_Best_t((weight_mat_num + 1):end);
V = s*y - Xb;
resTyXb = y - Xb;
w_y = zeros(n, weight_mat_num);
for i = 1:weight_mat_num
    w_y(:,i) = weight_total{i}*y;
    resTyXb = resTyXb - phi_Best_t(i)*w_y(:,i);
end


%----------------------------
%% Compute the root estimator of lambda.
dvT = zeros(n,weight_mat_num + p); %the derivative vector
for i = 1:weight_mat_num 
    dvT(:,i) = w_y(:,i);
end
dvT(:,(weight_mat_num + 1):end) = X;
dvT = - dvT;

%%% intialize data structure
temp_lambda_index = 1:weight_mat_num;
deriv_num = param_num - 1;
temp_C = zeros(deriv_num);
temp_a = zeros(deriv_num, 1);
temp_b = zeros(deriv_num, 1);
temp_c = zeros(deriv_num, 1);
temp_d = zeros(deriv_num, 1);
lambda_hat = zeros(1,weight_mat_num);
sXb = inv_s*Xb;

% pre calculation
w_sXb = zeros(n, weight_mat_num);
w_inv_s = cell(1, weight_mat_num);
Tg = cell(1,weight_mat_num);
TgS = cell(1,weight_mat_num);
for i = temp_lambda_index
    w_sXb(:,i) = weight_total{i}*sXb;
    w_inv_s{i} = weight_total{i}*inv_s;
    if homo == 1 
        Tg{i} = w_inv_s{i} - 1/n*trace(w_inv_s{i})*speye(n);
        TgS{i} = Tg{i} + Tg{i}';
    else
        Tg{i} = w_inv_s{i} - diag(diag(w_inv_s{i}));
        TgS{i} = Tg{i} + Tg{i}'; % symmetrized
    end
end

for i = temp_lambda_index % compute lambda_i

    dv = dvT(:, 1:(deriv_num+1)~=i); % compute the derivative vector for lambda_i: n*(weight_mat_num - 1 + p)
    
    temp_C(weight_mat_num:end,:) = X'*dv;
    temp_b(weight_mat_num:end,:) = X'*w_y(:,i);
    resyXb = resTyXb + phi_Best_t(i)*w_y(:,i);
    temp_c(weight_mat_num:end,:) = X'*resyXb;
    temp_d(weight_mat_num:end,:) = X'*w_y(:,i);
    inner_ind = 1;
    
    for k = temp_lambda_index(temp_lambda_index ~= i) % compute components of the quadratic equations (except lambda_i)
        temp_C(inner_ind,:) = (TgS{k}*V)'*dv + w_sXb(:,k)'*dv;
        temp_a(inner_ind,:) = (TgS{k}*w_y(:,i))'*w_y(:,i)/2;
        temp_b(inner_ind,:) = (TgS{k}*w_y(:,i))'*resyXb + w_y(:,i)'*w_sXb(:,k);
        temp_c(inner_ind,:) = resyXb'*Tg{k}*resyXb + resyXb'*w_sXb(:,k);
        temp_d(inner_ind,:) = (TgS{k}*w_y(:,i))'*V + w_y(:,i)'*w_sXb(:,k);
        inner_ind = inner_ind + 1;
    end
    C1 = ((TgS{i}*V)'*dv + w_sXb(:,i)'*dv)/temp_C;
    a1 = (TgS{i}*w_y(:,i))'*w_y(:,i)/2 - C1*temp_a;
    b1 = -((TgS{i}*w_y(:,i))'*resyXb + w_y(:,i)'*w_sXb(:,i)) + C1*temp_b;
    c1 = resyXb'*Tg{i}*resyXb + resyXb'*w_sXb(:,i) - C1*temp_c;
    d1 = (TgS{i}*w_y(:,i))'*V + w_y(:,i)'*w_sXb(:,i) - C1*temp_d;
    
    det = b1^2-4*a1*c1;
    if det <= 0
        lambda1_hat = -b1/(2*a1);
    else
        lambda1_hat = ((-b1-sqrt(det))/(2*a1))*(d1>0) + ((-b1+sqrt(det))/(2*a1))*(d1<=0);
    end

    lambda_hat(i) = lambda1_hat;
end

%----------------------------
%% Compute beta.
TWy = zeros(n,1);
Wn = sparse(n, n);
for i = 1:weight_mat_num
    TWy = TWy + lambda_hat(i)*w_y(:,i);
    Wn = Wn + lambda_hat(i)*weight_total{i};
end
temp_wy = y - TWy;
beta_hat = (XTX)\(X'*temp_wy);
gamma_hat = lambda_hat/sum(lambda_hat);
phi_hat = [sum(lambda_hat);gamma_hat';beta_hat]; 

%----------------------------
%% Compute standard error.
s = speye(n) - Wn;
if order == 0
    inv_s = s\speye(n);
else
    inv_s = power_sum(Wn, order);
end
% inv_s = pseudoinverse(s);
% inv_s = inv_s*speye(n);
% inv_s = inv_s.*(abs(inv_s)>1e-6); % approximate
Xb = X*beta_hat;
sXb = inv_s*Xb;

v_hat = temp_wy - Xb;
v_sq = (v_hat.^2)';
Sigma_hat = diag(v_sq);
[sige,mu3,mu4] = deal(mean(v_hat.^2),mean(v_hat.^3),mean(v_hat.^4));
dev = mu4-3*sige^2;

Gamma = zeros(param_num);
Omega = zeros(param_num);

% pre calculation
w_sXb = zeros(n, weight_mat_num);
w_inv_s = cell(1, weight_mat_num);
Tg = cell(1,weight_mat_num);
TgS = cell(1,weight_mat_num);
Tgdiag = zeros(n, weight_mat_num);
TgSigma = cell(1,weight_mat_num);
for i = temp_lambda_index
    w_sXb(:,i) = weight_total{i}*sXb;
    w_inv_s{i} = weight_total{i}*inv_s;
    if homo == 1 
        Tg{i} = w_inv_s{i} - 1/n*trace(w_inv_s{i})*speye(n);
        TgS{i} = Tg{i} + Tg{i}';
    else
        Tg{i} = w_inv_s{i} - diag(diag(w_inv_s{i}));
        TgS{i} = Tg{i} + Tg{i}'; % symmetrized
    end
    Tgdiag(:,i) = diag(Tg{i});
    TgSigma{i} = bsxfun(@times, TgS{i}, v_sq); % v_sq is a row vector, then each col of TgS{j} is times by v_sq
end

for i = 1:weight_mat_num 
    for j = i:weight_mat_num
        % elements of lambda_hat for Gamma
        Gamma(j,i) = sum(w_inv_s{i}.*TgSigma{j}, 'all') + (w_sXb(:,i))'*w_sXb(:,j);

        % elements of lambda_hat for Omega
        if homo == 1
            Omega(j,i) = sige^2*tr_AB(Tg{j},TgS{i}) + w_sXb(:,j)'*w_sXb(:,i)*sige + dev*Tgdiag(:,i)'*Tgdiag(:,j) + mu3*w_sXb(:,i)'*Tgdiag(:,j) + mu3*w_sXb(:,j)'*Tgdiag(:,i);
        else
            Omega(j,i) = tr_AB(TgSigma{j},TgSigma{i})/2 + sum(w_sXb(:,j).*v_sq'.*w_sXb(:,i));
        end

    end
    
    % elements of beta for Gamma
    Gamma((weight_mat_num + 1):end, i) = X'*w_sXb(:,i);
    % elements of beta for Omega
    if homo == 1
        Omega((weight_mat_num + 1):end, i) = X'*(sige*w_sXb(:,i) + mu3*Tgdiag(:,i));
    else
        Omega((weight_mat_num + 1):end, i) = X'*(Sigma_hat*w_sXb(:,i));
    end
end

if homo == 1
    Omega((weight_mat_num + 1):end,(weight_mat_num + 1):end) = sige*(XTX);
else
    Omega((weight_mat_num + 1):end,(weight_mat_num + 1):end) = X'*Sigma_hat*X;
end

Gamma((weight_mat_num + 1):end,(weight_mat_num + 1):end) = XTX;
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


