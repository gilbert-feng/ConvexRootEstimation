function [impact_mat, impact_sd_mat] = impact(X,phi,SE,Wn,beta_ind,order,weight_total)

% This is the function for root estimation.
%----------------------------
%% Input
% X: exogenous variable
% phi: given estimator
% SE: covariance matrix
% Wn: the aggregated weight matrix (to simplify computation)
% order: order of inverse approximation (>=0, 0: no approximation)
% weight_total: weight matrices
%% Ouput
% impact_mat: estimated impact measure
% impact_sd_mat: estimated impact measure's sd
%----------------------------

%% Preliminary setting
weight_mat_num = length(weight_total);
[n,~] = size(X);
if order == 0
    s = speye(n) - Wn;
    inv_s = s\speye(n);
else
    inv_s = power_sum(Wn, order);
end

%% Impact Measure
impact_mat = zeros(length(beta_ind), 3);
impact_sd_mat = zeros(length(beta_ind), 3);
temp_ADI_sd_mat = zeros(1,weight_mat_num + 1);
temp_ATI_sd_mat = zeros(1,weight_mat_num + 1);
temp_AII_sd_mat = zeros(1,weight_mat_num + 1);
beta_ind = beta_ind + (weight_mat_num + 1); % (start from weight_mat_num + 2)
temp_counter = 1;
for i = beta_ind
    %% impact measure
    impact_mat(temp_counter, 1) = 1/n*trace(inv_s)*phi(i); %ADI
    impact_mat(temp_counter, 2) = 1/n*sum(inv_s, 'all')*phi(i); %ATI
    impact_mat(temp_counter, 3) = impact_mat(temp_counter, 2) - impact_mat(temp_counter, 1); %AII

    %% impact measure std
    for j = 1:weight_mat_num
        temp_ADI_sd_mat(j) = 1/n*tr_AB((inv_s*weight_total{j}),inv_s)*phi(i);
        temp_ATI_sd_mat(j) = 1/n*phi(i)*sum(inv_s*weight_total{j}*inv_s, 'all');
        temp_AII_sd_mat(j) = temp_ATI_sd_mat(j) - temp_ADI_sd_mat(j);
    end
    temp_ADI_sd_mat(weight_mat_num + 1) = 1/n*trace(inv_s);
    temp_ATI_sd_mat(weight_mat_num + 1) = 1/n*sum(inv_s, 'all');
    temp_AII_sd_mat(weight_mat_num + 1) = temp_ATI_sd_mat(weight_mat_num + 1) - temp_ADI_sd_mat(weight_mat_num + 1);

    impact_sd_mat(temp_counter, 1) = sqrt(temp_ADI_sd_mat*SE([1:weight_mat_num,i-1],[1:weight_mat_num,i-1])*temp_ADI_sd_mat'); % no elements of lambda_c
    impact_sd_mat(temp_counter, 2) = sqrt(temp_ATI_sd_mat*SE([1:weight_mat_num,i-1],[1:weight_mat_num,i-1])*temp_ATI_sd_mat');
    impact_sd_mat(temp_counter, 3) = sqrt(temp_AII_sd_mat*SE([1:weight_mat_num,i-1],[1:weight_mat_num,i-1])*temp_AII_sd_mat');

    temp_counter = temp_counter + 1;
end