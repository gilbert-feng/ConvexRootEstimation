function [phi, Wy] = est_initial(n,p,X,y,weight_mat_num,weight_total)

% This is the function for initial IV estimation.
%----------------------------
%% Input
% n: sample number
% p: exogenous variable's dimension
% X: exogenous variable
% y: dependent variable
% weight_mat_num: number of weight matrices
% weight_total: total weight matrices cell
%% Ouput
% phi: estimated parameters
% Wy: the composite of weight matrices and the dependent variable (for later estimation)

%% Initial consistency estimator (2SLSE)
WX = zeros(n, p*weight_mat_num*2); % [WX, W2X]
Wy = zeros(n, weight_mat_num);
for i = 1:weight_mat_num
    WX(:, (1+(i-1)*p):(i*p)) = weight_total{i}*X;
    WX(:, p*weight_mat_num + ((1+(i-1)*p):(i*p))) = weight_total{i}*WX(:, (1+(i-1)*p):(i*p));
    Wy(:, i) = weight_total{i}*y;
end
Q = [X, WX];
QZ = Q'*[Wy, X];
QQ = Q'*Q;
phi = (QZ'*(QQ\QZ))\(QZ'*(QQ\(Q'*y)));
