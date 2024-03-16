function [invS] = power_sum(S, order)

% This function to approximate the inverse of the (I_n - S) matrix (usually a sparse matrix)
%----------------------------
%% Input
% S: a given square matrix
% order: order of inverse approximation (>0)
%% Ouput
% invS: inverse of matrix (I_n - S)
%----------------------------

n = length(S);
invS = speye(n);
k = 1;
while k <= order
    if  k==1
        m_k = S;
    else
        m_k = m_k*S;
    end
    invS = invS + m_k;
    k = k+1;
end
