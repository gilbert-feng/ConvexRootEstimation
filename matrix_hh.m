function [w]=matrix_hh(n, k1, k2)
% This function generates a matrix, where for the first group of spatial
% units, every unit is connected to the its nearest 2*k1 neighbors, and for
% the second group of spatial units, every unit is connected to its nearest
% 2*k2 neighbors.
% w is normalized to have row sums equal to one. 
% The elements of w_o are either 0 or 1.
n1 = round(n/3);

r1 = kron((1:n1)',ones(2*k1,1)); % Row indexes for nonzero elements.
r2 = kron((n1+1:n)',ones(2*k2,1)); % Row indexes for nonzero elements.

% Row 1.
c1 = [2:(k1+1),(n-k1+1):n]'; % Column indexes for nonzero elements.

% Rows 2 to k1.
c2 = zeros(2*k1,k1-1);
for i = 2:k1
    c2(:,i-1) = [1:(i-1),(i+1):(i+k1),(n-k1+i):n];
end

% Rows k1+1 to n1.
c3 = zeros(2*k1,n1-k1);
for i = (k1+1):n1
    c3(:,i-k1) = [(i-k1):(i-1),(i+1):(i+k1)];
end

% Rows n1+1 to n-k2.
c4 = zeros(2*k2,n-k2-n1);
for i = (n1+1):(n-k2)
    c4(:,i-n1) = [i-k2:i-1,i+1:i+k2];
end

% Rows n-k2+1 to n.
c5 = zeros(2*k2,k2);
for i = (n-k2+1):n
    c5(:,i-(n-k2)) = [1:(k2-n+i),i-k2:i-1,i+1:n];
end

w_o = sparse([r1;r2],[c1;c2(:);c3(:);c4(:);c5(:)],1);
w = spdiags(1./sum(w_o,2),0,n,n)*w_o;

