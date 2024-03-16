function simple_tr = tr_AB(A,B)

%-------------------------
% Get the trace of two compatible square matrices
%-------------------------

simple_tr = sum(A'.*B,'all');