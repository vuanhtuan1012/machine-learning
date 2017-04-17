function H = hypothesis(X, theta)
% hypothesis function
% compute value of hypothesis function
% input: X, theta0, theta1
% output: H

X0 = ones(length(X), 1);
H = [X0 X]*theta;