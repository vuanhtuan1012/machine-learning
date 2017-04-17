function j = getJ(X, Y, theta)
% getJ function
% compute value of cost function J
% input: X, Y, theta0, theta1
% output: j

H = hypothesis(X, theta);
j = meansqr(H-Y)/2;