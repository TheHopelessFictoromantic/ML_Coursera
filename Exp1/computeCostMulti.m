function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y
% Initialize some useful values

m= length(y);
error_sum_square=0;
for k= 1:m
    prediction= X*theta;
    error(k)= prediction(k)- y(k);
end
error_sum_square= sum(error.^2);
J= (1/(2*m))*error_sum_square; 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
% =========================================================================

end