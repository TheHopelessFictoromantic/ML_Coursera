function J = computeCost(X, y, theta)

%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values

m= length(y);
J=0;
error_sum_square=0;
for i= 1:m
    prediction(i)= theta(1) + (theta(2).*X(i,2)); 
    % here, we are doing X(i,2) because we always add a default one as the first column in 
    %our input feature which is usually omitted while findng the cost. we have to perform
    %element wise operation only with the elements of the second column. i
    %denotes the number of rows. i,2 means 2nd element of ith row
    error(i)= prediction(i)-y(i);
end
error_sum_square= sum(error.^2);
J= J+ (1/(2*m))*error_sum_square;    
end