function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)

%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values

m = length(y);
J_history= zeros(1,num_iters);

    function J_cost= costFunction(X_cost, y_cost, theta_cost)
        prediction= X_cost*theta_cost;
        error=(prediction-y_cost)'*X_cost;   %1xm X mxn => 1xn
        J_cost= error';
    end

        %but theta is nx1! so transpose the error
        %no need to sum it as we are changing J for every iteration 
        %and summing would give the same val for theta which we don't want

for iter = 1:num_iters
    J_hist= costFunction(X,y,theta);
    theta= theta- alpha*(1/m)*(J_hist);

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
        % ============================================================

    % Save the cost J in every iteration
    J_history(iter)= computeCostMulti(X,y,theta);

end
end