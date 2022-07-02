function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

a= X*theta;
prediction= sigmoid(a);

%computing the cost
cost= (-1)*(1/m)*(((y')*log(prediction)) + (((1-y)')*log(1-prediction)));
J= cost(:,1);

%computing the gradient
    function gradient= computeGradient(theta_compute, X_compute, y_compute)
        prediction_compute= sigmoid(X_compute*theta_compute);
        error= prediction_compute-y_compute;
        gradient_compute= (error')*X_compute;
        gradient= (1/length(y_compute))*(gradient_compute');
    end
grad= computeGradient(theta, X, y);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
% =============================================================

end