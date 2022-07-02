function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
reg_parameter= 0;

[J1,gradient]= costFunction(theta,X,y);
%compute regularized cost
theta1= theta;
theta1(1)=[];
theta_squared= sum(theta1.^2);
reg_parameter= ((lambda)/(2*m))*theta_squared;
J= J1 + reg_parameter;

%compute regularized gradient
gradient1= gradient(1);
grad_ToRegularize= gradient;
grad_ToRegularize(1)=[];
theta_ToRegularize= theta;
theta_ToRegularize(1)= [];
reg_grad_parameter= (lambda/m)*theta_ToRegularize;
grad1= grad_ToRegularize + reg_grad_parameter;
reg_grad1= grad1';
reg_grad= [gradient1 reg_grad1];
grad= reg_grad';
    

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
% =============================================================
end
