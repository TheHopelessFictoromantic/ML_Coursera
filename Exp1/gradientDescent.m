function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
% theta = GRADIENTDESENT(X, Y, theta, alpha, num_iters) updates theta by 
% taking num_iters gradient steps with learning rate alpha

m = length(y);
J_history = zeros(num_iters, 1);

        function [J1_cost, J2_cost]= costFunction(X_cost,y_cost,theta_cost)
            J1_cost= 0;
            J2_cost= 0;
            esum1= 0;
            esum2=0;
            for j= 1: length(y_cost)
                prediction(j)= theta_cost(1) + (theta_cost(2).*X_cost(j,2));
                error1(j)= prediction(j)- y_cost(j);
                error2(j)= (prediction(j)-y_cost(j)).*X_cost(j,2);
            end
            esum1= sum(error1);
            esum2= sum(error2);
            J1_cost= (1/length(y_cost))*esum1;
            J2_cost= (1/length(y_cost))*esum2;
        end

for iter = 1:num_iters
    [Jtheta1, Jtheta2]= costFunction(X,y,theta);
    theta(1)= theta(1)- alpha*Jtheta1;
    theta(2)= theta(2)- alpha*Jtheta2;
    
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
end

end