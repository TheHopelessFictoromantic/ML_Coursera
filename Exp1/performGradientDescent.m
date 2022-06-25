
function optimumJ= performGradientDescent(X,y,theta,alpha, num_iters)
theta = zeros(3, 1);
[~, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
optimumJ= J_history; 
end 