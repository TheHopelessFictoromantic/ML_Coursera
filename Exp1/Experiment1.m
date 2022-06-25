clear;
dir;

%simple headstart into the programming assignment
warmUpExercise();

%performing linear regression
data = load('ex1data1.txt'); % read comma separated data
X = data(:, 1); y = data(:, 2);

%plot the data
plotData(X,y);
title('Population-Profit Data plot');
xlabel('Population of City in 10,000s');
ylabel('Profit in $10,000s');

%initializing the parameters
m = length(X); % number of training examples
X = [ones(m,1),data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters
iterations = 1500;
alpha = 0.01;

%compute the cost for all the parameters with initial theta= [0;0]
computeCost(X, y, theta);

%compute the cost for a non-zero theta
computeCost(X, y,[-1; 2])

% Run gradient descent:
% Compute theta
theta = gradientDescent(X, y, theta, alpha, iterations);

% Print theta to screen
% Display gradient descent's result
fprintf('Theta computed from gradient descent:\n%f,\n%f',theta(1),theta(2))

% Plot the linear fit
% keep previous plot visible
hold on;
x= X(:,2);
plotLinearFit(X, x, theta, y);
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure

% Predict values for population sizes of 35,000 and 70,000
predict1 = [1, 3.5] *theta;
fprintf('\nFor population = 35,000, we predict a profit of %f\n', predict1*10000);
predict2 = [1, 7] * theta;
fprintf('For population = 70,000, we predict a profit of %f\n', predict2*10000);

% Visualizing J(theta_0, theta_1):
% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];    
	  J_vals(i,j) = computeCost(X, y, t);
    end
end

% Because of the way meshgrids work in the surf command, we need to 
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';

% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plotTheta(theta);
hold off;

% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Print out some data points
% First 10 examples from the dataset
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

[X, mu, sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];

% Run gradient descent
% Choose some alpha value
alpha = 0.1;
num_iters = 400;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, ~] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Display gradient descent's result
fprintf('Theta computed from gradient descent:\n%f\n%f\n%f',theta(1),theta(2),theta(3))

theta= zeros(2,1);
[theta_var, ~]= gradientDescent(X,y, theta, alpha, num_iters);
% Estimate the price of a 1650 sq-ft, 3 br house
price = [1650 3]*theta_var; % Enter your price formula here

fprintf('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $%f', price);

disp('The available alpha and iteration values for performing optimum gradien descent are as follows');
disp('1. alpha = 0.01');
disp('2. alpha = 0.03');
disp('3. alpha = 0.1');
disp('4. alpha = 0.3');
disp('5. alpha = 1');
figure;
choice=0;
while choice<26
choice= input('Enter a choice to choose an alpha and an iteration value: ');
switch choice
case 1
J1= performGradientDescent(X,y, theta, 0.01, 50);
subplot(551);
title('Cost Function with a learning rate of 0.01 and iterations of 50');
plot_learningrate(1:50, J1);

case 2
J2= performGradientDescent(X,y, theta, 0.01, 75);
subplot(552);
title('Cost Function with a learning rate of 0.01 and iterations of 75');
plot_learningrate(1:75, J2);

case 3
J3= performGradientDescent(X,y, theta, 0.01, 100);
subplot(553);
title('Cost Function with a learning rate of 0.01 and iterations of 100');
plot_learningrate(1:100, J3);

case 4
J4= performGradientDescent(X,y, theta, 0.01, 125);
subplot(554);
title('Cost Function with a learning rate of 0.01 and iterations of 125');
plot_learningrate(1:125, J4);

case 5
J5= performGradientDescent(X,y, theta, 0.01, 150);
subplot(555);
title('Cost Function with a learning rate of 0.01 and iterations of 150');
plot_learningrate(1:150, J5);

case 6
J6= performGradientDescent(X,y, theta, 0.03, 50);
subplot(556);
title('Cost Function with a learning rate of 0.03 and iterations of 50');
plot_learningrate(1:50, J6);

case 7
J7= performGradientDescent(X,y, theta, 0.03, 75);
subplot(557);
title('Cost Function with a learning rate of 0.03 and iterations of 75');
plot_learningrate(1:75, J7);

case 8
J8= performGradientDescent(X,y, theta, 0.03, 100);
subplot(558);
title('Cost Function with a learning rate of 0.03 and iterations of 100');
plot_learningrate(1:100, J8);

case 9
J9= performGradientDescent(X,y, theta, 0.03, 125);
subplot(559);
title('Cost Function with a learning rate of 0.03 and iterations of 125');
plot_learningrate(1:125, J9);

case 10
J10= performGradientDescent(X,y, theta, 0.03, 150);
subplot(5,5,10);
title('Cost Function with a learning rate of 0.03 and iterations of 150');
plot_learningrate(1:150, J10);

case 11
J11= performGradientDescent(X,y, theta, 0.1, 50);
subplot(5,5,11);
title('Cost Function with a learning rate of 0.1 and iterations of 50');
plot_learningrate(1:50, J11);

case 12
J12= performGradientDescent(X,y, theta, 0.1, 75);
subplot(5,5,12);
title('Cost Function with a learning rate of 0.1 and iterations of 75');
plot_learningrate(1:75, J12);

case 13
J13= performGradientDescent(X,y, theta, 0.1, 100);
subplot(5,5,13);
title('Cost Function with a learning rate of 0.1 and iterations of 100');
plot_learningrate(1:100, J13);

case 14
J14= performGradientDescent(X,y, theta, 0.1, 125);
subplot(5,5,14);
title('Cost Function with a learning rate of 0.1 and iterations of 125');
plot_learningrate(1:125, J14);

case 15
J15= performGradientDescent(X,y, theta, 0.1, 150);
subplot(5,5,15);
title('Cost Function with a learning rate of 0.1 and iterations of 150');
plot_learningrate(1:150, J15);

case 16
J16= performGradientDescent(X,y, theta, 0.3, 50);
subplot(5,5,16);
title('Cost Function with a learning rate of 0.3 and iterations of 50');
plot_learningrate(1:50, J16);

case 17
J17= performGradientDescent(X,y, theta, 0.3, 75);
subplot(5,5,17);
title('Cost Function with a learning rate of 0.3 and iterations of 75');
plot_learningrate(1:75, J17);

case 18
J18= performGradientDescent(X,y, theta, 0.3, 100);
subplot(5,5,18);
title('Cost Function with a learning rate of 0.3 and iterations of 100');
plot_learningrate(1:100, J18);

case 19
J19= performGradientDescent(X,y, theta, 0.3, 125);
subplot(5,5,19);
title('Cost Function with a learning rate of 0.3 and iterations of 125');
plot_learningrate(1:125, J19);

case 20
J20= performGradientDescent(X,y, theta, 0.3, 150);
subplot(5,5,20);
title('Cost Function with a learning rate of 0.3 and iterations of 150');
plot_learningrate(1:150, J20);

case 21
J21= performGradientDescent(X,y, theta, 1, 50);
subplot(5,5,21);
title('Cost Function with a learning rate of 1 and iterations of 50');
plot_learningrate(1:50, J21);

case 22
J22= performGradientDescent(X,y, theta, 1, 75);
subplot(5,5,22);
title('Cost Function with a learning rate of 1 and iterations of 75');
plot_learningrate(1:75, J22);

case 23
J23= performGradientDescent(X,y, theta, 1, 100);
subplot(5,5,23);
title('Cost Function with a learning rate of 1 and iterations of 100');
plot_learningrate(1:100, J23);

case 24
J24= performGradientDescent(X,y, theta, 1, 125);
subplot(5,5,24);
title('Cost Function with a learning rate of 1 and iterations of 125');
plot_learningrate(1:125, J24);

case 25
J25= performGradientDescent(X,y, theta, 1, 150);
subplot(5,5,25);
title('Cost Function with a learning rate of 1 and iterations of 150');
plot_learningrate(1:150, J25);

otherwise
disp('End')
end
end

% Run gradient descent
% Replace the value of alpha below best alpha value you found above
alpha_best = input('Enter the best value of alpha: ');
num_iters = input('Enter the best value of the number of iterations: ');

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, ~] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Display gradient descent's result
fprintf('Theta computed from gradient descent:\n%f\n%f\n%f',theta(1),theta(2),theta(3))

% Estimate the price of a 1650 sq-ft, 3 br house. You can use the same
% code you entered earlier to predict the price
[theta_var_new]= gradientDescent(X,y,theta,alpha, num_iters);

X_house= [2104 5; 1416 3; 1534 3; 852 2; 1650 3];

[X_house_norm, mu_house, sigma_house]= featureNormalize(X_house);

feet_norm= X_house_norm(4,1)
br_norm= X_house_norm(4,2)

price = [1, feet_norm, br_norm] * theta_var_new; % Enter your price formula here

fprintf('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $%f', price);

% Solve with normal equations:
% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations:\n%f\n%f\n%f', theta(1),theta(2),theta(3));

% Run gradient descent:
% Choose some alpha value






