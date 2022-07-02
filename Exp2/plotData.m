function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure;
X(:,1)=[];

%we are solving a classification problem
%this means it is important for us to separate the positive and the
%negative examples in the data set
%to aid us in this separation, we make use of the matlab function find
%which does the job of evaluating indices of the entries of the vector that
%are equal to the value we specify

positive_eg= find(y==1); %finds all the indices of the vector y which are one
negative_eg= find(y==0); %finds all the indices of the vector y which are zero
plot(X(positive_eg,1),X(positive_eg,2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
hold on;
plot(X(negative_eg,1), X(negative_eg,2), 'ko', 'MarkerFaceColor', 'r','MarkerSize', 7);
hold off;
xlabel('Exam 1 Score');
ylabel('Exam 2 Score');

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
% ========================================================================

end
