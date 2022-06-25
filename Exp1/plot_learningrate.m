function plot_learningrate(num_iters, J_history)
% Plot the convergence graph
plot(num_iters, J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');
end