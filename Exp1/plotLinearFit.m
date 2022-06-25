function plotLinearFit(X,X_new,theta,y)
fit= polyfit(X*theta, y,1);
yfit= abs(fit(2)*X(:,2));
plot(X_new, yfit, 'b-');
end