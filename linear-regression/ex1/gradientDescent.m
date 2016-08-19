function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    sum_of_cost=0;
    hypothesis=X*theta;

    sum_of_cost=hypothesis-y;

    % theta(1)-=alpha*(sum_of_cost'*X(:,1))/m;
    % theta(2)-=alpha*(sum_of_cost'*X(:,2))/m;

    theta-=(alpha*(sum_of_cost'*X)/m)';
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter)=computeCost(X, y, theta);

end

end
