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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

s_X=sigmoid(X*theta);

sum_t=0;
for i=2:length(theta),
	sum_t=sum_t+theta(i)^2;
end

regu= (lambda * sum_t) /(2*m);

J=sum(-y.*log(s_X)-(1-y).*log(1-s_X));
J=J/m + regu;

for i=1:length(theta),
	grad(i)=sum((s_X - y).*X(:,i));
	grad(i)=grad(i)/m;
	grad(i)=grad(i) + (lambda * theta(i))/m;
end

grad(1)=sum((s_X - y).*X(:,1));
grad(1)=grad(1)/m;



% =============================================================

end
