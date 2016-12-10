function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
%cost = 0;

%thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.


weightDecay = (1/2) * lambda * sum(sum(theta.*theta));

% M1(r, c) is theta(r)' * x(c)
M1 = theta * data;
% preventing overflows
M1 = bsxfun(@minus, M1, max(M1, [], 1));
% M2(r, c) is exp(theta(r)' * x(c))
M2 = exp(M1);
% M2 is the predicted matrix
M2 = bsxfun(@rdivide, M2, sum(M2));
% 1{·} operator only preserve a part of positions of log(M2)
M = groundTruth .* log(M2);

cost = -(1/numCases) * sum(sum(M)) + weightDecay;

% thetagrad
thetagrad = zeros(numClasses, inputSize);
% difference between ground truth and predict value
diff = groundTruth - M2;

for i=1:numClasses
    thetagrad(i,:) = -(1/numCases) * sum((data .* repmat(diff(i,:), inputSize, 1)) ,2)' + lambda * theta(i,:);
end

% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = thetagrad(:);
end