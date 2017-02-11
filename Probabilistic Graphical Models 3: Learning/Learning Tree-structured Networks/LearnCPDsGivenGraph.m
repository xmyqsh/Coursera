function [P loglikelihood] = LearnCPDsGivenGraph(dataset, G, labels)
%
% Inputs:
% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha)
% G: graph parameterization as explained in PA description
% labels: N x 2 true class labels for the examples. labels(i,j)=1 if the
%         the ith example belongs to class j and 0 elsewhere
%
% Outputs:
% P: struct array parameters (explained in PA description)
% loglikelihood: log-likelihood of the data (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset, 1);
K = size(labels,2);

loglikelihood = 0;
P.c = zeros(1,K);

% estimate parameters
% fill in P.c, MLE for class probabilities
% fill in P.clg for each body part and each class
% choose the right parameterization based on G(i,1)
% compute the likelihood - you may want to use ComputeLogLikelihood.m
% you just implemented.
%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
if size(size(G), 2) == 2
  G1 = G;
  G2 = G;
else
  G1 = reshape(G(:, :, 1), 10, 2);
  G2 = reshape(G(:, :, 2), 10, 2);
end

human = dataset([labels(:, 1) == 1], :, :);
alien = dataset([labels(:, 2) == 1], :, :);
P.c = [size(human, 1), size(alien, 1)] / N;

for i = 1 : 10
  if G1(i, 1) == 0
    [P.clg(i).mu_y(1), P.clg(i).sigma_y(1)] = FitGaussianParameters(human(:, i, 1));
    [P.clg(i).mu_x(1), P.clg(i).sigma_x(1)] = FitGaussianParameters(human(:, i, 2));
    [P.clg(i).mu_angle(1), P.clg(i).sigma_angle(1)] = FitGaussianParameters(human(:, i, 3));
  else
    [betaY, P.clg(i).sigma_y(1)] = FitLinearGaussianParameters(human(:, i, 1), reshape(human(:, G1(i,2), :), size(human, 1), 3));
    [betaX, P.clg(i).sigma_x(1)] = FitLinearGaussianParameters(human(:, i, 2), reshape(human(:, G1(i,2), :), size(human, 1), 3));
    [betaAngle, P.clg(i).sigma_angle(1)] = FitLinearGaussianParameters(human(:, i, 3), reshape(human(:, G1(i,2), :), size(human, 1), 3));
    P.clg(i).theta(1, :) = [betaY(4), betaY(1:3)', betaX(4), betaX(1:3)', betaAngle(4), betaAngle(1:3)'];
  end

  if G2(i, 1) == 0
    [P.clg(i).mu_y(2), P.clg(i).sigma_y(2)] = FitGaussianParameters(alien(:, i, 1));
    [P.clg(i).mu_x(2), P.clg(i).sigma_x(2)] = FitGaussianParameters(alien(:, i, 2));
    [P.clg(i).mu_angle(2), P.clg(i).sigma_angle(2)] = FitGaussianParameters(alien(:, i, 3));
  else
    [betaY, P.clg(i).sigma_y(2)] = FitLinearGaussianParameters(alien(:, i, 1), reshape(alien(:, G2(i,2), :), size(alien, 1), 3));
    [betaX, P.clg(i).sigma_x(2)] = FitLinearGaussianParameters(alien(:, i, 2), reshape(alien(:, G2(i,2), :), size(alien, 1), 3));
    [betaAngle, P.clg(i).sigma_angle(2)] = FitLinearGaussianParameters(alien(:, i, 3), reshape(alien(:, G2(i,2), :), size(alien, 1), 3));
    P.clg(i).theta(2, :) = [betaY(4), betaY(1:3)', betaX(4), betaX(1:3)', betaAngle(4), betaAngle(1:3)'];
  end
end

loglikelihood = ComputeLogLikelihood(P, G, dataset);

% These are dummy lines added so that submit.m will run even if you
% have not started coding. Please delete them.
% P.clg.sigma_x = 0;
% P.clg.sigma_y = 0;
% P.clg.sigma_angle = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('log likelihood: %f\n', loglikelihood);

