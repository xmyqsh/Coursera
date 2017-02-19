% File: EM_cluster.m
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [P loglikelihood ClassProb] = EM_cluster(poseData, G, InitialClassProb, maxIter)

% INPUTS
% poseData: N x 10 x 3 matrix, where N is number of poses;
%   poseData(i,:,:) yields the 10x3 matrix for pose i.
% G: graph parameterization as explained in PA8
% InitialClassProb: N x K, initial allocation of the N poses to the K
%   classes. InitialClassProb(i,j) is the probability that example i belongs
%   to class j
% maxIter: max number of iterations to run EM

% OUTPUTS
% P: structure holding the learned parameters as described in the PA
% loglikelihood: #(iterations run) x 1 vector of loglikelihoods stored for
%   each iteration
% ClassProb: N x K, conditional class probability of the N examples to the
%   K classes in the final iteration. ClassProb(i,j) is the probability that
%   example i belongs to class j

% Initialize variables
N = size(poseData, 1);
K = size(InitialClassProb, 2);

ClassProb = InitialClassProb;

loglikelihood = zeros(maxIter,1);

P.c = [];
P.clg.sigma_x = [];
P.clg.sigma_y = [];
P.clg.sigma_angle = [];

% EM algorithm
for iter=1:maxIter

  % M-STEP to estimate parameters for Gaussians
  %
  % Fill in P.c with the estimates for prior class probabilities
  % Fill in P.clg for each body part and each class
  % Make sure to choose the right parameterization based on G(i,1)
  %
  % Hint: This part should be similar to your work from PA8

  P.c = zeros(1,K);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  P.c = sum(ClassProb) / N;
  for j = 1 : 10
    if G(j, 1) == 0
      for k = 1 : K
        [P.clg(j).mu_y(k), P.clg(j).sigma_y(k)] = FitG(poseData(:, j, 1), ClassProb(:, k));
        [P.clg(j).mu_x(k), P.clg(j).sigma_x(k)] = FitG(poseData(:, j, 2), ClassProb(:, k));
        [P.clg(j).mu_angle(k), P.clg(j).sigma_angle(k)] = FitG(poseData(:, j, 3), ClassProb(:, k));
      end
    else
      for k = 1 : K
        [betaY, P.clg(j).sigma_y(k)] = FitLG(poseData(:, j, 1), squeeze(poseData(:, G(j, 2), :)), ClassProb(:, k));
        [betaX, P.clg(j).sigma_x(k)] = FitLG(poseData(:, j, 2), squeeze(poseData(:, G(j, 2), :)), ClassProb(:, k));
        [betaAngle, P.clg(j).sigma_angle(k)] = FitLG(poseData(:, j, 3), squeeze(poseData(:, G(j, 2), :)), ClassProb(:, k));
        P.clg(j).theta(k, :) = [betaY(4), betaY(1:3)', betaX(4), betaX(1:3)', betaAngle(4), betaAngle(1:3)'];
      end
    end
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % E-STEP to re-estimate ClassProb using the new parameters
  %
  % Update ClassProb with the new conditional class probabilities.
  % Recall that ClassProb(i,j) is the probability that example i belongs to
  % class j.
  %
  % You should compute everything in log space, and only convert to
  % probability space at the end.
  %
  % Tip: To make things faster, try to reduce the number of calls to
  % lognormpdf, and inline the function (i.e., copy the lognormpdf code
  % into this file)
  %
  % Hint: You should use the logsumexp() function here to do
  % probability normalization in log space to avoid numerical issues

  ClassProb = zeros(N,K);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  for i = 1 : N
    data = squeeze(poseData(i, :, :));
    ClassProb(i, :) = log(P.c);
    for j = 1 : 10
      if G(j, 1) == 0
        for k = 1 : K
          ClassProb(i, k) += locallognormpdf(data(j, 1), P.clg(j).mu_y(k), P.clg(j).sigma_y(k));
          ClassProb(i, k) += locallognormpdf(data(j, 2), P.clg(j).mu_x(k), P.clg(j).sigma_x(k));
          ClassProb(i, k) += locallognormpdf(data(j, 3), P.clg(j).mu_angle(k), P.clg(j).sigma_angle(k));
        end
      else
        parent = data(G(j, 2), :);
        for k = 1 : K
          theta = P.clg(j).theta(k, :);
          mu_y = [1, parent] * theta(1 : 4)';
          mu_x = [1, parent] * theta(5 : 8)';
          mu_angle = [1, parent] * theta(9 : 12)';
          ClassProb(i, k) += locallognormpdf(data(j, 1), mu_y, P.clg(j).sigma_y(k));
          ClassProb(i, k) += locallognormpdf(data(j, 2), mu_x, P.clg(j).sigma_x(k));
          ClassProb(i, k) += locallognormpdf(data(j, 3), mu_angle, P.clg(j).sigma_angle(k));
        end
      end
    end
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % Compute log likelihood of dataset for this iteration
  % Hint: You should use the logsumexp() function here
  loglikelihood(iter) = 0;
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  logSumClassProb = logsumexp(ClassProb);
  loglikelihood(iter) = sum(logSumClassProb);
  ClassProb = exp(ClassProb - repmat(logSumClassProb, 1, K));
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % Print out loglikelihood
  disp(sprintf('EM iteration %d: log likelihood: %f', ...
    iter, loglikelihood(iter)));
  if exist('OCTAVE_VERSION')
    fflush(stdout);
  end

  % Check for overfitting: when loglikelihood decreases
  if iter > 1
    if loglikelihood(iter) < loglikelihood(iter-1)
      break;
    end
  end

end

% Remove iterations if we exited early
loglikelihood = loglikelihood(1:iter);
end


% call local lognormpdf for speeding up
function val = locallognormpdf(x, mu, sigma)
val = - (x - mu).^2 / (2*sigma^2) - log (sqrt(2*pi) * sigma);
end
