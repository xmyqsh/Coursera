% File: EM_HMM.m
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [P loglikelihood ClassProb PairProb] = EM_HMM(actionData, poseData, G, InitialClassProb, InitialPairProb, maxIter)

% INPUTS
% actionData: structure holding the actions as described in the PA
% poseData: N x 10 x 3 matrix, where N is number of poses in all actions
% G: graph parameterization as explained in PA description
% InitialClassProb: N x K matrix, initial allocation of the N poses to the K
%   states. InitialClassProb(i,j) is the probability that example i belongs
%   to state j.
%   This is described in more detail in the PA.
% InitialPairProb: V x K^2 matrix, where V is the total number of pose
%   transitions in all HMM action models, and K is the number of states.
%   This is described in more detail in the PA.
% maxIter: max number of iterations to run EM

% OUTPUTS
% P: structure holding the learned parameters as described in the PA
% loglikelihood: #(iterations run) x 1 vector of loglikelihoods stored for
%   each iteration
% ClassProb: N x K matrix of the conditional class probability of the N examples to the
%   K states in the final iteration. ClassProb(i,j) is the probability that
%   example i belongs to state j. This is described in more detail in the PA.
% PairProb: V x K^2 matrix, where V is the total number of pose transitions
%   in all HMM action models, and K is the number of states. This is
%   described in more detail in the PA.

% Initialize variables
N = size(poseData, 1);
K = size(InitialClassProb, 2);
L = size(actionData, 2); % number of actions
V = size(InitialPairProb, 1);

ClassProb = InitialClassProb;
PairProb = InitialPairProb;

loglikelihood = zeros(maxIter,1);

P.c = [];
P.clg.sigma_x = [];
P.clg.sigma_y = [];
P.clg.sigma_angle = [];

% EM algorithm
for iter=1:maxIter

  % M-STEP to estimate parameters for Gaussians
  % Fill in P.c, the initial state prior probability (NOT the class probability as in PA8 and EM_cluster.m)
  % Fill in P.clg for each body part and each class
  % Make sure to choose the right parameterization based on G(i,1)
  % Hint: This part should be similar to your work from PA8 and EM_cluster.m

  P.c = zeros(1,K);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  for i = 1 : L
    P.c += ClassProb(actionData(i).marg_ind(1), :); % P.c is the initial state probability P(S1)
  end
  P.c /= L;

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

  % M-STEP to estimate parameters for transition matrix
  % Fill in P.transMatrix, the transition matrix for states
  % P.transMatrix(i,j) is the probability of transitioning from state i to state j
  P.transMatrix = zeros(K,K);

  % Add Dirichlet prior based on size of poseData to avoid 0 probabilities
  P.transMatrix = P.transMatrix + size(PairProb,1) * .05;

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  P.transMatrix += reshape(sum(PairProb), K, K);
  P.transMatrix = P.transMatrix ./ repmat(sum(P.transMatrix, 2), 1, K);
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


  % E-STEP preparation: compute the emission model factors (emission probabilities) in log space for each 
  % of the poses in all actions = log( P(Pose | State) )
  % Hint: This part should be similar to (but NOT the same as) your code in EM_cluster.m

  logEmissionProb = zeros(N,K);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  for i = 1 : N
    data = squeeze(poseData(i, :, :));
    for j = 1 : 10
      if G(j, 1) == 0
        for k = 1 : K
          logEmissionProb(i, k) += locallognormpdf(data(j, 1), P.clg(j).mu_y(k), P.clg(j).sigma_y(k));
          logEmissionProb(i, k) += locallognormpdf(data(j, 2), P.clg(j).mu_x(k), P.clg(j).sigma_x(k));
          logEmissionProb(i, k) += locallognormpdf(data(j, 3), P.clg(j).mu_angle(k), P.clg(j).sigma_angle(k));
        end
      else
        parent = data(G(j, 2), :);
        for k = 1 : K
          theta = P.clg(j).theta(k, :);
          mu_y = [1, parent] * theta(1 : 4)';
          mu_x = [1, parent] * theta(5 : 8)';
          mu_angle = [1, parent] * theta(9 : 12)';
          logEmissionProb(i, k) += locallognormpdf(data(j, 1), mu_y, P.clg(j).sigma_y(k));
          logEmissionProb(i, k) += locallognormpdf(data(j, 2), mu_x, P.clg(j).sigma_x(k));
          logEmissionProb(i, k) += locallognormpdf(data(j, 3), mu_angle, P.clg(j).sigma_angle(k));
        end
      end
    end
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


  % E-STEP to compute expected sufficient statistics
  % ClassProb contains the conditional class probabilities for each pose in all actions
  % PairProb contains the expected sufficient statistics for the transition CPDs (pairwise transition probabilities)
  % Also compute log likelihood of dataset for this iteration
  % You should do inference and compute everything in log space, only converting to probability space at the end
  % Hint: You should use the logsumexp() function here to do probability normalization in log space to avoid numerical issues

  ClassProb = zeros(N,K);
  PairProb = zeros(V,K^2);
  loglikelihood(iter) = 0;

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  for i = 1 : L
    % construct factors with three types and do inference to fill ClassProb and PairProb
    m = length(actionData(i).marg_ind); % 1 to m represents S variables
    F = repmat(struct('var', [], 'card', [], 'val', []), 1, 2 * m);
    % 1 is inital probability P(S1)
    F(1).var = [1]; F(1).card = [K]; F(1).val = log(P.c);
    % 2~m is transition probability P(Si|Si-1)
    transMat = log(reshape(P.transMatrix', 1, K * K));
    for j = 2 : m
      F(j).var = [j, j - 1]; F(j).card = [K, K]; F(j).val = transMat;
    end
    % m+1~2*m is emission probability P(O|S) = P(Pose|State)
    for j = 1 : m
      F(m + j).var = [j]; F(m + j).card = [K];
      F(m + j).val = logEmissionProb(actionData(i).marg_ind(j), :);
    end
    % inference
    [M, PCalibrated] = ComputeExactMarginalsHMM(F);
    for j = 1 : m
      ClassProb(actionData(i).marg_ind(j), :) = M(j).val;
    end
    for j = 1 : m - 1  % length(actionData(i).pair_ind) = m - 1
      trans = PCalibrated.cliqueList(j).val;
      PairProb(actionData(i).pair_ind(j), :) = trans - logsumexp(trans);
    end
    loglikelihood(iter) += logsumexp(PCalibrated.cliqueList(1).val); % summation on any clique is OK
  end
  ClassProb = exp(ClassProb); PairProb = exp(PairProb);
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % Print out loglikelihood
  disp(sprintf('EM iteration %d: log likelihood: %f', ...
    iter, loglikelihood(iter)));
  if exist('OCTAVE_VERSION')
    fflush(stdout);
  end

  % Check for overfitting by decreasing loglikelihood
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
