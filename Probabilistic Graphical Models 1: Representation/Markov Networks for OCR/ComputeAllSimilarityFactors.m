function factors = ComputeAllSimilarityFactors (images, K)
% This function computes all of the similarity factors for the images in
% one word.
%
% Input:
%   images: An array of structs containing the 'img' value for each
%     character in the word.
%   K: The alphabet size (accessible in imageModel.K for the provided
%     imageModel).
%
% Output:
%   factors: Every similarity factor in the word. You should use
%     ComputeSimilarityFactor to compute these.
%
% Copyright (C) Daphne Koller, Stanford University, 2012

n = length(images);
nFactors = nchoosek (n, 2);

factors = repmat(struct('var', [], 'card', [], 'val', []), nFactors, 1);

% Your code here:

%assignments = nchoosek ([1 : n], 2);
%for idx = 1 : nFactors
%  Index = AssignmentToIndex(assignments(idx, :), [K, K]);
%  factors(Index).var = assignments(idx);
%  factors(Index).card = [K, K];
%  factors(Index).val = ComputeSimilarityFactor(images, K, assignments(idx, 1), assignments(idx, 2));
%end

idx = 1;
for i = 1 : n - 1
  for j = i + 1 : n
    factors(idx) = ComputeSimilarityFactor(images, K, i, j);
    idx += 1;
  end
end

end

