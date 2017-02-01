function G = configuration_goodness(rbm_w, visible_state, hidden_state)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_state> is a binary matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
% <hidden_state> is a binary matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
% This returns a scalar: the mean over cases of the goodness (negative energy) of the described configurations.
    % http://blog.csdn.net/itplus/article/details/19168989
    % (3.21) E = -A'V - B'H - H'WV
    % multi cases  G = -scalar(E) / num_cases = (sum(A'V) + sum(B'H) + trace(H'WV)) / num_cases
    G = trace(hidden_state' * rbm_w * visible_state) / size(visible_state, 2);
end
