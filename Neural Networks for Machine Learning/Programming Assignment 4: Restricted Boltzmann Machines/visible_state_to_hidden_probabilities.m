function hidden_probability = visible_state_to_hidden_probabilities(rbm_w, visible_state)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_state> is a binary matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
% The returned value is a matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
% This takes in the (binary) states of the visible units, and returns the activation probabilities of the hidden units conditional on those states.
    % http://blog.csdn.net/itplus/article/details/19168989
    % (3.30) P(h_k = 1 | V) = sigmoid(b_k + sum_i=1,nv(w_k,i*v_i))
    % matrix format  P(H = vec(1) | V) = sigmoid(B + W*V)
    % hidden_probability = 1 ./ (1 + exp(-(rbm_w * visible_state)));
    hidden_probability = logistic(rbm_w * visible_state);
end
