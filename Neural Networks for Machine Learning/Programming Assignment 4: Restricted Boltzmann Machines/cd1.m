function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.

    % visible_data maybe probability
    visible_data = sample_bernoulli(visible_data);

    hidden_probability = visible_state_to_hidden_probabilities(rbm_w, visible_data);
    hidden_sample = sample_bernoulli(hidden_probability);
    data_greater_goodness = configuration_goodness_gradient(visible_data, hidden_sample);

    visible_reconstruction_probability = hidden_state_to_visible_probabilities(rbm_w, hidden_sample);
    visible_reconstruction_sample = sample_bernoulli(visible_reconstruction_probability);
    hidden_reconstruction_probability = visible_state_to_hidden_probabilities(rbm_w, visible_reconstruction_sample);
    % For question 7
    % hidden_reconstruction_sample = sample_bernoulli(hidden_reconstruction_probability);
    % reconstruction_less_goodness = configuration_goodness_gradient(visible_reconstruction_sample, hidden_reconstruction_sample);
    % For question 8
    reconstruction_less_goodness = configuration_goodness_gradient(visible_reconstruction_sample, hidden_reconstruction_probability);

    ret = data_greater_goodness .- reconstruction_less_goodness;
end
