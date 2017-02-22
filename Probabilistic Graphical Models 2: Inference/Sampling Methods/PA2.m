rand('seed', 1);

% Construct the toy network
%[toy_network, toy_factors] = ConstructToyNetwork(0.3, 1);
%[toy_network, toy_factors] = ConstructToyNetwork(1, 0.2);
[toy_network, toy_factors] = ConstructToyNetwork(0.5, 0.5);
toy_evidence = zeros(1, length(toy_network.names));

% MCMC Inference
transition_names = {'Gibbs', 'MHUniform', 'MHGibbs', 'MHSwendsenWang1', 'MHSwendsenWang2'};
%transition_names = {'MHSwendsenWang1', 'MHSwendsenWang2'};
%transition_names = {'MHSwendsenWang2'};

for j = 1:length(transition_names)
    samples_list = {};

    num_chains_to_run = 3;
    for i = 1:num_chains_to_run
        % Random Initialization
        A0 = ceil(rand(1, length(toy_network.names)) .* toy_network.card);

        % Initialization to all ones
        % A0 = i * ones(1, length(toy_network.names));

        [M, all_samples] = ...
            MCMCInference(toy_network, toy_factors, toy_evidence, transition_names{j}, 0, 4000, 1, A0);
        samples_list{i} = all_samples;
        figure, VisualizeToyImageMarginals(toy_network, M, i, transition_names{j});
    end

    vis_vars = [3];
    VisualizeMCMCMarginals(samples_list, vis_vars, toy_network.card(vis_vars), toy_factors, ...
      500, transition_names{j});
    disp(['Displaying results for MCMC with transition ', transition_names{j}]);
    disp(['Hit enter to continue']);
    pause;
end
