%% get data correlations

clear
load ../data/shuffled_data/data2014

[total_N, obvs] = size(train_reps);

for NN = 10:25
    n_reps = 20;
    pks = zeros(NN+1, n_reps);  % to save pk

    for rep = 1:n_reps
        disp(rep)
        rng(rep)
        id_N = randperm(total_N, NN); 
        X = train_reps(id_N,:); % select the training data
        % get data correlations
        pks(:,rep) = samplepk(X);
    end

    save_prefix = "../results/correlations/stimulus_" + NN;
    
    % save in .csv format
    writematrix(pks, save_prefix +'_pks.csv'); % save model weights 
end

%% get model correlations

clear
load ../data/shuffled_data/data2014

[total_N, obvs] = size(train_reps);

 name = 'indep'; % 'indep' 'pairwise' 'third' 

% parameters
for NN = 10:25
    prefix = "../data/trained_models/stimulus_" + NN;
    n_reps = 20;
    n_samples = NN*10000; % choose this better

    % to save model pk to
    m_pks = zeros(NN+1, n_reps);

    for rep = 1:n_reps
        disp(rep)
        rng(rep)
        % get model correlations

        %indep
        m = load(prefix + "_" + name + "_" + rep);
        disp("Model sampling")
        samples = maxent.generateSamples(m.model, n_samples); % returns a N x n_samples matrix
        disp("Working out pk")
        m_pks(:,rep) = samplepk(samples);
    end

    save_prefix = "../results/correlations/stimulus_" + NN;
    
    tag = name;
    if strcmp(name,'indep')
        tag = 'ind';
    end
    if strcmp(name,'pairwise')
        tag = 'ising';
    end
    
    % save in .csv format
    writematrix(m_pks, save_prefix +'_' + tag + '_pks.csv'); % save model weights 
end