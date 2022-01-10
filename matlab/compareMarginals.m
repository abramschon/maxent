%% get data correlations

clear
load ../data/shuffled_data/data2014

[total_N, obvs] = size(train_reps);

% parameters
for NN = 10:25

    shuffle = 'stimulus'; % 'stimulus' or 'time'
    prefix = "../data/trained_models/" + shuffle + "_" + NN;
    n_reps = 10;
    n_samples = NN*10000; % should work this out based on

    % to save pk
    pks = zeros(NN+1, n_reps);
    ind_pks = zeros(NN+1, n_reps);
    ising_pks = zeros(NN+1, n_reps);


    for rep = 1:n_reps
        disp(rep)
        rng(rep)
        id_N = randperm(total_N, NN); 
        X = train_reps(id_N,:); % select the training data
        % get data correlations
        pks(:,rep) = samplepk(X);

        % get model correlations

        %indep
        m_ind = load(prefix + "_indep_" + rep);
        disp("Indep sampling")
        samples = maxent.generateSamples(m_ind.model, n_samples); % returns a N x n_samples matrix
        disp("Indep pk")
        ind_pks(:,rep) = samplepk(samples);

        %pairwise
        m_ising = load(prefix + "_pairwise_" + rep);
        disp("Pairwise sampling")
        samples = maxent.generateSamples(m_ising.model, n_samples); % returns a N x n_samples matrix
        disp("Pairwise pk")
        ising_pks(:,rep) = samplepk(samples);
    end

    save_prefix = "../results/correlations/" + shuffle + "_" + NN;

    % save in matlab format
    save(save_prefix + "_indep_pwise_pks_" + n_reps, ...
        'pks', 'ind_pks', 'ising_pks')

    % save in .csv format
    writematrix(pks, save_prefix +'_pks.csv'); % save model weights 
    writematrix(ind_pks, save_prefix +'_ind_pks.csv'); % save model weights 
    writematrix(ising_pks, save_prefix +'_ising_pks.csv'); % save model weights 
end

%% get model correlations

clear
load ../data/shuffled_data/data2014

[total_N, obvs] = size(train_reps);

% parameters
for NN = 10:25
    % NN = 100;
    shuffle = 'stimulus'; % 'stimulus' or 'time'
    prefix = "../data/trained_models/" + shuffle + "_" + NN;
    n_reps = 10;
    n_samples = NN*10000;

    % to save pk
    pks = zeros(NN+1, n_reps);
    ind_pks = zeros(NN+1, n_reps);
    ising_pks = zeros(NN+1, n_reps);


    for rep = 1:n_reps
        disp(rep)
        rng(rep)
        id_N = randperm(total_N, NN); 
        X = train_reps(id_N,:); % select the training data
        % get data correlations
        pks(:,rep) = samplepk(X);

        % get model correlations

        %indep
        m_ind = load(prefix + "_indep_" + rep);
        disp("Indep sampling")
        samples = maxent.generateSamples(m_ind.model, n_samples); % returns a N x n_samples matrix
        disp("Indep pk")
        ind_pks(:,rep) = samplepk(samples);

        %pairwise
        m_ising = load(prefix + "_pairwise_" + rep);
        disp("Pairwise sampling")
        samples = maxent.generateSamples(m_ising.model, n_samples); % returns a N x n_samples matrix
        disp("Pairwise pk")
        ising_pks(:,rep) = samplepk(samples);
    end

    save_prefix = "../results/correlations/" + shuffle + "_" + NN;

    % save in matlab format
    save(save_prefix + "_indep_pwise_pks_" + n_reps, ...
        'pks', 'ind_pks', 'ising_pks')

    % save in .csv format
    writematrix(pks, save_prefix +'_pks.csv'); % save model weights 
    writematrix(ind_pks, save_prefix +'_ind_pks.csv'); % save model weights 
    writematrix(ising_pks, save_prefix +'_ising_pks.csv'); % save model weights 
end