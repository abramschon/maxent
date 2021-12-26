%% Getting 3rd order correlations from indep model
NN = 20;
indep = maxent.createModel(NN,'indep');
indep = maxent.trainModel(indep, randi([0,1], [NN,10000]) );
disp("Analytic 3 corrs")
corrs = corr3indep(indep);
disp("Sampling...")
samples = maxent.generateSamples(indep, NN*10000); % returns a N x n_samples matrix
disp("3 corrs")
corrs2 = samplecorr3(samples);
disp("pk")
pk = samplepk(samples);



%%
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

%% 3-wise correlations
clear
load ../data/shuffled_data/data2014

[total_N, obvs] = size(train_reps);

% parameters
NN = 100;
shuffle = 'stimulus'; % 'stimulus' or 'time'
prefix = "../data/trained_models/" + shuffle + "_" + NN;
n_reps = 10;
n_samples = NN*10;

% to save the 3-wise corrs 
corr3s = zeros(nchoosek(NN,3), n_reps);
ind_corr3s = zeros(nchoosek(NN,3), n_reps);
ising_corr3s = zeros(nchoosek(NN,3), n_reps);

for rep = 1:n_reps
    disp(rep)
    rng(rep)
    id_N = randperm(total_N, NN); 
    X = train_reps(id_N,:); % select the training data
    % get data correlations
    corr3s(:,rep) = samplecorr3(X);
    
    % get model correlations
    
    %indep
    m_ind = load(prefix + "_indep_" + rep);
    disp("Indep corrs")
    ind_corr3s(:,rep) = corr3indep(m_ind.model); % can do this without sampling
    
    %pairwise
    m_ising = load(prefix + "_pairwise_" + rep);
    disp("Pairwise sampling")
    samples = maxent.generateSamples(m_ising.model, n_samples); % returns a N x n_samples matrix
    disp("Pairwise corrs")
    ising_corr3s(:,rep) = samplecorr3(samples);
end

% save in matlab format
save(save_prefix + "_indep_pwise_corr3s_" + n_reps, ...
    'corr3s', 'ind_corr3s', 'ising_corr3s')

% save in .csv format
writematrix(corr3s, save_prefix +'_corr3s.csv'); % save model weights 
writematrix(ind_corr3s, save_prefix +'_ind_corr3s.csv'); % save model weights 
writematrix(ising_corr3s, save_prefix +'_ising_corr3s.csv'); % save model weights 

%% Getting model correlations

function corrs = corr3indep(m_indep)
avgs = maxent.getMarginals(m_indep);
combs = nchoosek(1:m_indep.ncells, 3);
corrs = prod(avgs(combs), 2).';
end

function corrs = samplecorr3(samples)
[N, n_samples] = size(samples);
combs = nchoosek(1:N, 3);
corrs = zeros(1,nchoosek(N, 3));
for i = 1:length(corrs) 
   corrs(i) =  sum( prod( samples(combs(i,:),:), 1) ); 
end
corrs = corrs / n_samples;
end

function pk = samplepk(samples)
[N, n_samples] = size(samples);
counts = sum(samples, 1); % how many neurons fire in each state
ks = unique(counts);
k_freq = hist(counts, ks); % how often each count pops up
pk = zeros(1, N+1);
for i = 1:length(ks)
    pk(ks(i)+1)= k_freq(i);
end
pk = pk / n_samples;
end

