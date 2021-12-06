%% load a subset of 10 neurons from the data
clear
data_path = "../data/";
load(data_path+"shuffled_data/data2014")

% we are going to use the data where all observations over all repeats were shuffled
NN = 10;
train = train_rand(1:NN,:); % select activity of the first 10 neurons
test = test_rand(1:NN,:);

size(train)

%% ============= independent model
m_indep = maxent.createModel(NN, 'indep'); %create model
m_indep = maxent.trainModel(m_indep, train); %train model

% compare marginals
m_marginals = maxent.getMarginals(m_indep)
data_marginals = maxent.getEmpiricalMarginals(train, m_indep)

% get model weights
weights = maxent.getFactors(m_indep);
writematrix(weights, data_path+'pop/indep.csv')

dist = maxent.getExplicitDistribution(m_indep);
writematrix(dist, data_path+'pop/indep_dist.csv')



%% ============= p(K)
m_ksync = maxent.createModel(NN,'ksync'); %create model
m_ksync = maxent.trainModel(m_ksync, train); %train model

% compare marginals
m_marginals = maxent.getMarginals(m_ksync)
data_marginals = maxent.getEmpiricalMarginals(train, m_ksync)

% get model weights
weights = maxent.getFactors(m_ksync);
writematrix(weights, data_path+'pop/ksync.csv')

% get explicit distribution
dist = maxent.getExplicitDistribution(m_ksync);
writematrix(dist, data_path+'pop/ksync_dist.csv')


%% ============= pairwise model
m_pwise = maxent.createModel(NN,'pairwise');
m_pwise = maxent.trainModel(m_pwise, train); %train model

% compare marginals
m_marginals = maxent.getMarginals(m_pwise)
data_marginals = maxent.getEmpiricalMarginals(train, m_pwise)

% get model weights
weights = maxent.getFactors(m_pwise);
writematrix(weights, data_path+'pop/m_pwise.csv') 

% get explicit distribution
dist = maxent.getExplicitDistribution(m_pwise);
writematrix(dist, data_path+'pop/m_pwise_dist.csv')

