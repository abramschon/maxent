%% Reading in the data from 2014 Searching ...
clear
load ../data/raw_data/2014SearchingCollective/bint

[reps, n_cells, time] = size(bint);
data = permute(bint, [2 3 1]);  % changes the order of axes so that it is n_cells, time, reps
% this makes it easier to collapse the time and reps axis using reshape
size(data)

% we can either reserve a proportion of the repeats and see whether the
% model generalises across repeats, or we can reserve a proportion of the
% frames at random so we see whether the model generalises 

% reserve repeats
n_train_reps = ceil(0.8 * reps); % proportion of reps we keep for training
idx_train = randperm(reps,n_train_reps);
idx_test = setdiff(1:reps,idx_train);
train_reps = reshape(data(:,:,idx_train), n_cells, []);
test_reps = reshape(data(:,:,idx_test), n_cells, []);
size(test_reps)
size(test_reps)

tot = 0;
for i=1:10
    tot = tot + sum( train_reps(:,i) - data(:,i,idx_train(1)) ); % these should match up, hence their sum should be 0
end
tot

% reserve random
data_comb = reshape(data, n_cells, []); %combine time and repetitions
n_samples = reps*time;
n_train_samples = ceil(0.8 * n_samples); % proportion of reps we keep for training
idx_train = randperm(n_samples,n_train_samples);
idx_test = setdiff(1:n_samples,idx_train);
train_rand =data_comb(:,idx_train);
test_rand = data_comb(:,idx_test);
size(train_rand)
size(test_rand)

save ../data/shuffled_data/data2014 train_reps test_reps train_rand test_rand

%% Write to csv
clear
data_path = "../data/shuffled_data/";
load(data_path + "data2014")

writematrix(train_rand, data_path+'time_shuffle_train_1.csv')
writematrix(train_reps, data_path+'repeat_shuffle_train_1.csv')
writematrix(test_rand, data_path+'time_shuffle_test_1.csv')
writematrix(test_reps, data_path+'repeat_shuffle_test_1.csv')
