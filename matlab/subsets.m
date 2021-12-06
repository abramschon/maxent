% select different subsets of the data and save
clear
load("../data/shuffled_data/data2014")
[total_N, obvs] = size(train_reps);
data_path = "../data/subsets/";

for NN = [10 40 100]
    for rep = 1:30
        rng(rep) %importantly, these correspond to the seeds used to train the maxent models
        id_N = randperm(total_N, NN); % id of neurons
        file_name = data_path + NN + "_" + rep;
        writematrix(id_N, file_name+'.csv'); % save model weights 
    end
end