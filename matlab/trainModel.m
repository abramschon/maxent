%% load data
clear
load ../data/shuffled_data/data2014

[total_N, obvs] = size(train_reps);

%% Fitting models
% We need to specify a model name 'name', number of neurons 'NN', which repeat of the
% train this is 'rep'(defining which subset we train on), and whether we
% use data shuffled over time or shuffled over stimulus 'shuffle'. 
% We then need to save the matlab models, and the model weights as csv
% files. We will name the files `shuffle_NN_name_rep`

for NN = 10:21 % define range of N interested in fitting
    for rep = 10:20 % define which subset of the data to train on
        shuffle = 'stimulus'; % how the data is shuffled: 'stimulus' or 'time'
        name = 'indep'; % 'indep' 'ksync' 'pairwise' 'third' or 'kpairwise'
        file_name = "../data/trained_models/" + shuffle + "_" + NN + "_" + name + "_" + rep;

        % select which neurons activities to train model on
        rng(rep) % set seed based on rep - is this a good idea to do?
        id_N = randperm(total_N, NN); % id of neurons activity we are going to train on

        train = train_reps(id_N,:); % select the training data
        
        if strcmp(name, 'third')
            correlations = cat(1,num2cell(nchoosek(1:NN,1),2), ...
                num2cell(nchoosek(1:NN,2),2),...
                num2cell(nchoosek(1:NN,3),2));

                model = maxent.createModel(NN,'highorder',correlations);
        else   
            model = maxent.createModel(NN, name); % declare the model
        end


        if strcmp(name,'ksync')
            model = getKSync(train);
        else
            % model = maxent.trainModel(model, train, 'threshold', 1, 'savefile', '../data/trained_models/checkpoint'); %train model while saving to file
            model = maxent.trainModel(model, train, 'threshold', 1); %train model without saving to file
        end

        % save model weights and save model
        save(file_name, 'model'); % save model
        weights = maxent.getFactors(model); % get model weights
        writematrix(weights, file_name+'.csv'); % save model weights 
    end
end