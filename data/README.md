# Data folder

The data for this project was downloaded from [research explorer](https://research-explorer.app.ist.ac.at/record/5562) and was collected as part of the 2014 paper *Searching for Collective Behavior in a Large Network of Sensory Neurons* by Tkačik et al. 

## Considerations
- As this is not my data, I am not sharing it until I am better informed.

- However, I want my code to be reproducible and others should have access to the same inputs to ensure the same outputs are produced.

- Finally, certain bits of data are too large to store so I wouldn't want to push them to github.

## Organisation

The data folder is organised as follows:
- raw_data:

    `Offline`: this data should be re-downloaded

    Contains raw data downloaded from research explorer.

- shuffled_data: 
    
    `Offline`: derived from raw data

    Contains the two shuffled datasets processed in matlab as .csv and .mat files the full unshuffled dataset in numpy format.
- subsets: 
    
    `Offline`: quickly determined via subsets.m code

    Defines the ID numbers for the different subsets

- pop:

    `Online`

    Saved matlab weights for the proof of principle (P.O.P) notebook

- trained_models:

    `Online`: useful to have a common repo since training might happen on different machines and weights are small files.

    Save the models trained in matlab as .mat and also their weights as .csv

To do (if it's worth it): 
- expectations:

    `Online`

    Extract expectations from different subsets so they can easily be read in for training python models 


