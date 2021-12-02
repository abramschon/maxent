# Maximum entropy models

This repo contains code related to my Master's project. At this stage, the aim is for code which illustrates an understanding of how maximum entropy models (MaxEnt models) work and can be fitted to data, as opposed to code which will replace existing maximum entropy packages. 
For examples of existing MaxEnt packages see: 

- [maxent_toolbox](https://orimaoz.github.io/maxent_toolbox/) (Matlab)
- [ConIII](https://eddielee.co/coniii/index.html) (Python)
- [CorBinian](https://github.com/mackelab/CorBinian) (Matlab)

The code is structured into the following directories:

- notebooks: jupyter notebooks 
- maxent: main Python code with MaxEnt implementations
- matlab: code in Matlab that makes use of the maxent_toolbox package.
- wolfram: exploration of analytic solutions for Ising, indepenent and population count models using the wolfram language
- results: collection of plots, saved weights and other outputs
- tests: tests for the code (PyTest)
- data: data that we use to train the maxent models, and files to make things reproducible