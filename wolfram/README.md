# Jupyter Notebooks with the Wolfram engine

We use this [package](https://github.com/WolframResearch/WolframLanguageForJupyter) to allow us to use the Wolfram Language in Jupyter. In short:
- Clone the repo

    `git clone https://github.com/WolframResearch/WolframLanguageForJupyter.git`

- Navigate into the repo and run:
    `./configure-jupyter.wls add`

This allows you to select the wolfram engine in Jupyter.
To get help with the Wolfram Language, see the [documentation](https://reference.wolfram.com/language/).

## Context
We create various maximum entropy models which model the joint probability distribution over a set of binary neurons. The Ising model is the maximum entropy distribution that reproduces the *mean probability of neurons spiking*, and the *pairwise correlations*. We also consider the maximum entropy distribution that reproduces the *probability that K cells spike*, the population count model, and distribution that reproduces the *mean probability of neurons spiking*, the independent model.

Though we can easily express the parameters of the independent and population count model in terms of their constraints, it is difficult to do the same for the Ising model, which we illustrate by finding analytic solutions for the parameters `h` (a vector), and `J` (a matrix) for small N.

## Notebooks:
- `analytic_ising.ipynb` -  Implementation of Ising model that analytically solves for `h` and `J`.
- `coarse_models.ipynb` - Implementation of independent and population count models that analytically determines their parameters.
