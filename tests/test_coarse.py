import time
import numpy as np

from maxent.coarse_models import Independent, PopCount

def test_indep_example():
    N = 10
    avgs = 0.5*np.ones(N) # prob of every neuron firing in a window is 0.5
    
    print("Init model")
    start = time.time()
    ind = Independent(N, avgs) 
    print(f"Time: {time.time()-start}s")

    pred_avgs = ind.averages()
    print("Predicted averages:", pred_avgs, sep="\n")
    print(f"P({ind.states[0]})={ind.p(ind.states[0])}")

    np.testing.assert_almost_equal(pred_avgs, avgs, decimal=2)

def test_indep_sampling():
    N = 10
    avgs = 0.5*np.ones(N) # prob of every neuron firing in a window is 0.5
    ind = Independent(N, avgs) 
    samples = ind.get_samples() #by default generates 4 * 100 000 samples
    sample_avgs = np.mean(samples,axis=0)

    np.testing.assert_almost_equal(sample_avgs, avgs, decimal=1) # will fail occasionally


def test_pop_count_example():
    N = 10
    p_K = np.ones(N+1)/(N+1)  # all counts are equally likely
    print("Init model")
    start = time.time()
    pop = PopCount(N,p_K)
    print(f"Time: {time.time()-start}s")

    pred_p_K = pop.prob_k()
    print("P(K):", pred_p_K, sep="\n")
    print(f"P({pop.states[0]})={pop.p(pop.states[0])}")

    np.testing.assert_almost_equal(pred_p_K, p_K, decimal=2)

def test_pop_count_sampling():
    N = 10
    p_K = np.ones(N+1)/(N+1)  # all counts are equally likely
    pop = PopCount(N,p_K)
    samples = pop.get_samples()
    sample_p_K = np.bincount(np.sum(samples,axis=1).astype(int))/len(samples)

    np.testing.assert_almost_equal(sample_p_K, p_K, decimal=2)


