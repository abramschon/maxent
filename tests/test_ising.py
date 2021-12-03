import time
import numpy as np
import matplotlib.pyplot as plt


from maxent.numeric_ising import Ising

def test_example():
    N = 5
    avgs = 0.5*np.ones(N) # prob of every neuron firing in a window is 0.5
    corrs = 0.2*np.triu(np.ones((N,N)),1) # prob of 2 neurons firing in the same window is 0.2 
    
    print("Init model")
    ising = Ising(N, avgs, corrs, lr=0.5) 
    
    print("Starting grad ascent")
    start = time.time()
    for _ in range(10):
        ising.gradient_ascent() # 500 steps 
    print(f"Stop grad ascent: {time.time()-start}s")
    
    pred_avgs = ising.averages()
    pred_corrs = ising.correlations()
    print("Predicted averages:", pred_avgs, "Predicted correlations:", pred_corrs,sep="\n")
    
    print(f"P({ising.states[0]})={ising.p(ising.states[0])}") # check it predicts 0 state

    # check it fits relatively well
    np.testing.assert_almost_equal(pred_avgs, avgs, decimal=2)
    np.testing.assert_almost_equal(pred_corrs, corrs, decimal=2)


def av_time_grad_ascent():
    # Calculate average times
    reps = 20
    startN = 8
    stopN = 10 #inclusive
    Ns = np.arange(startN,stopN+1)
    times = np.zeros( (reps,len(Ns)) )
    for i in range(reps):
        if not (i+1)%10:
            print("Repetitions: ", i+1)
        for N in Ns:
            avgs = 0.5*np.ones(N) # prob of every neuron firing in a window is 0.5
            corrs = 0.2*np.triu(np.ones((N,N)),1) # prob of 2 neurons firing in the same window is 0.2 
            ising = Ising(N, avgs, corrs, lr=0.5) 
            start = time.time()
            ising.gradient_ascent() # 500 steps 
            stop = time.time()
            times[i,N-startN]=stop-start
    
    av_times = np.mean(times,0)
    std_times = np.std(times,0)

    plt.plot(Ns, av_times, "k.")
    plt.plot(Ns, av_times+2*std_times/np.sqrt(reps), "r_")
    plt.plot(Ns, av_times-2*std_times/np.sqrt(reps), "r_")
    plt.title("Time for 100 steps of grad. ascent vs. system size")
    plt.xlabel("System size")
    plt.ylabel("Time (seconds)")
    plt.show()

