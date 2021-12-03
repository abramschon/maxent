import numpy as np

from maxent.three_wise import ThreeWise 

def test_runs():
    N = 4
    h = np.random.random_sample((N))
    J = np.triu( np.random.random_sample((N,N)), 1)
    K = np.zeros((N,N,N))

    alpha = 1
    for i in range(N-2):
        for j in range(i+1,N-1):
            for k in range(j+1, N):
                K[i,j,k] = alpha 
                alpha /= 10
    ex = ThreeWise(N, h, J, K)

    #calc 3 wise correlations
    for i in range(N-2):
        for j in range(i+1,N-1):
            for k in range(j+1, N):
                print(f"Correlation between {i},{j},{k}\n", ex.expectation(lambda s: s[:,i]*s[:,j]*s[:,k]) )