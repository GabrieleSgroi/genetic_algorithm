import numpy as np

def rank_selection(fitness):
    N=len(fitness)
    p=np.zeros(N)
    sorted_idx=np.argsort(fitness)
    for i in range(N):
        p[sorted_idx[i]]+=2*(i+1)/(N*(N+1))
    return p

def roulette_wheel(fitness):
    N=len(fitness)
    sum=np.sum(fitness)
    p=np.zeros(N)
    for i in range(N):
        p[i]+=fitness[i]/sum
    return p

def boltzmann_selection(fitness, T):
    f=np.array(fitness)
    norm=np.sum(np.exp(f/T))
    p=np.exp(f/T)/norm
    return p
