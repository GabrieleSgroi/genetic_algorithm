import numpy as np

def random_mutation(sol, mutation_rate, range):
    l=len(sol)
    sol=np.array(sol)
    N=np.random.binomial(l, mutation_rate)
    mutation_idx=np.random.choice(l, replace=False, size=N)
    additions=np.random.rand(N)*(range[1]-range[0])+range[0]
    sol[mutation_idx]+=additions
    sol=sol.tolist()
    return sol

def gaussian_mutation(sol, mutation_rate, std):
    l=len(sol)
    sol=np.array(sol)
    N=np.random.binomial(l, mutation_rate)
    mutation_idx=np.random.choice(l, replace=False, size=N)
    additions=np.random.normal(loc=0., scale=std, size=N)
    sol[mutation_idx]+=additions
    sol=sol.tolist()
    return sol
