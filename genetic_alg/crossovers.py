import numpy as np

def single_point_crossover(parents, crossover_p):
    if np.random.random()<crossover_p:
        l=len(parents[0])
        point=np.random.choice(l)
        child=np.concatenate((parents[0][:point],parents[1][point:]))
    else:
        child=parents[0]
    return child