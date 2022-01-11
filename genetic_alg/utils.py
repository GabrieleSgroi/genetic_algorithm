import numpy as np
from genetic_alg.ga import GeneticAlg

def flatten_weights(model):
    '''Returns the flattened weights of a Keras model'''
    chromosome=np.empty(0)
    for layer in model.get_weights():
        chromosome=np.append(chromosome, layer)
    return chromosome

def reshape_weights(model, params):
    '''Reshape flattened weights into the correct shape for the given Keras
       model'''
    w_nest=[]
    pointer=0
    for layer in model.get_weights():
        shape=layer.shape
        total=1
        for d in shape:
            total*=d
        w_nest.append(np.reshape(params[pointer:pointer+total], shape))
        pointer+=total
    return w_nest

def load_ga(dir, fitness_fn, mutation_fn, crossover_fn,
                 selection_fn, keep_parents=0, callback=None):
    '''Returns the loaded GeneticAlg instance'''
    loaded=GeneticAlg([], fitness_fn, mutation_fn, crossover_fn,
                 selection_fn, keep_parents=0, callback=callback)
    loaded.population=np.load(dir+'/saved_model/population.npy' ).tolist()
    loaded.fitness=np.load(dir+'/saved_model/fitness.npy' ).tolist()
    loaded.num_generations=np.load(dir+'/saved_model/generation.npy')
    loaded.fitness_hist=np.load(dir+'/saved_model/hist.npy').tolist()
    loaded.best_fitness=np.load(dir+'/saved_model/best_fitness.npy')
    loaded.best_solution=np.load(dir+'/saved_model/best_sol.npy')
    
    return loaded
