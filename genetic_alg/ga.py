import time
import numpy as np
import matplotlib.pyplot as plt
import os

class GeneticAlg():
    def __init__(self, initial_population, fitness_fn, mutation_fn, crossover_fn,
                 selection_fn, keep_parents=0, callback=None):
        '''General genetic algorithm.
           Args:
              model: numpy array of weights to mutate, first dimension must 
                     correspond to the different solutions in the population.
              fitness_fn: fitness function. Receives as an argument a set of weights
                          in the population and outputs the relative fitness. 
              mutation_fn: Mutation function. Receives as an argument a set of
                           weights and should return the mutated weights.
              crossover_fn: Crossover function. Receives as an argument a list with
                            two sets of weights and returns a child set with 
                            crossover applied.
              selection_fn: Selection function. Receive as an argument the fitness
                            of all members of the population and 
                            should return the probability for each solution to 
                            be selected as a parent.
              keep_parents: int. Parents to be kept for elitism selection.
              callback: list of callback functions to be called after each generation.
                        Receives as input the GeneticAlg instance.'''
          
        self.population=initial_population
        self.fitness_fn=fitness_fn
        self.mutation_fn=mutation_fn
        self.crossover_fn=crossover_fn
        self.selection_fn=selection_fn
        self.keep_parents=keep_parents
        self.fitness_hist=[]
        self.best_fitness=None
        self.best_solution=None
        self.num_generations=0
        self.fitness=None
        self.callback=callback
        

    def score(self):
        '''Compute fitness and save best solution'''
        self.fitness=list(map(self.fitness_fn, self.population))
        current_best=np.amax(self.fitness)
        if self.best_fitness is None:
            self.best_fitness=np.amax(self.fitness)
            self.best_solution=self.population[np.argmax(self.fitness)]
        elif current_best>self.best_fitness:
            self.best_fitness=np.amax(self.fitness)
            self.best_solution=self.population[np.argmax(self.fitness)]
        self.fitness_hist.append((self.num_generations, current_best, 
                                  np.mean(self.fitness), np.std(self.fitness)))

    def generate_new_pop(self):
        '''Generate a new population of solutions'''
        pop_size=len(self.population)
        parents_p=self.selection_fn(self.fitness)
        elites=[]
        #The parents with the highest probability are propagated to the nex generation
        elites_indices=np.argsort(parents_p)[:self.keep_parents] 
        for idx in elites_indices:
            elites.append(self.population[idx])
        children=[]
        for j in range(pop_size-self.keep_parents):
            parents_idx=np.random.choice(pop_size, size=2, 
                                     replace=False, p=parents_p)
            parents=[self.population[parents_idx[0]], self.population[parents_idx[1]]]
            child=self.crossover_fn(parents)
            child=self.mutation_fn(child)
            children.append(child)
        self.population=elites+children
         
    def run(self, num_generations, verbose=True, log_freq=1, plot_fitness=False):
        '''Run the genetic algorithm'''
        counter=1
        if self.fitness==None:
            self.score() #score once before the training
        start=time.time()
        for i in range(num_generations):
            self.generate_new_pop()    
            self.score()
            #compute time passed and expected
            now=time.time()
            extimated=((now-start)/counter)*(num_generations-counter)
            h=(now-start)//3600
            m=(now-start-h*3600)//60
            s=(now-start-h*3600-m*60)//1
            he=extimated//3600
            me=(extimated-he*3600)//60
            se=(extimated-he*3600-me*60)//1
            counter+=1
            self.num_generations+=1
            #print log            
            if verbose:
                if self.num_generations % log_freq==0:
                  print("-"*79)
                  print('Generation', self.num_generations)
                  print("-"*79)
                  print('Current best fitness:{},\
                         \nFitness mean:{},\nFitness std:{},\
                         \nAll-time best fitness:{}.'.format(
                          self.fitness_hist[-1][1],
                          self.fitness_hist[-1][2], 
                          self.fitness_hist[-1][3],
                          self.best_fitness)
                          )
                  print('Elapsed Time:{}h:{}m:{}s, Estimated to completion:{}h,{}m, {}s'.format(int(h),int(m),int(s),
                                                int(he),int(me),int(se)))
                  if plot_fitness:
                      plt.hist(self.fitness)
                      plt.xlabel('Fitness')
                      plt.title('Population fitness histogram')
                      plt.show()
            if self.callback is not None:
                for call in self.callback:
                    call(self)

    def save(self, dir):
        if not os.path.exists(dir+'/saved_model'):
            os.makedirs(dir+'/saved_model')
        np.save(dir+'/saved_model/population', self.population)
        np.save(dir+'/saved_model/fitness', self.fitness)
        np.save(dir+'/saved_model/generation', self.num_generations)
        np.save(dir+'/saved_model/hist', self.fitness_hist)
        np.save(dir+'/saved_model/best_fitness', self.best_fitness)
        np.save(dir+'/saved_model/best_sol', self.best_solution)
        print('Model saved')
