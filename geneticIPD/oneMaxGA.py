#    example which maximizes the sum of a list of integers
#    each of which can be 0 or 1

import random

from deap import base
from deap import creator
from deap import tools

creator.create("FitnessMax", base.Fitness, weights=(1.0,))          #Weight is a tuple, so here we use a single objective
creator.create("Individual", list, fitness=creator.FitnessMax)      #Define the structure of the individuals

toolbox = base.Toolbox()            #Initialize it?

# Attribute generator 
#                      define 'attr_bool' to be an attribute ('gene')
#                      which corresponds to integers sampled uniformly
#                      from the range [0,1] (i.e. 0 or 1 with equal
#                      probability)
toolbox.register("attr_bool", random.randint, 0, 1)             # Random boolean

# Structure initializers
#                         define 'individual' to be an individual
#                         consisting of 100 'attr_bool' elements ('genes')
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 100)        # 100 denotes the size of the list (or individual)

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)              # Defines the population

# the goal ('fitness') function to be maximized
def evalOneMax(individual):
    return (sum(individual),)             #, to emphasise the tuple nature

#----------
# Operator registration
#----------
# register the goal / fitness function
toolbox.register("evaluate", evalOneMax)                # Define the evaluation

# register the crossover operator
toolbox.register("mate", tools.cxTwoPoint)              # Define the crossover operation, already present for lists

# register a mutation operator with a probability to
# flip each attribute/gene of 0.05
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)                # Flip bit operation already defined

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selTournament, tournsize=3)            # 3 signifies that the best 3 are considered for the next generation

#----------

def main():
    random.seed(64)

    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=300)                         

    # CXPB  is the probability with which two individuals are crossed
    # MUTPB is the probability for mutating an individual
    # NGEN  is the number of generations for which the evolution runs
    CXPB, MUTPB, NGEN = 0.5, 0.2, 40
    
    print("Start of evolution")
    
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    print("  Evaluated %i individuals" % len(pop))
    
    # Begin the evolution
    for g in range(NGEN):
        print("-- Generation %i --" % g)
        
        offspring = toolbox.select(pop, len(pop))               # Select the next generation individuals
        offspring = list(map(toolbox.clone, offspring))         # Clone the selected individuals
    
        for child1, child2 in zip(offspring[::2], offspring[1::2]):             # Apply crossover and mutation on the offspring
            if random.random() < CXPB:                                              # cross two individuals with probability CXPB
                toolbox.mate(child1, child2)
                # fitness values of the children must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:                             # mutate an individual with probability MUTPB
                toolbox.mutate(mutant)
                del mutant.fitness.values
    
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        print("  Evaluated %i individuals" % len(invalid_ind))
        
        pop[:] = offspring          # The population is entirely replaced by the offspring
        fits = [ind.fitness.values[0] for ind in pop]               # Gather all the fitnesses in one list and print the stats

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
    
    print("-- End of (successful) evolution --")
    
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

if __name__ == "__main__":
    main()
