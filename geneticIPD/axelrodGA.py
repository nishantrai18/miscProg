import axelrod
import random
import numpy
import matplotlib.pyplot as plt
import pickle

from deap import base
from deap import creator
from deap import tools

scores = {}
scores['C'] = {}
scores['D'] = {}
scores['C']['C'] = (3,3)
scores['D']['C'] = (5,0)
scores['C']['D'] = (0,5)
scores['D']['D'] = (1,1)

def ScoreMatrix(moves):
    moveA = moves[0]
    moveB = moves[1]
    return scores[moveA][moveB][0]

# strategies = [s() for s in axelrod.basic_strategies]
#strategies = [axelrod.Cooperator(), axelrod.Defector(), axelrod.CyclerCCCD(), axelrod.CyclerCCD(), \
# axelrod.HardGoByMajority(), axelrod.GoByMajority(), axelrod.HardTitForTat(), axelrod.Random(), axelrod.TitForTat(), \
# axelrod.SuspiciousTitForTat(), axelrod.TitFor2Tats(), axelrod.Prober(), axelrod.Prober2(), axelrod.Prober3()]

strategies = [axelrod.Cooperator(), axelrod.Defector(), axelrod.CyclerCCD(), axelrod.HardGoByMajority(), axelrod.GoByMajority(), \
axelrod.HardTitForTat(), axelrod.TitFor2Tats(), axelrod.SuspiciousTitForTat(), axelrod.Random(), axelrod.TitForTat(), axelrod.Prober()]

creator.create("FitnessMax", base.Fitness, weights=(1.0,))          #Weight is a tuple, so here we use a single objective
creator.create("Individual", list, fitness=creator.FitnessMax)      #Define the structure of the individuals

toolbox = base.Toolbox()            #Initialize it?

memLength = 3

toolbox.register("attr_bool", random.randint, 0, 1)             # Random boolean
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, (4**memLength) + 2*memLength)        # 100 denotes the size of the list (or individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)              # Defines the population

def evalOneMax(individual):
    ply = axelrod.EvolveAxel(memLength, individual)
    val = [0, 0]
    for p in strategies:
        for i in range(0,100):
            ply.play(p)
        val[0] += sum( map( ScoreMatrix, zip(ply.history, p.history) ) )
        val[1] += sum( map( ScoreMatrix, zip(p.history, ply.history) ) )
        ply.history = []
        p.history = []
        ply.cooperations = 0
        ply.defections = 0
        p.cooperations = 0
        p.defections = 0

    val[0] = (val[0]*(1.0))/len(strategies)
    val[1] = (val[1]*(1.0))/len(strategies)
    # return (val[0], val[1])             #, to emphasise the tuple nature
    return (val[0],)             #, to emphasise the tuple nature

toolbox.register("evaluate", evalOneMax)                # Define the evaluation
toolbox.register("mate", tools.cxTwoPoint)              # Define the crossover operation, already present for lists
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)                # Flip bit operation already defined
toolbox.register("select", tools.selTournament, tournsize=3)            # 3 signifies that the best 3 are considered for the next generation

#----------

def main():
    random.seed(64)
    pop = toolbox.population(n = 60)

    # CXPB  is the probability with which two individuals are crossed
    # MUTPB is the probability for mutating an individual
    # NGEN  is the number of generations for which the evolution runs
    CXPB, MUTPB, NGEN = 0.75, (2.0/((4**memLength) + 2*memLength)), 250
    # CXPB, MUTPB, NGEN = 0.75, (1.0/70), 100
    
    avgFitList = []
    minFitList = []
    maxFitList = []
    fitList = []
    popList = []

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
        
        fitList = [ind.fitness.values for ind in pop]               # Gather all the fitnesses of the current population in one list
        if (g == 0):
            oldFitList = [ind.fitness.values for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        print("  Min %s" % min(fits)),
        print("  Max %s" % max(fits)),
        print("  Avg %s" % mean),
        print("  Std %s" % std)

        avgFitList.append(mean)
        minFitList.append(min(fits))
        maxFitList.append(max(fits))
    
    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

    # target = open('multObj.p', 'wb')
    # pickle.dump([avgFitList, minFitList, maxFitList], target)

    # for p in popList:
    #     print p

    # gen = range(0,NGEN)
    # fig, ax1 = plt.subplots()
    # # line1 = ax1.plot(gen, minFitList, "b-", label="Minimum Fitness")
    # l1 = ax1.plot(gen, maxFitList, "b-", label="Maximum Fitness")
    # l2 = ax1.plot(gen, avgFitList, "r-", label="Average Fitness")
    # ax1.set_xlabel("Generation")
    # # ax1.set_ylabel("Fitness", color="b")
    # for tl in ax1.get_yticklabels():
    #     tl.set_color("b")

    # # ax2 = ax1.twinx()
    # # line2 = ax2.plot(gen, avgFitList, "r-", label="Average Fitness")
    # # ax2.set_ylabel("Size", color="r")
    # # for tl in ax2.get_yticklabels():
    # #     tl.set_color("r")

    # lns = l1 + l2
    # labs = [l.get_label() for l in lns]
    # ax1.legend(lns, labs, loc="center right")

    # plt.show()

if __name__ == "__main__":
    main()
