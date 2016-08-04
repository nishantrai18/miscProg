import axelrod
import numpy

# strategies = [s() for s in axelrod.basic_strategies]

# strategies = [axelrod.LookerUp(), axelrod.Cooperator(), axelrod.Defector(), axelrod.CyclerCCCD(), axelrod.CyclerCCD(), axelrod.HardGoByMajority(), axelrod.GoByMajority(), axelrod.HardTitForTat(), \
#  axelrod.Random(), axelrod.TitForTat(), axelrod.TitFor2Tats(), axelrod.Prober(), axelrod.Prober2(), axelrod.Prober3()]

# strategies = [axelrod.RiskyQLearner(), axelrod.LookerUp(), axelrod.Cooperator(), axelrod.Defector(), axelrod.HardGoByMajority(), \
#  axelrod.Random(), axelrod.TitForTat()]

# strategies = [axelrod.Cooperator(), axelrod.Defector(), axelrod.CyclerCCD(), axelrod.HardGoByMajority(), axelrod.GoByMajority(), \
# axelrod.HardTitForTat(), axelrod.TitFor2Tats(), axelrod.SuspiciousTitForTat(), axelrod.Random(), axelrod.TitForTat(), axelrod.Prober()]

strategies = [axelrod.Cooperator(), axelrod.Defector(), axelrod.CyclerCCD(), axelrod.HardTitForTat(), \
 axelrod.TitFor2Tats(), axelrod.SuspiciousTitForTat(), axelrod.Random(), axelrod.TitForTat(), axelrod.Prober()]

#strategies = []

# strategies = [axelrod.LearnerAxel(memory_depth = 2, exploreProb = 0.2), axelrod.LookerUp()]
# strategies.append(axelrod.LearnerAxel(memory_depth = 2, exploreProb = 0.1))

#evolveCode = [0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0,\
# 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]						# For 8 basic strategies

#evolveCode = [1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1,\
# 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1]					# For all 112 strategies

#evolveCode = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0,\
# 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]						# For all strategies

# evolveCode = [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1,\
# 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0]			# For the restricted strategies

# evolveCode = [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1,\
#  1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0]							# For some restricted codes

evolveCode = [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,\
 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0]					# For final experiments

singEvolveCode = [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1,\
 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0]

multAxel = axelrod.EvolveAxel(3, evolveCode, 'MULT')
strategies.append(multAxel)
singAxel = axelrod.EvolveAxel(3, singEvolveCode, 'SING')
strategies.append(singAxel)
strategies.append(axelrod.LearnerAxel(memory_depth = 2, exploreProb = 0.1, learnerType = 2))

tournament = axelrod.Tournament(strategies)
results = tournament.play()

print results.scores
name = results.players
scor = results.normalised_scores

for i in range(len(scor)):
	print name[i], numpy.mean(scor[i])

plot = axelrod.Plot(results)
p = plot.boxplot()
p.show()
raw_input('Next')

p = plot.winplot()
p.show()
raw_input('Next')

p = plot.payoff()
p.show()
raw_input('Next')
