import matplotlib.pyplot as plt
import axelrod
import cProfile
import numpy

numpy.set_printoptions(precision=3)

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

def GetAvgScore(scoreList, turns):
	sumList = []
	avgList = []
	tmpSum = 0
	for i in range(len(scoreList)):
		tmpSum += scoreList[i]
		sumList.append(tmpSum)
	for i in range(len(scoreList)-turns):
		avgList.append( (((sumList[i+turns] - sumList[i])*(1.0))/turns) )
	return avgList

numTurns = 20000
selfScore = []
oppScore = []
selfAvgList = []
oppAvgList = []
avgScore = []
oppAvgScore = []

evolveCode = [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,\
 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0]					# For final experiments

singEvolveCode = [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1,\
 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0]

# strategies = [axelrod.Cooperator(), axelrod.Defector()]
# strategies = [axelrod.TitFor2Tats(), axelrod.SuspiciousTitForTat(), axelrod.TitForTat(), axelrod.Prober()]
strategies = [axelrod.Cooperator(), axelrod.Defector(), axelrod.CyclerCCD(),axelrod.HardTitForTat(),\
axelrod.TitFor2Tats(), axelrod.SuspiciousTitForTat(), axelrod.Random(), axelrod.TitForTat(), axelrod.Prober()]

learner = axelrod.LearnerAxel(memory_depth = 2, exploreProb = 0.1, learnerType = 2)
multAxel = axelrod.EvolveAxel(3, evolveCode, 'MULT')
singAxel = axelrod.EvolveAxel(3, evolveCode, 'SING')

ply = learner

# print ply

for p in strategies:
	print "Currently playing against strategy:", p
	for turn in range(numTurns):
		ply.play(p)
	selfList = map(ScoreMatrix, zip(ply.history, p.history))
	oppList = map(ScoreMatrix, zip(p.history, ply.history))
	selfScore.append(sum(selfList))
	oppScore.append(sum(oppList))
	selfAvgList.append(GetAvgScore(selfList, 1000))
	oppAvgList.append(GetAvgScore(oppList, 1000))
	avgScore.append(numpy.mean(selfList))
	oppAvgScore.append(numpy.mean(oppList))
	ply.reset()

avgScore = numpy.array(avgScore)
oppAvgScore = numpy.array(oppAvgScore)

for i in range(len(strategies)):
	print strategies[i], '&',

print

for i in range(4):
	print avgScore[i], '(', oppAvgScore[i], ') &',

print

for i in range(4,len(avgScore)):
	print avgScore[i], '(', oppAvgScore[i], ') &',

print

# fig = plt.figure()
# st = fig.suptitle("Learning Performance against multiple strategies", fontsize="x-large")

# axn = range(len(selfAvgList[0]))
# for i in range(len(selfAvgList)):
# 	ax = fig.add_subplot(2,1,i+1)
# 	ax.set_title(strategies[i])
# 	ax.plot(axn, selfAvgList[i], color = 'blue')
# 	ax.plot(axn, oppAvgList[i], color = 'red')
# 	# plt.ylim(0,5)
# 	# ax.set_xscale('log')
# # plt.plot(ax, oppAvgList[0])

# plt.show()

# plt.savefig("learnerInitial.eps", bbox_inches="tight", format = 'eps')
# plt.savefig("learnerInitial.png", bbox_inches="tight", format = 'png')