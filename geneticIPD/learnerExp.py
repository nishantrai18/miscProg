import axelrod

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

numTurns = 200000
selfScore = []
oppScore = []
selfAvgList = []
oppAvgList = []

strategies = [axelrod.LookerUp(), axelrod.Cooperator()]
ply = axelrod.LearnerAxel(memory_depth = 2, exploreProb = 0.2)

print ply

for s in strategies:
	for turn in range(numTurns):
	    ply.play(p2)
	selfList = map(ScoreMatrix, zip(ply.history, p.history))
	oppList = map(ScoreMatrix, zip(p.history, ply.history))
    selfScore.append(sum(selfList))
    oppScore.append(sum(oppList))
	selfAvgList.append(GetAvgScore(selfList, 10))
	oppAvgList.append(GetAvgScore(oppList, 10))
	ply.reset()