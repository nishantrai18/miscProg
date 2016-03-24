import axelrod
from axelrod import Actions, Player, init_args, random_choice
import operator
import random

C, D = Actions.C, Actions.D

def encode(moveA, moveB):
    choice = moveA + moveB
    if (choice == 'DD'):
        return 'P'
    elif (choice == 'DC'):
        return 'T'
    elif (choice == 'CD'):
        return 'S'
    elif (choice == 'CC'):
        return 'R'
    else:
        return '0'

def decode(moveList, memLength):
    val = 0
    for m in moveList:
        val *= 4
        if (m == 'P'):
            val += 0
        elif (m == 'T'):
            val += 1
        elif (m == 'S'):
            val += 2
        elif (m == 'R'):
            val += 3
        else:
            raw_input("ERROR DURING DECODE. WHAT TO DO NEXT?")
    return val

def decodeMove(choice):
    if (choice):
        return C
    else:
        return D

def encodeMove(choice):
    if (choice == C):
        return True
    else:
        return False

def correction(num):
    ans = 0.1*(1 + 10*((2.0)**(-0.02*(num**(0.5)))))
    return ans

class Learner(Player):
    """A player uses a predefined array of choices. It considers the k previous
    moves and decides which move to make based on the provided list.

    An optional memory attribute (Default = 3) will limit the number of turns remembered.
    """

    classifier = {
        'stochastic': True,
        'inspects_source': False,
        'makes_use_of': set(),
        'manipulates_source': False,
        'manipulates_state': False,
        'memory_depth': 3  # memory_depth may be altered by __init__
    }

    @init_args
    def __init__(self, memory_depth = 3, exploreProb = 0.1):
        """
        Parameters
        ----------
        memory_depth, int >= 0
            The number of rounds to use for the calculation of the cooperation
            and defection probabilities of the opponent.
        """

        Player.__init__(self)
        self.qTabSize = (4**memory_depth)
        # qTabSize : It refers to the size of the qTable. It consists of all possible states in the game
        self.qTab = [dict({True: 0, False: 0}) for i in range(0, self.qTabSize)]
        # qTab : The QTable which stores the Q values. It's updated during gameplay
        self.turns = [dict({True: 0, False: 0}) for i in range(0, self.qTabSize)]
        self.classifier['memory_depth'] = memory_depth
        # The memory depth to consider while playing the game
        self.memory = self.classifier['memory_depth']
        self.explore = exploreProb
        # Initialize the payoff matrix for the game
        (R, P, S, T) = self.tournament_attributes["game"].RPST()
        self.payoff = {C: {C: R, D: S}, D: {C: T, D: P}}
        self.prevState = 0
        self.prevAction = False
        print self.payoff

    def strategy(self, opponent):
        """
        A player chooses the best action based on qTab a predefined array
        of choices. It considers the k previous moves and decides which move 
        to make based on the computed Q Table.
        The length of the state list is 4^mem
        """
        
        mem = self.memory
        randomPlay = False
        opHistory = opponent.history[-mem:]
        myHistory = self.history[-mem:]
        
        if ((len(opHistory) < mem) or (len(myHistory) < mem)):
            randomPlay = True

        # In case the memory isn't enough, play a random move
        if (randomPlay):    
            return random_choice()

        # print (self.prevState, self.prevAction, opponent.history[-1], self.payoff[decodeMove(self.prevAction)][opponent.history[-1]])

        # Update the q table when results of the previous turn are available
        self.qTabUpdate(self.prevState, self.prevAction, self.payoff[decodeMove(self.prevAction)][opponent.history[-1]])

        choice = []
        for i in range(0,mem):
            choice.append(encode(myHistory[i], opHistory[i]))
            # Get the encoding for the state
        ids = decode(choice, mem)
        # print ids

        self.totTurns += 1
        # if (self.totTurns%1000 == 0):
        #     print self.totTurns, self.explore*correction(self.totTurns)

        self.prevState = ids
        if (random.random() < self.explore*correction(self.totTurns)):
            self.prevAction = encodeMove(random_choice())
            # print self.prevAction
        else:
            self.prevAction = max(self.qTab[ids].iteritems(), key=operator.itemgetter(1))[0]

        return decodeMove(self.prevAction)

    def qTabUpdate(self, state, action, reward):
        """
        Performs the update of the Q Table
        """
        self.turns[state][action] += 1.0
        # print self.qTab
        # print state, action, reward
        self.qTab[state][action] += (1.0/self.turns[state][action])*(reward - self.qTab[state][action])

    def __repr__(self):
        """The string method for the strategy."""
        name = 'LearnMem' + (self.memory > 0) * (": %i" % self.memory)
        return name
        
    def reset(self):
        """
        Resets scores, QTable and history
        """
        Player.reset(self)

        self.qTab = [dict({True: 0, False: 0}) for i in range(0, self.qTabSize)]
        self.turns = [dict({True: 0, False: 0}) for i in range(0, self.qTabSize)]
        self.totTurns = 0
        self.prevState = 0
        self.prevAction = False

def main():
    ply = Learner(memory_depth = 1)
    print ply
    p1 = axelrod.Random()
    # for turn in range(10):
    #     p1.play(p2)
    # print p1.history, p2.history
    for turn in range(1000):
        ply.play(p1)
    print ply.qTab
    # print ply.history
    # print p1.history

main()