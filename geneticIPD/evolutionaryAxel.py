from axelrod import Actions, Player, init_args

C, D = Actions.C, Actions.D

class EvolutionaryAxel(Player):
    """A player uses a predefined array of choices. It considers the k previous
    moves and decides which move to make based on the provided list.

    An optional memory attribute (Default = 3) will limit the number of turns remembered.
    """

    classifier = {
        'stochastic': False,
        'inspects_source': False,
        'makes_use_of': set(),
        'manipulates_source': False,
        'manipulates_state': False,
        'memory_depth': 3  # memory_depth may be altered by __init__
    }

    @init_args
    def __init__(self, memory_depth = 3, action_codes = ([False]*70)):
        """
        Parameters
        ----------
        memory_depth, int >= 0
            The number of rounds to use for the calculation of the cooperation
            and defection probabilities of the opponent.
        action_codes, bool list
            Indicates the exact action to perform based on the other players' history.
        """

        Player.__init__(self)
        self.actions = action_codes
        self.classifier['memory_depth'] = memory_depth
        self.memory = self.classifier['memory_depth']

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
            if (choice == 'P'):
                val += 0
            elif (choice == 'T'):
                val += 1
            elif (choice == 'S'):
                val += 2
            elif (choice == 'R'):
                val += 3
            else:
                raw_input("ERROR DURING DECODE. WHAT TO DO NEXT?")
        return val

    def strategy(self, opponent):
        """A player uses a predefined array of choices. It considers the k previous
        moves and decides which move to make based on the provided list.

        The length of the action list is 4^mem + 2*mem; The 2*mem extra bits are due to
        assuming what were the mem moves played by each before the game started.
        
        The format of actions is,
        [------Valid Codes (4^mem)-------] + [----My pre-moves (mem)----] + [----Ops pre-moves (mem)----]
        """
        
        mem = self.memory
        opHistory = opponent.history[-mem:]
        myHistory = self.history[-mem:]
        
        if (len(opHistory) < mem):
            tmpList = []
            deficit = mem - len(opHistory)
            for i in range(deficit, 0, -1):
                tmpList.append(self.actions[-i])
            opHistory = tmpList.extend(opHistory)
        if (len(myHistory) < mem):
            tmpList = []
            deficit = mem - len(myHistory)
            for i in range(deficit, 0, -1):
                tmpList.append(self.actions[-i])
            myHistory = tmpList.extend(myHistory)

        choice = []
        for i in range(0,mem):
            choice.append(encode(myHistory[i], opHistory[i]))
        ids = decode(choice, mem)
        if (self.action[ids]):
            return C
        else:
            return D

    def __repr__(self):
        """The string method for the strategy."""
        name = 'Go By Majority' + (self.memory > 0) * (": %i" % self.memory)
        if self.soft:
            name = "Soft " + name
        else:
            name = "Hard " + name
        return name
