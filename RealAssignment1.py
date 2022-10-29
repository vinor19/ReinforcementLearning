import random
import numpy as np
import json
import matplotlib
from matplotlib import pyplot as plt
from multiprocessing import Pool, cpu_count
import time

# Game interface
class BoardGame:
    def __init__(self):
        self.board = []

    def copy(self):
        return self

    def reset(self):
        pass

    def getAvailableMoves(self):
        return []

    def move(self,spot):
        return (False,spot)

    def gameOver(self):
        return True

    def getCost(self):
        return 0

    def toState(self):
        return ""

# Player interface
class Player:
    def getMove(self,game: BoardGame):
        return 0

# TicTacToe Implementation
class TicTacToe(BoardGame):
    def __init__(self):
        self.board = [0]*9
        self.CurrentPlayer = 1
        self.piecesPlayed = 0
        self.winner = 0
        self.over = False

    def copy(self):
        newgame = TicTacToe()
        newgame.board = self.board.copy()
        newgame.CurrentPlayer = self.CurrentPlayer
        newgame.piecesPlayed = self.piecesPlayed
        newgame.winner = self.winner
        newgame.over = self.over
        return newgame
    
    def reset(self):
        self.board = [0]*9
        self.CurrentPlayer = 1
        self.piecesPlayed = 0
        self.winner = 0
        self.over = False

    def move(self, spot):
        if self.over:
            return (False, 0)
        if self.board[spot] != 0:   
            self.CurrentPlayer = 1 if self.CurrentPlayer == 2 else 2
            return False, -100
        self.board[spot] = self.CurrentPlayer
        self.piecesPlayed += 1
        self.CurrentPlayer = 1 if self.CurrentPlayer == 2 else 2
        if self.piecesPlayed>4:
            self.gameOver() 
        return True, self.getCost(self.CurrentPlayer)

    def gameOver(self):
        if not self.over:
            for i in range(3):
                if (self.board[i*3] != 0 and self.board[i*3] == self.board[i*3+1] and self.board[i*3+1] == self.board[i*3+2]):
                    self.over = True
                    self.winner = self.board[i*3]
                    return (True, self.winner)
            for i in range(3):
                if (self.board[i] != 0 and self.board[i] == self.board[i+3] and self.board[i+3] == self.board[i+6]):
                    self.over = True
                    self.winner = self.board[i]
                    return (True, self.winner)
            if (self.board[0] != 0 and self.board[0] == self.board[4] and self.board[4] == self.board[8]):
                self.over = True
                self.winner = self.board[0]
                return (True, self.board[0])
            if (self.board[2] != 0 and self.board[2] == self.board[4] and self.board[4] == self.board[6]):
                self.over = True
                self.winner = self.board[2]
                return (True, self.board[2])
            
            if(self.piecesPlayed >=9):
                self.over = True
                return(True, 0)
            return (False,0)
        return (True,self.winner)

    def playGame(self, p1: Player, p2: Player):
        while not self.over:
            if self.CurrentPlayer == 1:
                move = p1.getMove(self)
                self.move(move)
            else:
                move = p2.getMove(self)
                self.move(move)
        return self.winner, self.piecesPlayed

    def getCost(self,p):
        if self.over:
            if self.winner == 0:
                return -50
            if self.winner == p:
                return -100
            if self.winner != p:
                return 100-self.piecesPlayed*3
        else:
            return 1

    def getAvailableMoves(self):
        return [idx for idx, val in enumerate(self.board) if val == 0]

    def toState(self):
        return str(self.board)


""" MATHIAS EXPERT """
class MathiasExpert(Player):
    def __init__(self):
        f = open("MathExpert.json","r")
        self.hashMoves: dict = json.load(f) # The expert has all the states stored in a json file I load in to use as a dictionary
        f.close()
        self.mathMoves = {"NW": 0, "N": 1, "NE": 2, "W":3, "C": 4, "E": 5, "SW": 6, "S": 7, "SE": 8}

    # Converts my boards to states used in the expert dictionary
    def toState(self, board: list) -> int:
            """Get the board state

            Returns:
                int: A unique integer identifying the current state

            Note that the state is calculated differently based on whose turn it currently is.
            Essentially, the state is represented as a number in base 3.
            The current player is represented as a "1" in the number, and the opposite player is "2".
            Empty is "0".

            """
            empty_spots = board.count(0)
            val = 0
            for x in board:
                val = val * 3 + (0
                                if x == 0
                                else (x + empty_spots) % 2 + 1)
            return str(int(val))
    
    # Needed for Player in
    def getMove(self, game: TicTacToe):
        return self.mathMoves[self.hashMoves.get(self.toState(game.board),"NW")]

class TictactoeAI(Player):
    def __init__(self, gamma = 0.9, game:TicTacToe = TicTacToe(), epsilon = 0.1, N = 1):
        self.Q_ø = {}
        self.epsilon = epsilon
        self.gamma = gamma
        self.game = game
        self.N = N

    def reset(self):
        self.Q_ø = {}

    def getMove(self,game:BoardGame):
        actions = game.getAvailableMoves()
        if actions != []:
            stateActionValues = [self.Q_ø.setdefault(game.toState(),[0]*9)[move] for move in actions]
            action = actions[np.argmin(stateActionValues)]
            return action
        else:
            return 0

    def getTrainingMove(self,game:BoardGame):
        actions = game.getAvailableMoves()
        if actions != []:
            if random.uniform(0, 1) <= self.epsilon:
                action = random.choice(actions)
            else:
                stateActionValues = [self.Q_ø.setdefault(game.toState,[0]*9)[move] for move in actions]
                action = actions[np.argmin(stateActionValues)]
            return action
        else:
            return 0

    def train(self, epochs = 100, episodes = 5, opponent: Player = None):
        if not opponent:
            opponent = self
        costSum = [0]*epochs
        for i in range(epochs):
            costSum[i] += self.learn(opponent, episodes)
        return costSum

    def learn(self, opponent: Player, episodes):
        costSum = 0
        for i in range(episodes):
            for p in [1,2]:
                self.game.reset()
                N = random.randint(1,self.N)
                if (p == 2):
                    self.game.move(opponent.getMove(self.game))
                while not self.game.over:
                    action0 = None
                    state0 = self.game.toState()
                    giN = 0
                    for i in range(0,N):
                        action = self.getTrainingMove(self.game)
                        self.game.move(action)
                        self.game.move(opponent.getMove(self.game))

                        #Save action0
                        if i == 0:
                            action0 = action
                            self.Q_ø.setdefault(state0,[0]*9)

                        #Summing cost
                        giN += self.game.getCost(p)
                        
                        if self.game.over:
                            # self.Q_ø.setdefault(self.game.toState(),[self.game.getCost(p)]*9)
                            break #Reached a terminal state
                    N=self.N
                    newValue = (
                        (1-self.gamma) * self.Q_ø[state0][action0] + 
                        (self.gamma) * ((giN) + min(self.Q_ø.setdefault(self.game.toState(),[0]*9)))
                        )
                    self.Q_ø[state0][action0] = newValue
                self.game.reset()
                if p == 1:
                    winner, _ = self.game.playGame(self, MathiasExpert())
                else:
                    winner, _ = self.game.playGame(MathiasExpert(), self)
                if winner == p:
                    costSum -=2
                elif winner == 0:
                    costSum += 0
                else:
                    costSum += 2
        return costSum

# Training the AI and returning a list containing the scores for each test
def testTrainingRewardsReturned(ai:TictactoeAI, rounds, tests):
    scores = []
    for j in range(tests):
        ai.reset()
        scores.append(ai.train(opponent = MathiasExpert(), epochs=rounds))
        if j%(tests/10) == tests/10-1:
            print(ai.epsilon, ai.N, str((j+1)/tests*100)+"%")
    print(ai.epsilon, ai.N, "Done")
    return (scores, ai.epsilon, ai.N)

colors = ['cyan','aquamarine','blue','yellow','orange','red','lime','green','darkgreen']

# Main that implements multiprocessing to run the tests faster, seeding does not work for this
def multiprocessingMain(aiList, rounds, tests):
    with Pool(cpu_count()-1) as p:
        start_time = time.time()
        result = p.starmap(
            testTrainingRewardsReturned, 
            [( aiList[i], rounds, tests) 
                for i in range(len(aiList))
            ])
        _, ax = plt.subplots()
        labels = []
        for i in range(len(result)):
            mean = np.array(result[i][0]).mean(axis=0)
            std = np.array(result[i][0]).std(axis=0)/np.sqrt(tests)
            ax.plot(range(0,rounds),mean, color=colors[i])
            ax.fill_between(range(0,rounds),mean+std, mean-std, facecolor=colors[i], alpha=0.3)
            labels.append("N"+str(result[i][2])+"E"+str((result[i][1])))

        plt.xlabel("Epochs trained")
        plt.ylabel("Costs")
        plt.title("Training methods")
        ax.legend(labels)
        print("--- %s seconds ---" % (time.time() - start_time))
        plt.savefig('learning-curve.png')
        plt.show()

if __name__ == '__main__':
    # random.seed(42)
    rounds = 1500
    tests = 20
    eList = [0.1,0.2,0.3]
    NList = [1,2,3]
    aiList = [
                TictactoeAI(epsilon = e, N=N, gamma=0.95)
                for e in eList
                for N in NList
            ]
    multiprocessingMain(aiList, rounds, tests)