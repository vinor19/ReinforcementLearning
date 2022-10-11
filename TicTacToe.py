import random
import numpy as np
import json
import matplotlib
from matplotlib import pyplot as plt
from multiprocessing import Pool, cpu_count
import time
import pickle

matplotlib.use("TkAgg")

class TicTacToe:
    def __init__(self):
        self.board = [0,0,0,0,0,0,0,0,0]
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
        self.board = [0,0,0,0,0,0,0,0,0]
        self.CurrentPlayer = 1
        self.piecesPlayed = 0
        self.winner = 0
        self.over = False

    def move(self, spot, player = 0):
        if self.over:
            return (False, 0)
        player = self.CurrentPlayer if player == 0 else player
        if self.board[spot] != 0:   
            self.CurrentPlayer = 1 if player == 2 else 2
            return False, -100
        self.board[spot] = player
        self.piecesPlayed += 1
        self.CurrentPlayer = 1 if player == 2 else 2
        if self.piecesPlayed>4:
            self.GameOver()
        return True, 0

    def BoardOver(self,board):
        for i in range(3):
            if (board[i*3] != 0 and board[i*3] == board[i*3+1] and board[i*3+1] == board[i*3+2]):
                winner = board[i*3]
                return (True, winner)
        for i in range(3):
            if (board[i] != 0 and board[i] == board[i+3] and board[i+3] == board[i+6]):
                winner = board[i]
                return (True, winner)
        if (board[0] != 0 and board[0] == board[4] and board[4] == board[8]):
            winner = board[0]
            return (True, board[0])
        if (board[2] != 0 and board[2] == board[4] and board[4] == board[6]):
            winner = board[2]
            return (True, board[2])
        
        if(board.count(0) == 0):
            return(True, 0)
        return (False,0)

    def GameOver(self):
        if not self.over:
            self.over, self.winner = self.BoardOver(self.board)
            return self.over, self.winner
        return (True,self.winner)

    def GetCost(self,p):
        if self.over:
            if self.winner == 0:
                return -0.1
            if self.winner == p:
                return -1
            if self.winner != p:
                return 1
        else:
            return 0
            
def getAvailableMoves(game):
    return [idx for idx, val in enumerate(game.board) if val == 0]

def boardToState(board):
    return str(board)


def startAIGame(ai, game  = TicTacToe(), p = 2):
    global state
    gameover = False
    winner = 0
    player = p
    while not gameover:
        print("---------------")
        print(game.board[0:3])
        print(game.board[3:6])
        print(game.board[6:9])
        if game.CurrentPlayer == player:
            game.move(int(input('Where do you place your piece:\n'))-1)
        else:
            # print(getAvailableMoves(game))
            piece = ai.chooseAction(game)
            print(piece)
            game.move(piece)
        gameover, winner = game.over, game.winner
        state = game.board
    return winner

def startAIvsAIGame(ai, ai2, game  = TicTacToe(), printStates = False):
    global state
    game = game
    gameover = False
    winner = 0
    player = 1
    while not gameover:
        if printStates:
            print("---------------")
            print(game.board[0:3])
            print(game.board[3:6])
            print(game.board[6:9])
        if game.CurrentPlayer == player:
            game.move(ai.chooseAction(game))
        else:
            game.move(ai2.chooseAction(game))
        gameover, winner = game.over, game.winner
        state = game.board
    return winner


class TictactoeAI:
    def __init__(self, gamma = 0.9, theta = 0.01, game = TicTacToe(), lr = 0.2, epsilon = 0.3, N = 1):
        self.Q_ø = {}
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.theta = theta
        self.game = game
        self.N = N

    def reset(self):
        self.Q_ø = {}

    def copy(self):
        newai = TictactoeAI(self.gamma, self.theta, self.game.copy(), self.lr, self.epsilon, self.N)
        newai.Q_ø = self.Q_ø.copy()
        return newai

    def chooseActionTraining(self, game):
        actions = getAvailableMoves(game)
        if actions != []:
            if random.uniform(0, 1) <= self.epsilon:
                action = random.choice(actions)
            else:
                stateActionValues = [self.Q_ø.setdefault(boardToState(game.board),np.zeros(9))[move] for move in actions]
                action = actions[np.argmin(stateActionValues)]
            return action
        else:
            return 0

    def chooseAction(self, game):
        actions = getAvailableMoves(game)
        if actions != []:
            stateActionValues = [self.Q_ø.setdefault(boardToState(game.board),np.zeros(9))[move] for move in actions]
            action = actions[np.argmin(stateActionValues)]
            return action
        else: # Actions empty
            return 0

    def train(self,againstExpert = False, againstSelf = False, epochs:int = 100, episodes = 100):
        sumReward = np.zeros(epochs)
        for i in range(epochs):
            reward = 0
            if againstExpert:
                reward += self.learnvsexpert(episodes, p=1)
                reward += self.learnvsexpert(episodes, p=2)
            if againstSelf:
                reward += self.learnvsself(episodes, p=1)
                reward += self.learnvsself(episodes, p=2)
            sumReward[i]+=reward
        return sumReward
        

    def learnvsexpert(self, episodes, p):
        rewardSum = 0
        for i in range(episodes):
            self.game.reset()
            if (p == 2):
                self.game.move(mathMoves[mathExpert[toState(self.game.board)]])
            while not self.game.over:
                action0 = None
                state0 = None
                giN = 0
                nextgamestate = self.game.copy()
                for i in range(self.N):
                    action= self.chooseActionTraining(nextgamestate)
                    _, cost = nextgamestate.move(action)
                    nextgamestate.move(mathMoves[mathExpert.get(toState(nextgamestate.board),"NE")])

                    #Save state0 and action0
                    if i == 0:
                        action0 = action
                        state0 = boardToState(self.game.board)
                        
                    giN += np.power(self.gamma,i) * cost
                    if nextgamestate.over:
                        self.Q_ø.setdefault(boardToState(nextgamestate.board),np.full(9,nextgamestate.GetCost(p)))
                        break #Reached a terminal state
                
                newValue = (
                    (1-self.lr) * self.Q_ø.setdefault(state0,np.zeros(9))[action0] + 
                    self.lr*((giN) + np.power(self.gamma,(i+1)) * min(self.Q_ø.setdefault(boardToState(nextgamestate.board),np.zeros(9))))
                    )
                self.Q_ø[state0][action0] = newValue
                self.game.move(self.chooseActionTraining(self.game))
                self.game.move(mathMoves[mathExpert.get(toState(self.game.board),"NE")])
            winner, _ = startAIGamevsMATH(self, game  = TicTacToe(), p = p, doPrint = False)
            if winner == p:
                rewardSum -=2
            elif winner == 0:
                rewardSum += 0
            else:
                rewardSum += 2
        return rewardSum

    def learnvsself(self, episodes, p):
        rewardSum = 0
        for i in range(episodes):
            self.game.reset()
            if (p == 2):
                self.game.move(self.chooseAction(self.game))
            while not self.game.over:
                action0 = None
                state0 = None
                giN = 0
                nextgamestate = self.game.copy()
                for i in range(0,self.N):
                    action = self.chooseActionTraining(nextgamestate)
                    _, cost = nextgamestate.move(action)
                    nextgamestate.move(self.chooseAction(self.game))

                    #Save state0 and action0
                    if i == 0:
                        action0 = action
                        state0 = boardToState(self.game.board)
                    
                    giN += np.power(self.gamma,i) * cost

                    if nextgamestate.over:
                        self.Q_ø.setdefault(boardToState(nextgamestate.board),np.full(9,nextgamestate.GetCost(p)))
                        break #Reached a terminal state
                
                newValue = (
                    (1-self.lr) * self.Q_ø.setdefault(state0,np.zeros(9))[action0] + 
                    self.lr*((giN) + np.power(self.gamma,(i+1)) * min(self.Q_ø.setdefault(boardToState(nextgamestate.board),np.zeros(9))))
                    )
                self.Q_ø[state0][action0] = newValue
                self.game.move(self.chooseActionTraining(self.game))
                self.game.move(self.chooseAction(self.game))
            winner, _ = startAIGamevsMATH(self, game  = TicTacToe(), p = p, doPrint = False)
            if winner == p:
                rewardSum -=2
            elif winner == 0:
                rewardSum += 0
            else:
                rewardSum += 2
        return rewardSum        
""" MATHIAS EXPERT """
f = open("MathExpert.json","r")
mathExpert = json.load(f)
f.close()
def printBoard(board):
    print(board[0:3])
    print(board[3:6])
    print(board[6:9])

def startAIGamevsMATH(ai: TictactoeAI, game  = TicTacToe(), p = 1, doPrint = False):
    game = game
    gameover = False
    winner = 0
    if doPrint:
        printBoard(game.board)
    while not gameover:
        if game.CurrentPlayer == p:
            move = ai.chooseAction(game)
            game.move(move)
            if doPrint:
                print("---------------")
                printBoard(game.board)
                print("AI moved")
        else:
            move = mathMoves[mathExpert[toState(game.board)]]
            game.move(move)
            if doPrint:
                print("---------------")
                printBoard(game.board)
                print("Expert moved")
        gameover, winner = game.over, game.winner
        
    return winner, game.piecesPlayed

def toState(board: list) -> int:
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

mathMoves = {"NW": 0, "N": 1, "NE": 2, "W":3, "C": 4, "E": 5, "SW": 6, "S": 7, "SE": 8}

""" TRAINING SCRIPTS """

def testTrainingRewardsReturned(ai:TictactoeAI, rounds, tests):
    scores = []
    for j in range(tests):
        ai.reset()
        scores.append(ai.train(againstSelf = True, epochs=rounds, episodes=5))
        if j%(tests/10) == tests/10-1:
            print(ai.epsilon, ai.N, str((j+1)/tests*100)+"%")
    print(ai.epsilon, ai.N, "Done")
    return (scores, ai.epsilon, ai.N)

def testTrainingRewardsReturnedExpert(ai, rounds, tests):
    scores = []
    for j in range(tests):
        ai.reset()
        scores.append(ai.train(againstExpert = True,againstSelf = False, epochs=rounds, episodes=5))
        if j%(tests/10) == tests/10-1:
            print(ai.epsilon, ai.N, str((j+1)/tests*100)+"%")
    print(ai.epsilon, ai.N, "Done")
    return (scores, ai.epsilon, ai.N)

colors = [(0,0,1),(0,0,0.5),(0,0,0.3),(1,0,0),(0.5,0,0),(0.3,0,0),(0,1,0),(0,0.5,0),(0,0.3,0)]

def multiprocessingMain(aiList, rounds, tests):
    with Pool(cpu_count()-1) as p:
        start_time = time.time()
        result = p.starmap(
            testTrainingRewardsReturnedExpert, 
            [( aiList[i], rounds,tests) 
                for i in range(len(aiList))
            ])
        _, ax = plt.subplots()
        labels = []
        for i in range(len(result)):
            mean = np.array(result[i][0]).mean(axis=0)
            ste = np.array(result[i][0]).std(axis=0)/np.sqrt(tests)
            ax.plot(range(0,rounds),mean, color=colors[i])
            ax.fill_between(range(0,rounds),mean+ste, mean-ste, facecolor=colors[i], alpha=0.1)
            labels.append("N"+str(result[i][2])+"E"+str((result[i][1])))
            print("N"+str(result[i][2])+"E"+str((result[i][1])), colors[i])

        with open("test1", "wb") as fp:
            pickle.dump(result, fp)

        plt.xlabel("Epochs trained")
        plt.ylabel("Costs")
        plt.title("Training methods")
        ax.legend(labels)
        print("--- %s seconds ---" % (time.time() - start_time))
        plt.savefig('learning-curve.png')
        plt.show()

def Main(aiList, rounds, tests):
    start_time = time.time()
    random.seed(42)
    result = []
    for i in range(len(aiList)):
        result.append(testTrainingRewardsReturnedExpert( aiList[i], rounds,tests))
    _, ax = plt.subplots()
    labels = []
    for i in range(len(result)):
        mean = np.array(result[i][0]).mean(axis=0)
        std = np.array(result[i][0]).std(axis=0)/np.sqrt(tests)
        ax.plot(range(0,rounds),mean, color=colors[i])
        ax.fill_between(range(0,rounds),mean+std, mean-std, facecolor=colors[i], alpha=0.1)
        labels.append("N"+str(result[i][2])+"E"+str((result[i][1])))

    with open("test1", "wb") as fp:
        pickle.dump(result, fp)

    plt.xlabel("Epochs trained")
    plt.ylabel("Costs")
    plt.title("Training methods")
    ax.legend(labels)
    print("--- %s seconds ---" % (time.time() - start_time))
    plt.savefig('learning-curve.png')
    plt.show()

if __name__ == '__main__':
    rounds = 200
    tests = 50
    lr = 0.01
    gamma = 0.85
    epsilonValues = [0.1,0.2,0.3]
    NValues = [2,3,4]
    aiList = [TictactoeAI(epsilon = E, N=N, gamma=gamma, lr=lr) for E in epsilonValues for N in NValues]
    multiprocessingMain(aiList, rounds, tests)
    