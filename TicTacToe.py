from queue import Empty
import random
import numpy as np
import copy as cp
import json
import matplotlib
from matplotlib import pyplot as plt
from multiprocessing import Process
matplotlib.use('TkAgg')

class TicTacToe:
    def __init__(self):
        self.board = [0,0,0,0,0,0,0,0,0]
        self.CurrentPlayer = 1
        self.piecesPlayed = 0

    def move(self, spot, player = 0):
        if self.GameOver()[0]:
            return (False, 0)
        player = self.CurrentPlayer if player == 0 else player
        if self.board[spot] != 0:   
            self.CurrentPlayer = 1 if player == 2 else 2
            return False, -100
        self.board[spot] = player
        self.piecesPlayed += 1
        self.CurrentPlayer = 1 if player == 2 else 2
        return True, 0 if not self.GameOver()[0] else 10
    
    def GameOver(self):
        for i in range(3):
            if (self.board[i*3] != 0 and self.board[i*3] == self.board[i*3+1] and self.board[i*3+1] == self.board[i*3+2]):
                return (True, self.board[i*3])
        for i in range(3):
            if (self.board[i] != 0 and self.board[i] == self.board[i+3] and self.board[i+3] == self.board[i+6]):
                return (True, self.board[i])
        if (self.board[0] != 0 and self.board[0] == self.board[4] and self.board[4] == self.board[8]):
            return (True, self.board[0])
        if (self.board[2] != 0 and self.board[2] == self.board[4] and self.board[4] == self.board[6]):
            return (True, self.board[2])
        
        if(self.piecesPlayed >=9):
            return(True, 0)
        return (False,0)

game = TicTacToe()
# winner = startExpert(game)
# print(state[0:3])
# print(state[3:6])
# print(state[6:9])
# print("Player " + str(winner) + " won the match!")

def getAvailableMoves(game):
    possibilities = []
    for i in range(0,len(game.board)):
        if game.board[i] == 0:
            possibilities.append(i)
    return possibilities

def minimax(board, depth, alpha, beta, maximizing_player):
    children = getAvailableMoves(board)
    if depth == -1:
        return 0, evaluate(board)
    if depth == 0 or board.GameOver()[0]:
        return children[0], evaluate(board)

    best_move = children[0]

    if maximizing_player:
        max_eval = np.NINF        
        for child in children:
            board_copy = cp.deepcopy(board)
            board_copy.move(child)
            current_eval = minimax(board_copy, depth-1, alpha, beta, False)[1]
            if current_eval > max_eval:
                max_eval = current_eval
                best_move = child
            alpha = max(alpha, current_eval)
            if beta <= alpha:
                break
        return best_move, max_eval

    else:
        min_eval = np.Inf
        for child in children:
            board_copy = cp.deepcopy(board)
            board_copy.move(child)
            current_eval = minimax(board_copy, depth-1, alpha, beta, True)[1]
            if current_eval < min_eval:
                min_eval = current_eval
                best_move = child
            beta = min(beta, current_eval)
            if beta <= alpha:
                break
        return best_move, min_eval

def evaluate(board):
    if board.GameOver()[1] == 1:
        return -1
    if board.GameOver()[1] == 2:
        return 1
    return 0

def startExpert2(game):
    global state
    gameover = False
    winner = 0
    player = 1
    depth = 8
    while not gameover:
        print("---------------")
        print(game.board[0:3])
        print(game.board[3:6])
        print(game.board[6:9])
        if game.CurrentPlayer == player:
            piecePlayed = False
            piecePlayed, _ = game.move(int(input('Where do you place your piece:\n'))-1)
        else:
            print(getAvailableMoves(game))
            piece, _ = minimax(game, len(getAvailableMoves(game))-1, np.NINF, np.Inf, player == 1)
            print(piece)
            piecePlayed, _ = game.move(piece)
        depth -= 1
        gameover, winner = game.GameOver()
        state = game.board
    return winner

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
        gameover, winner = game.GameOver()
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
        gameover, winner = game.GameOver()
        state = game.board
    return winner

def startAIGamevsExpert(ai, game  = TicTacToe(), p = 1):
    game = game
    gameover = False
    winner = 0
    player = p
    while not gameover:
        # print("---------------")
        # print(game.board[0:3])
        # print(game.board[3:6])
        # print(game.board[6:9])
        if game.CurrentPlayer == player:
            game.move(minimax(game, len(getAvailableMoves(game))-1, np.NINF, np.Inf, True)[0])
        else:
            game.move(ai.chooseAction(game))
        gameover, winner = game.GameOver()
    return winner



def boardToState(board):
    output = ""
    for i in board:
        output += str(i)
    return output

# f = open("Tree.json", "r")
# aiTree = json.load(f).keys()

class TictactoeAI:
    def __init__(self, gamma = 0.9, theta = 0.01, game = TicTacToe(), lr = 0.2, epsilon = 0.3, N = 1):
        self.Q_ø = {}
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.theta = theta
        self.game = game
        self.N = N

    def chooseActionTraining(self, game):
        boardHash = boardToState(game.board)
        actions = getAvailableMoves(game)
        if actions != []:
            if np.random.uniform(0, 1) <= self.epsilon:
                action = random.choice(actions)
            else:
                stateActionValues = self.Q_ø.get(boardHash,np.zeros(9).tolist())
                if stateActionValues == np.zeros(9).tolist():
                    action = random.choice(actions)
                else: 
                    action = np.argmax(stateActionValues)
                
            return action
        else:
            return 0

    def chooseAction(self, game):
        boardHash = boardToState(game.board)
        stateActionValues = self.Q_ø.get(boardHash,np.zeros(9).tolist())
        actions = getAvailableMoves(game)
        if actions != []:
            if stateActionValues == np.zeros(9).tolist():
                action = actions[0]
            else: 
                action = np.argmax(stateActionValues)
            return action
        else:
            return 0

    def train(self,againstExpert = False, againstSelf = False, epochs = 100, episodes = 100):
        for i in range(epochs):
            if againstExpert:
                self.learnvsexpert(episodes, p=1)
                self.learnvsexpert(episodes, p=2)
            if againstSelf:
                self.learnvsself(episodes, p=1)
                self.learnvsself(episodes, p=2)

    def learnvsexpert(self, episodes, p = 1):
        # converged = False
        for i in range(episodes):
            # if i%100 == 0:
            #     print(str(i/episodes*100)+"%")
            game = cp.deepcopy(self.game)
            if (p == 2):
                game.move(minimax(game, len(getAvailableMoves(game))-1, np.NINF, np.Inf, p==2)[0])
            while not game.GameOver()[0]:
                action0 = None
                state0 = None
                actionOpponent = None
                giN = 0
                nextgamestate = cp.deepcopy(game)
                for i in range(0,self.N):
                    action = self.chooseActionTraining(nextgamestate)
                    reward1, _ = nextgamestate.move(action)
                    actionExpert, _ = minimax(nextgamestate, len(getAvailableMoves(game))-1, np.NINF, np.Inf, p==2)
                    reward2, _ = nextgamestate.move(actionExpert)
                    if i == 0:
                        action0 = action
                        actionOpponent = actionExpert
                        state0 = boardToState(game.board)
                        self.Q_ø[boardToState(state0)] = self.Q_ø.get(boardToState(game.board),np.zeros(9).tolist())
                    giN += self.gamma**i * (reward1-reward2)
                newValue = (
                    (1-self.lr) * self.Q_ø[boardToState(state0)][action0] + 
                    self.lr*((giN) + self.gamma**self.N * max(self.Q_ø.get(boardToState(nextgamestate.board),np.zeros(9).tolist())) - self.Q_ø[boardToState(state0)][action0])
                    )
                self.Q_ø[boardToState(state0)][action0] = newValue
                game.move(action0)
                game.move(actionOpponent)

    def learnvsself(self, episodes, p = 1):
        # converged = False
        for i in range(episodes):
            # if i%1000 == 0:
            #     print(str(i/episodes*100)+"%")
            game = cp.deepcopy(self.game)
            if (p == 2):
                game.move(self.chooseAction(game))
            while not game.GameOver()[0]:
                action0 = None
                state0 = None
                actionOpponent = None
                giN = 0
                nextgamestate = cp.deepcopy(game)
                for i in range(0,self.N):
                    action = self.chooseActionTraining(nextgamestate)
                    _, reward1 = nextgamestate.move(action)
                    actionOther = self.chooseAction(nextgamestate)
                    _, reward2 = nextgamestate.move(actionOther)
                    if nextgamestate.GameOver()[0] and nextgamestate.GameOver()[1]  == 0:
                        reward1 = 5
                        reward2 = 0
                    if i == 0:
                        action0 = action
                        actionOpponent = actionOther
                        state0 = boardToState(game.board)
                        self.Q_ø[boardToState(state0)] = self.Q_ø.get(boardToState(game.board),np.zeros(9).tolist())
                    giN += self.gamma**i * (reward1-reward2)
                    if nextgamestate.GameOver()[0]:
                        break
                newValue = (
                    (1-self.lr) * self.Q_ø[boardToState(state0)][action0] + 
                    self.lr*((giN) + self.gamma**(i) * max(self.Q_ø.get(boardToState(nextgamestate.board),np.zeros(9).tolist())))
                    )
                self.Q_ø[boardToState(state0)][action0] = newValue
                game.move(action0)
                game.move(actionOpponent)
        
""" MATHIAS EXPERT """
f = open("MathExpert.json","r")
mathExpert = json.load(f)
f.close()
def printBoard(board):
    print(board[0:3])
    print(board[3:6])
    print(board[6:9])

def startAIGamevsMATH(ai, game  = TicTacToe(), p = 1, doPrint = False):
    game = game
    gameover = False
    winner = 0
    player = p
    if doPrint:
        printBoard(game.board)
    while not gameover:
        if game.CurrentPlayer == player:
            move = mathMoves[mathExpert[toState(game.board)]]
            game.move(move)
            if doPrint:
                print("---------------")
                printBoard(game.board)
                print("Expert moved")
        else:
            move = ai.chooseAction(game)
            game.move(move)
            if doPrint:
                print("---------------")
                printBoard(game.board)
                print("AI moved")
        gameover, winner = game.GameOver()
        
    return winner

def startvsMATH(ai, game  = TicTacToe(), p = 2, doPrint = False):
    game = game
    gameover = False
    winner = 0
    player = p
    if doPrint:
        printBoard(game.board)
    while not gameover:
        if game.CurrentPlayer == player:
            move = mathMoves[mathExpert[toState(game.board)]]
            game.move(move)
            if doPrint:
                print("---------------")
                printBoard(game.board)
                print("Expert moved")
        else:
            game.move(int(input('Where do you place your piece:\n')))
            if doPrint:
                print("---------------")
                printBoard(game.board)
                print("AI moved")
        gameover, winner = game.GameOver()
        
    return winner

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




# f = open("AI.json","w")
# json.dump(ai.Q_ø, f)
# f.close()
# f = open("AI.json","r")
# f.close()
# print(len(ai.Q_ø.keys()))

# scoresExpert = np.zeros(rounds).tolist()
# draws = np.zeros(rounds).tolist()
def testTraining(ai, scores: list, rounds, tests):
    for j in range(tests):
        aiInTraining = cp.deepcopy(ai)
        for i in range(rounds):
            Wins = [0,0,0]
            # aivsexpert.train(againstExpert = True, epoch = 1, episodes=5)
            aiInTraining.train(againstSelf = True, epochs=5, episodes=5)
            nextgame = TicTacToe()
            winner = startAIGamevsMATH(aiInTraining, game=nextgame, p=2)
            if winner == 2:
                Wins[1] += 1
            elif winner == 1:
                Wins[0] += 1
            else:
                Wins[2] += 1
            nextgame = TicTacToe()
            winner = startAIGamevsMATH(aiInTraining, game=nextgame, p=1)
            if winner == 1:
                Wins[1] += 1
            elif winner == 2:
                Wins[0] +=1
            else:
                Wins[2] += 1
            # scoresExpert[i]+=Wins[0]/tests
            scores[i]+=(Wins[0]-Wins[1]+Wins[2]/2)/tests
            # draws[i]+=Wins[2]/tests
        print(j)


if __name__ == '__main__':
    rounds = 100
    tests = 100
    scoresN1 = np.zeros(rounds).tolist()
    scoresN2 = np.zeros(rounds).tolist()
    scoresN3 = np.zeros(rounds).tolist()
    aivsn1 = TictactoeAI(N=1, lr=0.05)
    aivsn2 = TictactoeAI(N=2, lr=0.05)
    aivsn3 = TictactoeAI(N=3, lr=0.05)
    # p1 = Process(target=testTraining, args=(aivsn1, scoresN1, rounds, tests))
    # p1.start()
    # p2 = Process(target=testTraining, args=(aivsn2, scoresN2, rounds, tests))
    # p2.start()
    # p3 = Process(target=testTraining, args=(aivsn3, scoresN3, rounds, tests))
    # p3.start()
    # p1.join()
    # p2.join()
    # p3.join()
    testTraining(aivsn1, scoresN1, rounds=rounds, tests=tests)
    testTraining(aivsn2, scoresN2, rounds=rounds, tests=tests)
    testTraining(aivsn3, scoresN3, rounds=rounds, tests=tests)
    _, ax = plt.subplots()
    # line1 = ax.plot(scoresExpert)
    episodes = np.array(range(1,1+len(scoresN1)))
    # coefs1 = np.polynomial.polynomial.polyfit(episodes,scoresN1,1)
    # coefs2 = np.polynomial.polynomial.polyfit(episodes,scoresN2,1)
    # coefs3 = np.polynomial.polynomial.polyfit(episodes,scoresN3,1)

    # line1 = ax.plot(np.polynomial.polynomial.polyval(episodes,coefs1))
    # line2 = ax.plot(np.polynomial.polynomial.polyval(episodes,coefs2))
    # line3 = ax.plot(np.polynomial.polynomial.polyval(episodes,coefs3))

    line1 = ax.plot(scoresN1)
    line2 = ax.plot(scoresN2)
    line3 = ax.plot(scoresN3)
    # line3 = ax.plot(draws)
    plt.xlabel("per 5 Epoch of 5 training games")
    plt.ylabel("Wins")
    plt.title("Training methods")
    ax.legend(["ScoreN1","ScoreN2","ScoreN3"])
    # plt.show()
    plt.savefig('foo1.png')

