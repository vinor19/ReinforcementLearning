import random
import numpy as np
import copy as cp
import json

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
    if depth == 0 or board.GameOver()[0]:
        return 0, evaluate(board)

    children = getAvailableMoves(board)
    best_move = children[0]

    if maximizing_player:
        max_eval = np.NINF        
        for child in children:
            board_copy = cp.deepcopy(board)
            board_copy.move(child)
            current_eval = minimax(board_copy, depth - 1, alpha, beta, False)[1]
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
            current_eval = minimax(board_copy, depth - 1, alpha, beta, True)[1]
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
            piece, _ = minimax(game, depth, np.NINF, np.Inf, player == 1)
            print(piece)
            piecePlayed, _ = game.move(piece)
        depth -= 1
        gameover, winner = game.GameOver()
        state = game.board
    return winner

def startAIGame(ai, game  = TicTacToe()):
    global state
    gameover = False
    winner = 0
    player = 2
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
            # print(getAvailableMoves(game))
            piece = ai.chooseAction(game)
            print(piece)
            piecePlayed, _ = game.move(piece)
        depth -= 1
        gameover, winner = game.GameOver()
        state = game.board
    return winner

def startAIGamevsExpert(ai, game  = TicTacToe()):
    global state
    gameover = False
    winner = 0
    player = 2
    depth = 7
    while not gameover:
        print("---------------")
        print(game.board[0:3])
        print(game.board[3:6])
        print(game.board[6:9])
        if game.CurrentPlayer == player:
            piece, _ = minimax(game, depth, np.NINF, np.Inf, player == 1)
            print(piece)
            piecePlayed, _ = game.move(piece)
        else:
            # print(getAvailableMoves(game))
            piece = ai.chooseAction(game)
            print(piece)
            piecePlayed, _ = game.move(piece)
        depth -= 1
        gameover, winner = game.GameOver()
        state = game.board
    return winner

def boardToState(board):
    output = ""
    for i in board:
        output += str(i)
    return output
f = open("Tree.json", "r")
aiTree = json.load(f).keys()

class TictactoeAI:
    def __init__(self, gamma = 0.9, theta = 0.01, game = TicTacToe(), lr = 0.2, epsilon = 0.3, N = 1):
        self.Q_ø = {}
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.theta = theta
        self.game = game
        self.N = N
        #Initialize Q_ø

    def chooseActionTraining(self):
        boardHash = boardToState(self.game.board)
        if np.random.uniform(0, 1) <= self.epsilon:
            action = random.randint(0,8)
        else:
            stateActionValues = self.Q_ø.get(boardHash,np.zeros(9).tolist())
            action = np.argmax(stateActionValues)
            return action

    def chooseAction(self, game):
        boardHash = boardToState(game.board)
        value_max = np.NINF
        
        stateActionValues = self.Q_ø.get(boardHash,np.zeros(9).tolist())
        print(stateActionValues)
        if stateActionValues == np.zeros(9).tolist():
            return random.randint(0,8)
        action = np.argmax(stateActionValues)
        return action

    def learn(self, episodes):
        # converged = False
        for i in range(episodes):
            if i%100 == 0:
                print(str(i/100)+"%")
            game = cp.deepcopy(self.game)
            while not game.GameOver()[0]:
                action0 = None
                state0 = None
                giN = 0
                nextgamestate = cp.deepcopy(game)
                for i in range(0,self.N):
                    action = self.chooseActionTraining()
                    if i == 0:
                        action0 = action
                        state0 = boardToState(game.board)
                        self.Q_ø[boardToState(state0)] = self.Q_ø.get(boardToState(game.board),np.zeros(9).tolist())
                    reward1, _ = nextgamestate.move(action)
                    actionExpert, _ = minimax(nextgamestate, 9-nextgamestate.piecesPlayed, np.NINF, np.Inf, game.CurrentPlayer == 1)
                    reward2, _ = nextgamestate.move(actionExpert)
                    giN += self.gamma**i * (reward1-reward2)
                #Initialize new state if needed
                newValue = (
                    (1-self.lr) * self.Q_ø[boardToState(state0)][action0] + 
                    self.lr*((giN) + self.gamma**self.N * max(self.Q_ø.get(boardToState(nextgamestate.board),np.zeros(9).tolist())) - self.Q_ø[boardToState(state0)][action0])
                    )
                self.Q_ø[boardToState(state0)][action0] = newValue
                game = nextgamestate

                
ai = TictactoeAI(N=3)

# ai.learn(10000)
# f = open("AI.json","w")
# json.dump(ai.Q_ø, f)
# f.close()
f = open("AI.json","r")
ai.Q_ø = json.load(f)
f.close()

print(startAIGamevsExpert(ai))

