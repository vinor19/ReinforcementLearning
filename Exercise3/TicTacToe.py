import random


class TicTacToe:
    def __init__(self):
        self.board = [0,0,0,0,0,0,0,0,0]
        self.CurrentPlayer = 1
        self.piecesPlayed = 0

    def move(self, spot):
        if self.board[spot] != 0:
            return False
        self.board[spot] = self.CurrentPlayer
        self.piecesPlayed += 1
        self.CurrentPlayer = 1 if self.CurrentPlayer == 2 else 2
        return True
    
    def GameOver(self):
        if(self.piecesPlayed >=9):
            return(True, 0)
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
        return (False,0)
    


def start(game):
    gameover = False
    winner = 0
    while not gameover:
        print(game.board[0:3])
        print(game.board[3:6])
        print(game.board[6:9])
        piecePlayed = False
        while not piecePlayed:
            piecePlayed = game.move(int(input('Where do you place your piece:\n'))-1)
            print("Next move "+ str(game.CurrentPlayer))
        gameover, winner = game.GameOver()
    return winner
state = []
def start(game):
    global state
    gameover = False
    winner = 0
    while not gameover:
        print(game.board[0:3])
        print(game.board[3:6])
        print(game.board[6:9])
        piecePlayed = False
        while not piecePlayed:
            piecePlayed = game.move(random.randint(0,8))
            print("Next move "+ str(game.CurrentPlayer))
        gameover, winner = game.GameOver()
        state = game.board
    return winner

game = TicTacToe()
winner = start(game)
print(state[0:3])
print(state[3:6])
print(state[6:9])
print("Player " + str(winner) + " won the match!")



