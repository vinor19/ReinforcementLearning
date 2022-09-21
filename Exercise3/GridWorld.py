from tkinter import Grid
import numpy


class GridWorld:
    def __init__(self):
        self.board = numpy.array(16)
        self.exits = [0,15]
        self.position = 6

    def move(self, direction):
        match direction:
            case 0:
                self.position-=4
                if self.position < 0:
                    self.position+=4
            case 1:
                self.position+=1
                if self.position % 4 == 0:
                    self.position-=1
            case 2:
                self.position+=4
                if self.position > 15:
                    self.position-=4
            case 3:
                self.position-=1
                if self.position % 4 == 3:
                    self.position+=1

    def over(self):
        return self.position in self.exits

def start(game):
    gameover = False
    reward = 0
    while not gameover:
        print(game.position)
        game.move(int(input('What direction do you want nesw:\n'))-1)
        reward-=1
        gameover = game.over()
    return reward

game = GridWorld()
print(start(game))