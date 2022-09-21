import numpy

class GridWorld:
    def __init__(self):
        self.exits = [0,15]
        self.position = 6

    def move(self, direction):
        if direction == 0:
                if self.position-4 >= 0:
                    self.position-=4
        elif direction == 1:
                if self.position % 4 != 0:
                    self.position+=1
        elif direction == 2:
                if self.position <= 15:
                    self.position+=4
        elif direction == 3:
                if self.position % 4 != 3:
                    self.position-=1

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