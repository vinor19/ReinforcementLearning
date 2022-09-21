import numpy as np
from numpy import random
from pprint import pprint

class GridWorld:
    def __init__(self):
        self.exits = [0,15]
        self.position = random.randint(1,14)

    def move(self, direction):
        if direction == 0:
                if self.position-4 >= 0:
                    self.position-=4
        elif direction == 1:
                if (self.position+1) % 4 != 0:
                    self.position+=1
        elif direction == 2:
                if self.position+4 <= 15:
                    self.position+=4
        elif direction == 3:
                if (self.position-1) % 4 != 3:
                    self.position-=1
        return self.position, -1

    def over(self):
        return self.position in self.exits

    def reset(self):
        self.position = random.randint(1,14)

def start(game):
    gameover = False
    reward = 0
    while not gameover:
        print("Current position: " + str(game.position))
        _, moveReward = game.move(int(input('What direction do you want nesw:\n')))
        reward += moveReward
        gameover = game.over()
    return reward

def startValue(game, valueIterator):
    gameover = False
    reward = 0
    while not gameover:
        print(game.position)
        _, moveReward = game.move(valueIterator.decideMove(game.position))
        reward += moveReward
        gameover = game.over()
    return reward


class ValueIterator:
    def __init__(self, gamma):
        self.policy = np.array([None for _ in range(16)])
        self.values = np.zeros(16)
        self.game = GridWorld()
        self.states = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
        self.actions = [0,1,2,3]
        self.gamma = gamma

        # Initialize policy 
        for i in self.states:
            self.policy[i] = [0,1,2,3]

    # Trains until converges
    def train(self):
        change = True
        while change:
            oldValues = self.values.copy()
            for state in self.states:
                self.decideMoveWithUpdate(state)
            change = not (oldValues == self.values).all()

    
    def decideMoveWithUpdate(self, state):
        # Set up rewards
        maxReward = np.NINF
        actionReward = {}

        for a in self.actions:
            # Simulating the game for move a
            self.game.position = state
            newPos, reward = self.game.move(a)
            reward += self.gamma * self.values[newPos]
            maxReward = max(maxReward, reward)
            actionReward[a] = reward
        
        # Update the values and policies
        self.values[state] = maxReward
        self.policy[state] = [
            action for action,
            reward in actionReward.items() if reward == maxReward
            ]

        return self.decideMove(state)

    def decideMove(self, state):
        return self.policy[state][random.randint(len(self.policy[state]))]

def to2DList(array):
    return [
            array[0:4].tolist(),
            array[4:8].tolist(),
            array[8:12].tolist(),
            array[12:16].tolist()
            ]

if __name__ == "__main__":
    game = GridWorld()

    # Initializing ValueIterator
    test = ValueIterator(0.9)

    # Before training
    print("Reward before training: " + str(startValue(game, test)))

    # After training
    test.train()

    # Resetting makes the starting position random
    game.reset()
    print("Reward after training: " + str(startValue(game, test)))

    pprint(to2DList(test.values))
    pprint(to2DList(test.policy))