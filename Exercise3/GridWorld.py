import numpy as np
from matplotlib import pyplot as plt
from numpy import random
from pprint import pprint

"""" Setting up GridWorld information and useful functions """

allowedActions = ['u','d','l','r']

class GridWorld:
    def __init__(self, size = 4):
        self.size = size
        self.gridSpace = size*size
        self.exits = [0,(size*size)-1]
        self.position = random.randint(1,(size*size)-2)

    def move(self, direction):
        if direction == 'u':
                if self.position-self.size  >= 0:
                    self.position-=self.size
        elif direction == 'd':
                if self.position+self.size  < self.gridSpace:
                    self.position+=self.size 
        elif direction == 'r':
                if (self.position+1) % self.size != 0:
                    self.position+=1
        elif direction == 'l':
                if (self.position-1) % self.size != self.size-1:
                    self.position-=1
        return self.position, -1

    def over(self):
        return self.position in self.exits

    def reset(self):
        self.position = random.randint(1,14)

def start(game, maxMoves = 50):
    game.reset()
    gameover = False
    totalReward = 0
    moves = 0
    while not gameover and moves < maxMoves:
        moves += 1
        print("Current position: " + str(game.position))
        _, moveReward = game.move(input('What direction do you want U, D, L or R:\n').lower())
        totalReward += moveReward
        gameover = game.over()
    return totalReward

def startValue(game, valueIterator, maxMoves = 50):
    game.reset()
    gameover = False
    totalReward = 0
    moves = 0
    while not gameover and moves < maxMoves:
        moves += 1
        _, moveReward = game.move(valueIterator.decideMove(game.position))
        totalReward += moveReward
        gameover = game.over()
    return totalReward

def startPolicy(game, policyIterator, maxMoves = 50):
    game.reset()
    gameover = False
    totalReward = 0
    moves = 0
    while not gameover and moves < 50:
        moves += 1
        _, moveReward = game.move(policyIterator.decideMove(game.position))
        totalReward += moveReward
        gameover = game.over()
    return totalReward


"""" Setting up ValueIterator information and useful functions """
valueRewards = []

class ValueIterator:
    def __init__(self, gamma, game = GridWorld()):
        self.policy = np.array([None for _ in range(game.gridSpace)])
        self.values = np.zeros(game.gridSpace)
        self.game = game
        self.states = range(1,game.gridSpace-1)
        self.statesPlus = range(0,game.gridSpace) # Unused
        self.actions = allowedActions
        self.gamma = gamma

        # Initialize policy 
        for i in self.states:
            self.policy[i] = self.actions.copy()

    # Trains until converges
    def train(self):
        valueRewards.append(self.currentReward())
        change = True
        while change:
            oldValues = self.values.copy()
            for state in self.states:
                self.decideMoveWithUpdate(state)
            valueRewards.append(self.currentReward())
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
        self.values[state] = round(maxReward,4)
        self.policy[state] = [
            action for action,
            reward in actionReward.items() if reward == maxReward
            ]

    def decideMove(self, state):
        return self.policy[state][random.randint(len(self.policy[state]))]

    def currentReward(self, iter = 1000):
        averageReward = 0
        for _ in range(iter):
            averageReward += startValue(self.game,self)/iter
        return averageReward
        

def to2DList(array):
    return np.reshape(array,(int(np.sqrt(len(array))),int(np.sqrt(len(array))))).tolist()


"""" Setting up PolicyIterator information and useful functions """
policyRewards = []
class PolicyIterator:
    def __init__(self, gamma, theta, game = GridWorld()):
        self.policy = np.array([None for _ in range(game.gridSpace)])
        self.values = np.zeros(game.gridSpace)
        self.game = game
        self.states = range(1,game.gridSpace-1)
        self.statesPlus = range(0,game.gridSpace) # Unused
        self.actions = allowedActions
        self.gamma = gamma
        self.theta = theta

        for i in self.states:
            self.policy[i] = self.actions.copy()

    def evaluatePolicy(self):
        converged = False
        while not converged:
            delta = 0
            for state in self.states:
                oldValue = self.values[state]
                total = 0
                weight = 1 / len(self.policy[state])
                for a in self.policy[state]:
                    self.game.position = state
                    state_, reward = self.game.move(a)
                    total += weight * (reward + self.gamma * self.values[state_])
                self.values[state] = np.round(total, 4)
                delta = max(delta, np.abs(oldValue - self.values[state]))
                converged = True if delta < self.theta else False
 
    def train(self):
        self.improvePolicy() 

    def improvePolicy(self):
        policyRewards.append(self.currentReward())
        stable = False
        while not stable:
            self.evaluatePolicy()
            stable = True
            for state in self.states:
                oldActions = self.policy[state]
                newActions = []
                actionRewards = {}
                maxReward = np.NINF
                for a in self.actions:
                    self.game.position = state
                    state_, reward = self.game.move(a)
                    reward += (reward + self.gamma * self.values[state_])
                    maxReward = max(maxReward, reward)
                    actionRewards[a] = reward

                self.policy[state] = [
                    action for action,
                    reward in actionRewards.items() if reward == maxReward
                    ]
                if oldActions != self.policy[state]:
                    stable = False
            policyRewards.append(self.currentReward())
                    
    def decideMove(self, state):
        return self.policy[state][random.randint(len(self.policy[state]))]
    
    def currentReward(self, iter = 1000):
        averageReward = 0
        for _ in range(iter):
            averageReward += startPolicy(self.game,self)/iter
        return averageReward

if __name__ == "__main__":

    # Arrange
    size = 13 # Length and width of the grid
    theta = 1e-2 # Error tolerance
    gamma = 0.9  # Discount
    game = GridWorld(size) # game of size "size"

    # Value iterator action
    print("\n_______Value Iterator______\n")
    test = ValueIterator(gamma, game=game)

    print("Reward before training: " + str(startValue(game, test)))

    test.train()
    print("Reward after training: " + str(startValue(game, test)))

    # Policy iterator action
    print("\n_______Policy Iterator______\n")
    pIterator = PolicyIterator(gamma, theta,game=game)

    print("Reward before training: " + str(startValue(game, pIterator)))

    pIterator.train()
    print("Reward after training: " + str(startValue(game, pIterator)))

    # Display result
    _, ax = plt.subplots()
    line1 = ax.plot([round(item,1) for item in valueRewards])
    line2 = ax.plot([round(item,3) for item in policyRewards])
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Value vs Policy iterators")
    ax.legend(["Value","Policy"])
    plt.show()