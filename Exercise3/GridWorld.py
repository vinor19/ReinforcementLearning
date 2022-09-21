import numpy as np
from numpy import random
from pprint import pprint

"""" Setting up GridWorld information and useful functions """

allowedActions = ['u','d','l','r']

class GridWorld:
    def __init__(self):
        self.exits = [0,15]
        self.position = random.randint(1,14)

    def move(self, direction):
        if direction == 'u':
                if self.position-4 >= 0:
                    self.position-=4
        elif direction == 'd':
                if self.position+4 <= 15:
                    self.position+=4
        elif direction == 'r':
                if (self.position+1) % 4 != 0:
                    self.position+=1
        elif direction == 'l':
                if (self.position-1) % 4 != 3:
                    self.position-=1
        return self.position, -1

    def over(self):
        return self.position in self.exits

    def reset(self):
        self.position = random.randint(1,14)

def start(game):
    game.reset()
    gameover = False
    reward = 0
    while not gameover:
        print("Current position: " + str(game.position))
        _, moveReward = game.move(input('What direction do you want U, D, L or R:\n').lower())
        reward += moveReward
        gameover = game.over()
    return reward

def startValue(game, valueIterator):
    game.reset()
    gameover = False
    reward = 0
    while not gameover:
        _, moveReward = game.move(valueIterator.decideMove(game.position))
        reward += moveReward
        gameover = game.over()
    return reward

def startPolicy(game, policyIterator):
    game.reset()
    gameover = False
    reward = 0
    while not gameover:
        _, moveReward = game.move(policyIterator.decideMove(game.position))
        reward += moveReward
        gameover = game.over()
    return reward


"""" Setting up ValueIterator information and useful functions """

class ValueIterator:
    def __init__(self, gamma):
        self.policy = np.array([None for _ in range(16)])
        self.values = np.zeros(16)
        self.game = GridWorld()
        self.states = range(1,15)
        self.statesPlus = range(0,16) # Unused
        self.actions = allowedActions
        self.gamma = gamma

        # Initialize policy 
        for i in self.states:
            self.policy[i] = self.actions.copy()

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


"""" Setting up PolicyIterator information and useful functions """

class PolicyIterator:
    def __init__(self, gamma, theta):
        self.policy = np.array([None for _ in range(16)])
        self.values = np.zeros(16)
        self.game = GridWorld()
        self.states = range(1,15)
        self.statesPlus = range(0,16) # Unused
        self.actions = allowedActions
        self.gamma = gamma
        self.theta = theta
        self.rewards = {}

        self.initP()

    def initP(self):
        # Initialize policy 
        for i in self.states:
            self.policy[i] = self.actions.copy()
        for state in self.states:
            for a in self.actions:
                self.game.position = state
                state_, reward = self.game.move(a)
                self.rewards[(state_, reward, state, a)] = 1

    def evaluatePolicy(self):
        converged = False
        while not converged:
            delta = 0
            for state in self.states:
                oldValue = self.values[state]
                total = 0
                weight = 1 / len(self.policy[state])
                for a in self.policy[state]:
                    for key in self.rewards:
                        (newState, reward, oldState, action) = key
                        if oldState == state and action == a:
                            total += weight * self.rewards[key] * (reward + self.gamma * self.values[newState])
                self.values[state] = np.round(total, 4)
                delta = max(delta, np.abs(oldValue - self.values[state]))
                converged = True if delta < self.theta else False
 
    def improvePolicy(self):
        stable = False
        while not stable:
            self.evaluatePolicy()
            stable = True
            for state in self.states:
                oldActions = self.policy[state]
                value = []
                newActions = []
                for a in self.actions:
                    weight = 1 / len(self.policy[state])
                    for key in self.rewards:
                        (newState, reward, oldState, action) = key
                        if oldState == state and action == a:
                            value.append(weight * self.rewards[key]*(reward + self.gamma*self.values[newState]))
                            newActions.append(a)
                value = np.array(np.round(value,5))
                best = np.where(value == value.max())[0]
                bestActions = [newActions[item] for item in best]
                self.policy[state] = bestActions

                if oldActions != bestActions:
                    stable = False
                    
    def decideMove(self, state):
        return self.policy[state][random.randint(len(self.policy[state]))]

if __name__ == "__main__":
    game = GridWorld()
    print("\n_______Value Iterator______\n")
    test = ValueIterator(0.9)

    print("Reward before training: " + str(startValue(game, test)))

    test.train()
    print("Reward after training: " + str(startValue(game, test)))
    pprint(to2DList(test.values))
    pprint(to2DList(test.policy))

    print("\n_______Policy Iterator______\n")
    pIterator = PolicyIterator(0.9, 1e-2)

    pIterator.improvePolicy()
    pprint(to2DList(pIterator.values))
    pprint(to2DList(pIterator.policy))

    print("Reward after training: " + str(startValue(game, pIterator)))