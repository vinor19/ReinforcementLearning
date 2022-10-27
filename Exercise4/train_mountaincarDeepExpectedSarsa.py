import time
time.clock = time.time
import gym
import numpy as np
#from gym import wrappers
import random
import math
import torch
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from Network import Network
import torch.nn.functional as F

EPS_START = 0.9  # e-greedy threshold start value
EPS_END = 0.05  # e-greedy threshold end value
EPS_DECAY = 200  # e-greedy threshold decay
EPISODES = 300  # number of episodes
GAMMA = 0.995  # Q-learning discount factor
LR = 0.001  # NN optimizer learning rate
N=3

BATCH_SIZE = 64  # Q-learning batch size

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
path='model_scripted.pt'

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def lastsamples(self, batch_size):
        return self.memory[-batch_size:]

    def __len__(self):
        return len(self.memory)


#env = gym.make("MountainCar-v0")
env = gym.make("MountainCar-v0").env
model = Network()
if use_cuda:
    model.cuda()
memory = ReplayMemory(100000)
optimizer = optim.Adam(model.parameters(), LR)
steps_done = 0
episode_durations = []


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if sample > eps_threshold:
        return model(Variable(state).type(FloatTensor)).data.max(1)[1].view(1, 1)
    else:
        return LongTensor([[random.randrange(2)]])  # return env.action_space.sample()

# print( env.action_space.n)
# print(env.observation_space.shape[0])

def run_episode(e, environment):
    rewardTracker = []
    episodeSum = 0
    G = 0
    state = environment.reset()
    steps = 0
    

    stateOld = state
    actionOld = select_action(FloatTensor([state]))
    rewardSum = 0
    n = random.randint(1,N)
    while True:
        # env.render()
        action = actionOld
        state = stateOld
        for i in range(n):
            statePrev = state
            state, reward, done, _ = environment.step(action[0, 0].item())
            action = select_action(FloatTensor([state]))
            G += reward
            rewardSum += np.power(reward,i-1)
            steps += 1

            if done:
                break

        newQ = model(FloatTensor([state]))[0]
        # negative reward when attempt ends
        if done:
            reward = -1
            G += reward
            rewardSum += reward
            newQ = FloatTensor([[0]]*env.action_space.n)

        sum = FloatTensor([0])
        prevQ = F.softmax(-1*model(FloatTensor([statePrev]))[0],dim=0)
        for i in range(env.action_space.n):
            sum += prevQ[i]*(newQ[i])

        newQ = sum

        n=N
        memory.push((FloatTensor([stateOld]),
                    actionOld,  # action is already a tensor
                    FloatTensor([rewardSum]),
                    newQ
                    ))
        stateOld = state
        actionOld = action
        rewardSum = 0


        learn()

        if done:
            print(steps)
            episodeSum += G
            rewardTracker.append(G)
            episode_durations.append(steps)
            if e %100 == 0:
            #   env.render()
              print("{2} Episode {0} finished after {1} steps"
                   .format(e, steps, '\033[92m' if steps >= 195 else '\033[99m'))
              print('Average reward for 100 episode= {}'.format(episodeSum / 100))

              episodeSum = 0
              break
            
            # plot_durations()
            break

def learn():
    if len(memory) < BATCH_SIZE:
        return

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(BATCH_SIZE)

    batch_state, batch_action, batch_reward, QValue = zip(*transitions)

    batch_state = Variable(torch.cat(batch_state))
    batch_action = Variable(torch.cat(batch_action))
    batch_reward = Variable(torch.cat(batch_reward))
    next_q_values = Variable(torch.cat(QValue))

    # current Q values are estimated by NN for all actions
    current_q_values = model(batch_state).gather(1, batch_action)
    # next Q values are estimated by NN for all next actions
    sum = 0
    

    q_values = batch_reward + (np.power(GAMMA,N) * next_q_values )

    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(current_q_values, q_values.view(-1,1))

    # backpropagation of loss to NN
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


for e in range(EPISODES):


    run_episode(e, env)

            


print('Complete')

path='model_scripted.pt'
torch.save(model.state_dict(), path)
