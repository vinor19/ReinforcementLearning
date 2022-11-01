import time

import matplotlib
from matplotlib import pyplot as plt
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
from ReplayMemory import ReplayMemory
path='model_scripted_des.pt'
matplotlib.use("TkAgg")

use_cuda = False
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

class DES():
    def __init__(self, env = "MountainCar-v0", N=1, LR = 0.001, GAMMA = 0.99, EPS_START = 0.9, EPS_END = 0.05, EPS_DECAY = 200, BATCH_SIZE = 64):
        self.env_name = env
        self.EPS_START = EPS_START  # e-greedy threshold start value
        self.EPS_END = EPS_END  # e-greedy threshold end value
        self.EPS_DECAY = EPS_DECAY  # e-greedy threshold decay
        self.GAMMA = GAMMA  # Q-learning discount factor
        self.LR = LR  # NN optimizer learning rate
        self.N=N
        self.BATCH_SIZE = BATCH_SIZE  # Q-learning batch size
        self.env = gym.make(env)
        if self.env.action_space.shape == ():
            self.action_space_size = self.env.action_space.n
            self.available_actions = range(self.env.action_space.n)
        else:
            self.action_low = self.env.action_space.low[0]
            self.action_high = self.env.action_space.high[0]
            self.action_space_size = 8*self.env.action_space.shape[0]+1
            self.available_actions = [self.action_low]*self.action_space_size
            for i in range(self.action_space_size):
                self.available_actions[i] +=  abs(self.action_low-self.action_high)/(self.action_space_size-1) * i
        self.model = Network(self.env.observation_space.shape[0], self.action_space_size)
        if use_cuda:
            self.model.cuda()
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), self.LR)
        self.steps_done = 0
        

    def select_action(self, state):
        self.steps_done
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1

        if sample > eps_threshold:
            if self.env.action_space.shape == ():
                return IntTensor([[self.available_actions[self.model(Variable(state).type(FloatTensor)).data.max(1)[1].view(1, 1).item()]]])
            else:
                return self.model(Variable(state).type(FloatTensor)).data.max(1)[1].view(1, 1)
        else:
            return LongTensor([[random.randrange(self.action_space_size-1)]])  # return env.action_space.sample()

    def train(self, episodes = 300):
        rewardTracker = []
        for e in range(episodes):
            rewardTracker.append(self.run_episode(e))
            if e %100 == 0 and e != 0:
                # self.env.render()
                print("{2} Episode {0} finished after {1} steps"
                    .format(e, rewardTracker[e]*-1, '\033[92m' if rewardTracker[e]*-1 >= 195 else '\033[99m'))
                print('Average reward for 100 episode= {}'.format(sum(rewardTracker[e-100:e]) / 100))
        return rewardTracker

    def run_episode(self, e):
        episodeSum = 0
        G = 0
        state = self.env.reset()
        steps = 0
        
        rewardSum = 0
        while True:
            # self.env.render()
            for i in range(self.N):
                action = self.select_action(FloatTensor([state])).long()
                if self.env.action_space.shape == ():
                    applied_action = action[0, 0].item()
                else:
                    applied_action = [self.available_actions[action[0, 0].item()]]
                if i == 0:
                    stateOld = state
                    actionOld = action
                state, reward, done, _ = self.env.step(applied_action)
                
                steps += 1
                G += reward
                rewardSum +=  np.power(self.GAMMA,i) * reward
                if done:
                    break

            self.memory.push((FloatTensor([stateOld]),
                        actionOld,  # action is already a tensor
                        FloatTensor([rewardSum]),
                        FloatTensor([state])
                        ))

            rewardSum = 0

            self.learn()

            if done:
                # self.env.render()
                # print('\033[92m',steps)
                episodeSum += G
                return G
    
    def probability_distribution(self, q_values):
        optimal = q_values.data.max(1)[1]
        eps = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * (self.steps_done) / self.EPS_DECAY)
        result = []

        for i in range(self.BATCH_SIZE):
            for j in range(self.action_space_size):
                if optimal[i].item() == j:
                    result.append(1 - eps + eps/(self.action_space_size))
                else:
                    result.append(eps/(self.action_space_size))
        return [result[i*self.action_space_size:i*self.action_space_size+self.action_space_size] for i in range(self.BATCH_SIZE)]
    
    def learn(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        # random transition batch is taken from experience replay memory
        transitions = self.memory.sample(self.BATCH_SIZE)

        batch_state, batch_action, batch_reward, batch_state_new = zip(*transitions)

        batch_state = Variable(torch.cat(batch_state))
        batch_action = Variable(torch.cat(batch_action))
        batch_reward = Variable(torch.cat(batch_reward))
        batch_state_new = Variable(torch.cat(batch_state_new))

        # current Q values are estimated by NN for all actions
        current_q_values = self.model(batch_state)

        # print(batch_action)
        current_q_values = current_q_values.gather(1, batch_action)

        new_q_values = self.model(batch_state_new).detach()
        # print(new_q_values)
        prob = Tensor(self.probability_distribution(new_q_values)).detach()

        # next Q values are estimated by NN for all next actions
        expected_values1 = prob * new_q_values
        # print("Before summing",expected_values)
        expected_values = np.power(self.GAMMA,self.N) * torch.sum(expected_values1.detach(),dim=1)

        q_values = (batch_reward + expected_values).view(-1,1)
        # loss is measured from error between current and newly expected Q values
        loss = F.smooth_l1_loss(current_q_values, q_values)

        # backpropagation of loss to NN
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

colors = ['blue','red','green']

if __name__ == '__main__':
    random.seed(1)
    start_time = time.time()
    episodes = 50
    tests = 5
    env_being_tested = ["MountainCar-v0","Acrobot-v1","Pendulum-v1"]
    tmp = gym.make(env_being_tested[2])
    # print("This right here",tmp.action_space.sample())
    # env_being_tested = "MountainCar-v0"
    ai_list = [
        (env_being_tested[1],1,0.001,0.995,0.9,0.05,200,64),
        (env_being_tested[1],3,0.001,0.995,0.9,0.05,200,64),
        (env_being_tested[1],5,0.001,0.995,0.9,0.05,200,64)
    ]
    ai_learncurve = []
    for j in range(len(ai_list)):
        for i in range(tests):
            ai = DES(*ai_list[j])
            ai_learncurve.append(ai.train(episodes))
            print("Intermediary Done")
            # print(ai_learncurve)

        print("Done",j)

    _, ax = plt.subplots()

    for i in range(len(ai_list)):
        # print(ai_learncurve[i*tests:i*tests+tests])
        mean = np.array(ai_learncurve[i*tests:i*tests+tests]).mean(axis=0)
        # print(mean)
        std = np.array(ai_learncurve[i*tests:i*tests+tests]).std(axis=0)/np.sqrt(tests)
        ax.plot(range(0,episodes),mean, color=colors[i])
        ax.fill_between(range(0,episodes),mean+std, mean-std, facecolor=colors[i], alpha=0.2)

    plt.xlabel("Epochs trained")
    plt.ylabel("Costs")
    plt.title("Training methods")
    ax.legend(["DES_N1","DES_N3","DES_N5"])
    print("--- %s seconds ---" % (time.time() - start_time))
    plt.savefig('learning-curve.png')
    plt.show()