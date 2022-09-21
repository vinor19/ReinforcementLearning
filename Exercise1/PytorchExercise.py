from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import scipy.io
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as dsets

dp = 0

fulldata = scipy.io.loadmat("mixoutALL_shifted.mat")
data = fulldata["mixout"][0]
input = [np.array([np.array(x) for x in d]) for d in data]
input = np.array([np.hstack([x, np.zeros([3, 205 - x[0].size])]) for x in input])
large = 0
for i in range(0,len(input)):
    input[i] = np.hstack([input[i], np.zeros([3, 205 - input[i][0].size])])
inputTensor = torch.FloatTensor(np.array(input))



# rnn = nn.RNN(input_size=205, hidden_size = 3, num_layers=2)
# output, hn = rnn(inputTensor)
# print(output)
const = [x[0] for x in fulldata["consts"]["key"][0][0][0]]
print(const)
print(const[fulldata["consts"]["charlabels"][0][0][0][dp]-1])
x = []
y = []
for i in range(0,205):
    x = x+[sum(input[dp][0][:i])]
    y = y+[sum(input[dp][1][:i])]
plt.plot(x,y, c = "blue")
plt.show()


import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

rnn = RNN(205, 205, output_size = 20)