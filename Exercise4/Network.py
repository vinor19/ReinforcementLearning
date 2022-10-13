import torch.nn as nn
import torch.nn.functional as F
HIDDEN_LAYER = 256  # NN hidden layer size
EPS_START = 0.9  # e-greedy threshold start value
EPS_END = 0.05  # e-greedy threshold end value
EPS_DECAY = 200  # e-greedy threshold decay

class Network(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(2, HIDDEN_LAYER)
        self.l2 = nn.Linear(HIDDEN_LAYER, 3)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x