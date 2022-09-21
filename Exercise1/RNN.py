import torch
import torch.nn as nn
import scipy.io
import numpy as np
import numpy.random
import torch.utils.data

fulldata = scipy.io.loadmat("mixoutALL_shifted.mat")

data = fulldata["mixout"][0]

maxl = max(map(lambda d: len(d[0]),data))

input = np.array([np.hstack([x, np.zeros([3, maxl - x[0].size])]) for x in [np.array([np.array(x) for x in d]) for d in data]])

const = [x[0] for x in fulldata["consts"]["key"][0][0][0]]



input = list(zip(torch.FloatTensor(input),np.array([x-1 for x in fulldata["consts"]["charlabels"][0][0][0]])))
np.random.shuffle(input)

train = input[0:1500]
test = input[1499:-1]

batch_size = 100
n_iters = 3000
num_epochs = n_iters / (len(train) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test, 
                                          batch_size=batch_size, 
                                          shuffle=False)


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your RNN
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, input_dim)
        # batch_dim = number of samples per batch
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        # (layer_dim, batch_size, hidden_dim)
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach the hidden state to prevent exploding/vanishing gradients
        # This is part of truncated backpropagation through time (BPTT)
        out, hn = self.rnn(x, h0.detach())

        # Index hidden state of last time step
        # out.size() --> 100, 3, max
        # out[:, -1, :] --> 100, 20 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 20
        return out

input_dim = maxl
hidden_dim = maxl
layer_dim = 1
output_dim = 20

model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)

criterion = nn.CrossEntropyLoss()

learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

# print(len(list(model.parameters())))
# print(list(model.parameters())[4].size())

# Number of steps to unroll
# seq_dim = 3
iter = 0
for epoch in range(num_epochs):
    for i, (data, labels) in enumerate(train_loader):
        # images = images.view(-1, seq_dim, input_dim).requires_grad_()
        model.train()
        # Load images as tensors with gradient accumulation abilities

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        # outputs.size() --> 100, 20
        outputs = model(data)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        iter += 1

        if iter % 500 == 0:
            model.eval()
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for data, labels in test_loader:
                # Load images to a Torch tensors with gradient accumulation abilities
                # images = images.view(-1, seq_dim, input_dim)

                # Forward pass only to get logits/output
                outputs = model(data)

                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)

                # Total number of labels
                total += labels.size(0)

                # Total correct predictions
                correct += (predicted == labels).sum()

            accuracy = 100 * correct / total

            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))


errors = 0
for i in range(0,len(test)):
    output = model(torch.FloatTensor(test[i][0]).view(-1, 3, input_dim))
    if (const[test[i][1]] != const[torch.max(output.data,1)[1]]):
        print(const[test[i][1]] )
        print(const[torch.max(output.data,1)[1]])
        errors+=1
print(errors)