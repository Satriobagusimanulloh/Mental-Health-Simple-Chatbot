import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.linear_ih = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.linear_hh = nn.Linear(in_features=hidden_size, out_features= hidden_size)
        self.linear_ho = nn.Linear(in_features=hidden_size, out_features=num_classes)
        self.relu = nn.ReLU()

    def forward(self, X):
        X = self.linear_ih(X)
        X = self.relu(X)
        X = self.linear_hh(X)
        X = self.relu(X)
        X = self.linear_ho(X)

        return X