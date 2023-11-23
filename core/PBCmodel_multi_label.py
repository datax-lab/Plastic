import torch
import torch.nn as nn
import torch.nn.functional as F

class PlasticModel(nn.Module):
    def __init__(self, kernel_size):
        super(PlasticModel, self).__init__()
        
        eps_value = 1e-01
        momentum_value = 0.99
        max_kernel_size = 1000 + 1
        
        self.kernel_size = kernel_size
        
        conv1 = nn.Conv1d(self.kernel_size, 128, kernel_size=4, stride=1)
        pool1 = nn.MaxPool1d(max_kernel_size - 4)

        conv2 = nn.Conv1d(self.kernel_size, 128, kernel_size=8, stride=1)
        pool2 = nn.MaxPool1d(max_kernel_size - 8)

        conv3 = nn.Conv1d(self.kernel_size, 128, kernel_size=16, stride=1)
        pool3 = nn.MaxPool1d(max_kernel_size - 16)
    
        batch1 = nn.BatchNorm1d(128, momentum = momentum_value, eps = eps_value)
        batch2 = nn.BatchNorm1d(128, momentum = momentum_value, eps = eps_value)
        batch3 = nn.BatchNorm1d(128, momentum = momentum_value, eps = eps_value)

        fc1 = nn.Linear(384, 512)
        fc2 = nn.Linear(512, 512)
        fc3 = nn.Linear(512, 14) 
        
        batch_fc1 = nn.BatchNorm1d(512, momentum = momentum_value, eps = eps_value)
        batch_fc2 = nn.BatchNorm1d(512, momentum = momentum_value, eps = eps_value)
        batch_fc3 = nn.BatchNorm1d(14, momentum = momentum_value, eps = eps_value)

        self.conv1_module = nn.Sequential(conv1, batch1, nn.ReLU(), pool1)
        self.conv2_module = nn.Sequential(conv2, batch2, nn.ReLU(), pool2)
        self.conv3_module = nn.Sequential(conv3, batch3, nn.ReLU(), pool3)
        
        self.fc_module = nn.Sequential(fc1, batch_fc1, nn.ReLU(), 
                                       fc2, batch_fc2, nn.ReLU(), 
                                       fc3, batch_fc3)

    def forward(self, x):
        out1 = self.conv1_module(x)
        out2 = self.conv2_module(x)
        out3 = self.conv3_module(x)
        out = torch.cat([out1, out2, out3], dim = 1)
        out = torch.flatten(out, 1)
        out = self.fc_module(out)
        return out

    def init_layer(self, layer):
        nn.init.xavier_normal_(layer.weight)
        layer.bias.data.fill_(0)
        return layer
    