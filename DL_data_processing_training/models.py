#!/usr/bin/python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, functional as F, Linear, MaxPool2d, Module
import numpy as np
import pandas as pd
import os
import glob
import pdb
import yaml


os.chdir('/raid/togzhan_syrymova/lump_project/scripts/spectrogram/')
config = yaml.safe_load(open("config.yaml"))
print(config)
learning_rate = config['learning_rate']
channels_in = config['channels_in']
D_out = config['D_out']
# D_in = config['D_in']
H = config['hidden_size']
kernel_size = config['kernel_size']
stride = config['stride']
drop = config['dropout'] 
##

class SeRNN_FWXX(nn.Module):
    """Recurrent neural network"""
    def __init__(self, batch_size, device, channels_in, D_out, H):
        super(SeRNN_FWXX, self).__init__()

        self.input_size = channels_in    # 6 input channels
        self.num_outputs = D_out   # controls for two motors
        self.batch = batch_size
        self.device = device

        # model
        self.num_layers = 1
        self.hidden_size =  H
        self.lstm = nn.ModuleList([
            nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first = True)
         for i in range(15)])
        self.fc1 = nn.Linear(self.hidden_size*15, 64*4)
        
        self.fc3 = nn.Linear(64*4, 64)
        
        self.fc2 = nn.Linear(64, self.num_outputs)
        #self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        ii = 0
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        
        h_relu = torch.FloatTensor([]).to(self.device)
        for layers in self.lstm:
            layers.flatten_parameters()
            x_data, _  = layers(torch.unsqueeze(x[:, :, ii], 2), (h0, c0))
            h_relu = torch.cat((h_relu, x_data), dim=2)
            ii = ii+1

        h_relu = F.relu(self.fc1(h_relu[:, -1, :]))
        h_relu = F.relu(self.fc3(h_relu))
        h_relu = (self.fc2(h_relu))
        
        return h_relu
    
class RNN_Model(nn.Module):
    def __init__(self, batch_size, device, channels_in, D_out, H): #input_size, output_size, hidden_dim, n_layers):
        super(RNN_Model, self).__init__()

        # Defining some parameters
        self.hidden_dim = H
        self.n_layers = 15
        self.batch_size = batch_size
        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(15, self.hidden_dim, self.n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, D_out)
    
    def forward(self, x):            
        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(self.batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden
    
class ConvLayerNet3(torch.nn.Module):
    def __init__(self,  batch_size, device, channels_in, D_in, H):
        super(ConvLayerNet3, self).__init__()
        #(self,  channels_in, channels_out, D_in, H, D_out, device, drop=0.0, kernel_size=1, stride=1
        self.device = device
        self.channels_out = 1
        self.kernel_size = kernel_size
        self.stride = stride
        self.D_in = D_in
        self.drop = drop
        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv1d(1, self.channels_out, kernel_size=self.kernel_size, stride=self.stride),
            nn.Conv1d(1, self.channels_out, kernel_size=self.kernel_size, stride=self.stride),
        ) for i in range(15)])
                                    
        W_out = ((self.D_in - self.kernel_size)//self.stride + 1)*self.channels_out
        W_out = ((W_out - self.kernel_size)//self.stride + 1)*15*self.channels_out
        W_out = np.int64(W_out) 
        H3 = H//2
        self.linear1_1 = torch.nn.Linear(W_out, H)
        self.linear1_2 = torch.nn.Linear(H, H3)
        self.linear1_3 = torch.nn.Linear(H3, 32)
        self.linear2 = torch.nn.Linear(32, 1)
        
        self.relu = torch.nn.ReLU()
        self.drop1   = torch.nn.Dropout(p=self.drop)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        ii = 0
        h_relu = torch.FloatTensor([]).to(self.device)
        for layers in self.convs:
            x_data = layers(torch.unsqueeze(x[:, ii, :], 1))
            x_data = self.relu(x_data)
            b, h, w = x_data.size()
            x_data = x_data.view(b,-1)
            h_relu = torch.cat([h_relu, x_data], axis = 1)
            ii = ii+1
        h_relu = self.drop1(self.linear1_1(h_relu))#.clamp(min=0)#.clamp(min=0)
        h_relu = self.relu(h_relu)
        h_relu = self.drop1(self.linear1_2(h_relu))#.clamp(min=0)#.clamp(min=0)
        h_relu = self.relu(h_relu)
        h_relu = self.drop1(self.linear1_3(h_relu))#.clamp(min=0)#.clamp(min=0)
        h_relu = self.relu(h_relu)
        y_pred = (self.linear2(h_relu))#
        
        return y_pred

class ConvLayerNet4(torch.nn.Module):
    def __init__(self,  batch_size, device, channels_in, D_in, H):
        super(ConvLayerNet4, self).__init__()
        #(self,  channels_in, channels_out, D_in, H, D_out, device, drop=0.0, kernel_size=1, stride=1
        self.device = device
        self.channels_out = 1
        self.kernel_size = kernel_size
        self.stride = stride
        self.D_in = D_in
        self.drop = drop
        self.convs = nn.Conv1d(15, 15, kernel_size=self.kernel_size, stride=self.stride)
        self.convs = nn.Conv1d(15, self.channels_out, kernel_size=self.kernel_size, stride=self.stride)  
        
        W_out = ((self.D_in - self.kernel_size)//self.stride + 1)*self.channels_out
        W_out = ((W_out - self.kernel_size)//self.stride + 1)*15*self.channels_out
        W_out = np.int64(W_out) 
        H3 = H//2
        self.linear1_1 = torch.nn.Linear(W_out, H)
        self.linear1_2 = torch.nn.Linear(H, H3)
        self.linear1_3 = torch.nn.Linear(H3, 32)
        self.linear2 = torch.nn.Linear(32, 1)
        
        self.relu = torch.nn.ReLU()
        self.drop1   = torch.nn.Dropout(p=self.drop)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        ii = 0
        h_relu = torch.FloatTensor([]).to(self.device)
        for layers in self.convs:
            x_data = layers(torch.unsqueeze(x[:, ii, :], 1))
            x_data = self.relu(x_data)
            b, h, w = x_data.size()
            x_data = x_data.view(b,-1)
            h_relu = torch.cat([h_relu, x_data], axis = 1)
            ii = ii+1
        h_relu = self.drop1(self.linear1_1(h_relu))#.clamp(min=0)#.clamp(min=0)
        h_relu = self.relu(h_relu)
        h_relu = self.drop1(self.linear1_2(h_relu))#.clamp(min=0)#.clamp(min=0)
        h_relu = self.relu(h_relu)
        h_relu = self.drop1(self.linear1_3(h_relu))#.clamp(min=0)#.clamp(min=0)
        h_relu = self.relu(h_relu)
        y_pred = (self.linear2(h_relu))#
        
        return y_pred    
    
class ConvLayerNet(torch.nn.Module):

    def __init__(self,  batch_size, device, channels_in, D_in, H):
        super(ConvLayerNet, self).__init__()
        self.device = device
        self.channels_out = 1
        self.kernel_size = kernel_size
        self.stride = stride
        self.D_in = D_in
        self.drop = drop
        self.convs = nn.ModuleList([
            nn.Conv2d(1, channels_out, kernel_size=kernel_size, stride=stride)
        for i in range(15)])
        
        W_out = ((D_in - kernel_size)//stride + 1)*channels_out
        self.pool = MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.linear1_1 = torch.nn.Linear(H3, 64)
        self.linear2 = torch.nn.Linear(64, D_out)
        self.fc1 = Linear(18 * 16 * 16, 64)
        self.fc2 = Linear(64, 1)

    def forward(self, x):
        ii = 0
        h_relu = torch.FloatTensor([]).to(self.device)
        for layers in self.convs:
            print(np.shape(torch.unsqueeze(x[:, ii, :], 2)))
            x_data = layers(torch.unsqueeze(x[:, ii, :], 2)) # torch.unsqueeze(x[:, ii, :, :], 1
            print(np.shape(x_data))
            x_data = self.relu(x_data)
            print(x_data.size())
            b, h, w = x_data.size()
            x_data = x_data.view(b,-1)
            h_relu = torch.cat([h_relu, x_data], axis = 1)
            ii = ii+1
            
        x = F.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(-1, 18 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
    
    
# class SeRNN_FWXX2(nn.Module):
#     """Recurrent neural network"""
#     def __init__(self, batchSize, device):
#         super(SeRNN_FWXX2, self).__init__()

#         self.input_size = 1    # 6 input channels
#         self.num_outputs = 1   # controls for two motors
#         self.batch = batchSize
#         self.device = device
#         self.channels_out = 1
#         # model
#         self.num_layers = 1
#         self.hidden_size = 32
#         self.conv = nn.ModuleList([
#             nn.Conv2d(1, self.channels_out, kernel_size=3, stride=2),
#          for i in range(15)])
        
#         self.lstm = nn.ModuleList([
#             nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first = True)
#          for i in range(15)])
#         self.fc1 = nn.Linear(self.hidden_size*15, 64)
#         self.fc3 = nn.Linear(64, 64)
#         self.fc2 = nn.Linear(64, self.num_outputs)
#         #self.dropout = nn.Dropout(p=0.2)

#     def forward(self, x):
#         ii = 0
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

#         h_relu = torch.FloatTensor([]).to(self.device)
#         for layers in self.lstm:
#             layers.flatten_parameters()
#             x_data, _  = layers(torch.unsqueeze(x[:, :, ii], 2), (h0, c0))
#             h_relu = torch.cat((h_relu, x_data), dim=2)
#             ii = ii+1

#         h_relu = F.relu(self.fc1(h_relu[:, -1, :]))
#         h_relu = self.fc3(h_relu)
#         h_relu = self.fc2(h_relu)
        
#         return h_relu
