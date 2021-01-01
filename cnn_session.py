#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 16:57:39 2019

@author: boyanglyu
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim




np.set_printoptions(threshold=1000)
device = "cuda" if torch.cuda.is_available() else "cpu"


def idx2onehot(idx, n):
    idx = idx.type(torch.LongTensor)
    assert torch.max(idx).item() < n
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n)
    onehot.scatter_(1, idx, 1)
    return onehot


class CNN_autoencoder(nn.Module):
    # input length 129, channel 8

    def __init__(self,
                 num_classes=4,
                 channel = 20):
        super(CNN_autoencoder, self).__init__()
        self.n_out = num_classes
        self.C = 20
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, self.C , kernel_size=(1,10)), #200, and no seconde layer
            nn.BatchNorm2d(self.C ),
            nn.ReLU(),
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(self.C, self.C , kernel_size=(1,5)),
            nn.BatchNorm2d(self.C ),
            nn.ReLU(),
            )
             #nn.MaxPool2d((1,2), (1,2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(self.C , 20, kernel_size=(channel, 1)),
            nn.BatchNorm2d(20 ),
            nn.ReLU())

        self.hidden0 = nn.Sequential(
                nn.Linear(940, 256), #940 for others
                nn.ReLU())
        self.out = nn.Sequential(
                torch.nn.Linear(256, self.n_out))  #torch.nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.dropout(out)
        
        out = self.layer2(out)
        out = self.dropout(out)
        
        out = self.layer3(out)
        out = self.dropout(out)
        
        out = out.view(out.size(0), -1)

        out = self.hidden0(out)

        out = self.out(out)
        return out


def test_model(model):
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        acc, std = accuracy(y_pred, y_test)
        # std

        loss = criterion(y_pred, y_test)
        return acc, loss, std
        
def accuracy(y_pred, y_test):
    
    _, predicted = torch.max(y_pred, 1)
    
    total = y_test.size(0)
    print(predicted.shape, total)
    correct = (predicted == y_test)
    
    if total/ 108 == 2:
        print('2 pred')
        part1 = correct[:108].sum().item() / 108
        part2 = correct[108:].sum().item() / 108
        std = np.std([part1, part2])
    elif total / 108 == 3:
        print('3 pred')
        part1 = correct[:108].sum().item()  / 108
        part2 = correct[108:216].sum().item()  / 108
        part3 = correct[216: ].sum().item() / 108
        std = np.std([part1, part2, part3])
    return correct.sum().item()/total, std
        

sub_num = '1'
# session data for training
sess = '1'

data = np.load('/home/boyang/AF/alignment/comparison_data/subject_' + sub_num + '_data_win_60.npy')
label=np.load('/home/boyang/AF/alignment/comparison_data/subject_' + sub_num + '_label_win_60.npy')


print(data.shape)
   

data_all = data.copy() 
label_all = label.copy()

if sess == '4':
    X_test = data_all[:324]
    X_train = data_all[324:]
    y_test = label_all[:324]
    y_train = label_all[324:]
elif sess == '1':
    X_test = data_all[108:]
    X_train = data_all[:108]
    y_test = label_all[108:]
    y_train = label_all[:108]
elif sess == '2':
    X_test_1 = data_all[:108]
    X_test_2 = data_all[216:]
    X_test = np.concatenate((X_test_1, X_test_2))

    y_test_1 = label_all[:108]
    y_test_2 = label_all[216:]
    y_test = np.concatenate((y_test_1, y_test_2))

    X_train = data_all[108:216]
    y_train = label_all[108:216]
elif sess == '3':
    X_test_1 = data_all[:216]
    X_test_2 = data_all[324:]
    X_test = np.concatenate((X_test_1, X_test_2))

    y_test_1 = label_all[:216]
    y_test_2 = label_all[324:]
    y_test = np.concatenate((y_test_1, y_test_2))
    X_train = data_all[216:324]
    y_train = label_all[216:324]





X_train = torch.from_numpy(X_train).type('torch.FloatTensor').unsqueeze(1).to(device)
X_test = torch.from_numpy(X_test).type('torch.FloatTensor').unsqueeze(1).to(device) #permute(0,1,3,2).
y_train = torch.from_numpy(y_train).type('torch.LongTensor').to(device)
y_test = torch.from_numpy(y_test).type('torch.LongTensor').to(device)

X_train = torch.cat((X_train[:,:,:20,:], X_train[:,:,20:,:]), dim=1)
X_test = torch.cat((X_test[:,:,:20,:], X_test[:,:,20:,:]), dim=1)

permutation = torch.randperm(X_train.size()[0])
X_train = X_train[permutation]
y_train = y_train[permutation]

num_epochs =300
# batch_size = 50
learning_rate = 1e-3
max_acc = []
max_std = []
for i in range(5):
    model = CNN_autoencoder().to(device) 
    criterion = nn.CrossEntropyLoss().to(device)  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    acc_list = []
    test_loss = []
    std_list = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        epoch_acc = 0
        model.train()

        optimizer.zero_grad()
        y_pred = model(X_train)

        loss = criterion(y_pred, y_train)

        loss.backward()
        optimizer.step()    

        running_loss = loss.item()

        print('running_loss is ', running_loss)
        running_loss = 0
        if epoch % 5 == 0:
            print('[%d] ' % (epoch + 1))  

            acc,one_test_loss, std = test_model(model)
            acc_list.append(acc)
            std_list.append(std)
            test_loss.append(one_test_loss.item())

            print('current test loss is ', one_test_loss.item())
            print('current acc is ', acc)
    max_acc.append(np.max(acc_list))
    max_std.append(std_list[np.argmax(acc_list)])
    
with open('comparison_by_session_cnn.txt', 'a+' ) as f:
    f.write('subject ' + sub_num + ' session ' + sess  + '\n')
    f.write('session by session accuracy is ' + str(np.mean(max_acc)) + '\n')
    f.write('session by session std is ' + str(np.sqrt(np.mean(np.square(max_std)))) + '\n')


