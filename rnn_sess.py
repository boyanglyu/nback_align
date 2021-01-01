import torch 
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

import torch.optim as optim



    

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # input data is (batch, seq, feature)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2) 
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Set initial hidden and cell states 
#         print('x size',len(x))
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
#         print(out.shape)
#         print(out[:, -1, :].shape)
        #out = self.fc(out.view(len(x), -1))
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

def idx2onehot(idx, n):
    idx = idx.type(torch.LongTensor)
    assert torch.max(idx).item() < n
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n)
    onehot.scatter_(1, idx, 1)
    return onehot

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
    

sub_num = '5'
sess = '4'

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




X_train = torch.from_numpy(X_train).type('torch.FloatTensor').permute(0,2,1).to(device)
X_test = torch.from_numpy(X_test).type('torch.FloatTensor').permute(0,2,1).to(device) #permute(0,1,3,2).
y_train = torch.from_numpy(y_train).type('torch.LongTensor').to(device)
y_test = torch.from_numpy(y_test).type('torch.LongTensor').to(device)


permutation = torch.randperm(X_train.size()[0])
X_train = X_train[permutation]
y_train = y_train[permutation]


input_size = 40
hidden_size = 20
num_layers = 3
num_classes = 4

learning_rate = 5e-4
window_size = 60
# tau = 1.3
num_epochs =400
max_acc = []
max_std = []

for i in range(5):
    model = RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes).to(device) 

    print(model)
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
            test_loss.append(one_test_loss.item())
            std_list.append(std)

            print('current test loss is ', one_test_loss.item())
            print('current acc is ', acc)
    max_acc.append(np.max(acc_list))
    max_std.append(std_list[np.argmax(acc_list)])
    
with open('comparison_by_session_rnn.txt', 'a+' ) as f:
    f.write('\n')
    f.write('subject ' + sub_num + ' session ' + sess  + '\n')
    f.write(str(max_acc) + '\n')
    f.write(str(max_std) + '\n')
    
    f.write('subject ' + sub_num + ' session ' + sess  + '\n')
    f.write('session by session accuracy is ' + str(np.mean(max_acc)) + '\n')
    f.write('session by session std is ' + str(np.sqrt(np.mean(np.square(max_std)))) + '\n')



    
# for ele in file_prefix[0:4]:
#     data = np.load(save_path +  ele + '_'+ str(window_size) + '_brake_data.npy')
#     label = np.load(save_path +  ele + '_'+ str(window_size) + '_brake_label.npy')
#     tau = np.median(label)
#     label[label > tau] = 0
#     label[label!= 0] = 1
#     for i in range(len(sess_list)):
#         X_train, X_test, y_train, y_test, weight = prepare_train_test(data[:,:,44:], label,sess_list[i])
#         criterion = nn.BCEWithLogitsLoss().to(device) 
#         model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
#         optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#         model = train_model(model, X_train, y_train, X_test, y_test, num_epochs)
#         model.eval()
#         acc,one_test_loss = test_model(model, X_test,y_test)
#         acc_dict[ele].append(acc)
#     with open('separate_sess_rnn.txt', 'a+') as f:
#         f.write('learning_rate ' + str(learning_rate)+'\n') 
#         f.write('subject ' + ele +'\n') 
#         f.write('Accuracy is '+ str(acc_dict[ele])+ '\n')         
#         f.write('Average Accuracy is '+ str(np.average(acc_dict[ele]))+ '\n')         

# with open('separate_sess_rnn.txt', 'a+') as f:
#     f.write('Average Accuracy is '+ str(np.average(acc_dict.values()))+ '\n')         
    
    
    
    