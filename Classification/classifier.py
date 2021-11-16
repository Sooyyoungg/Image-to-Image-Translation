import torch.cuda
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from data_loader import data_loader
from CNN import CNN

### Environment setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print("current cuda device:", torch.cuda.current_device())
print("the number of gpu I can use:", torch.cuda.device_count())

### hyperparameters
lr = 0.001
training_epoch = 30
batch_size = 16

### Model
print("setting model start")
CNN = CNN().to(device)
parameters = CNN.parameters()
optimizer = torch.optim.Adam(parameters, lr=lr)
criterion = torch.nn.MSELoss().to(device)

### Data Loader
print("data loading start")
train_x, train_y, test_x, test_y = data_loader("/home/connectome/conmaster/GANBERT/abcd_t1_t2_diff_info.csv")
print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)

train_data = TensorDataset(train_x, train_y)
test_data = TensorDataset(test_x, test_y)

Train_data = DataLoader(train_data, batch_size, shuffle=True)
Test_data = DataLoader(test_data, batch_size, shuffle=True)

train_batch_size = len(Train_data)
test_batch_size = len(Test_data)
print(train_batch_size, test_batch_size)

### Training
print("training start")
for epoch in range(training_epoch):
    avg_loss = 0
    for X, Y in Train_data:
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        Y_predict = CNN(X)
        loss = criterion(Y, Y_predict)
        loss.bckward()
        optimizer.step()

        avg_loss += loss / train_batch_size

    print("[Epoch: {:>4}] nLoss = {:>.9}".format(epoch + 1, avg_loss))

### Testing